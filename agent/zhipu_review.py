"""
zhipu_review.py
~~~~~~~~~~~~~~~~
使用 智谱AI GLM-4V-Flash 对候选股票进行图表分析评分。
继承自 BaseReviewer 基础架构。

用法：
    python agent/zhipu_review.py
    python agent/zhipu_review.py --config config/zhipu_review.yaml

配置：
    默认读取 config/zhipu_review.yaml。

环境变量：
    ZHIPU_API_KEY  —— 智谱AI API Key（必填）

输出：
    ./data/review/{pick_date}/{code}.json   每支股票的评分 JSON
    ./data/review/{pick_date}/suggestion.json  汇总推荐建议
"""

import argparse
import os
import sys
import base64
from pathlib import Path
from typing import Any

import yaml
import requests  # 智谱官方推荐用 HTTP 调用多模态

from base_reviewer import BaseReviewer

# ────────────────────────────────────────────────
# 配置加载
# ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "zhipu_review.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    # 路径参数（相对路径默认基于项目根目录）
    "candidates": "data/candidates/candidates_latest.json",
    "kline_dir": "data/kline",
    "output_dir": "data/review",
    "prompt_path": "agent/prompt.md",
    # 智谱模型参数
    "model": "glm-4v-flash",  # 最强免费多模态
    "request_delay": 5,
    "skip_existing": True,
    "suggest_min_score": 4.0,
}


def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _ROOT) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    cfg_path = config_path or _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = {**DEFAULT_CONFIG, **raw}

    # BaseReviewer 依赖这些路径字段为 Path 对象
    cfg["candidates"] = _resolve_cfg_path(cfg["candidates"])
    cfg["kline_dir"] = _resolve_cfg_path(cfg["kline_dir"])
    cfg["output_dir"] = _resolve_cfg_path(cfg["output_dir"])
    cfg["prompt_path"] = _resolve_cfg_path(cfg["prompt_path"])

    return cfg


class ZhipuReviewer(BaseReviewer):
    def __init__(self, config):
        super().__init__(config)
        
        # 智谱 API Key
        # api_key = os.environ.get("ZHIPU_API_KEY", "")
        api_key = "8898c4ad05e64c58b3669beebabe1014.GgDcUhtAT3PFYcgz"
        if not api_key:
            print("[ERROR] 未找到环境变量 ZHIPU_API_KEY，请先设置后重试。", file=sys.stderr)
            sys.exit(1)
        
        self.api_key = api_key
        self.model = config.get("model", "glm-4v-flash")
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    @staticmethod
    def image_to_base64(path: Path) -> str:
        """图片转 base64（智谱多模态必须）"""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def review_stock(self, code: str, day_chart: Path, prompt: str) -> dict:
        """
        调用 智谱 GLM-4V-Flash API，对单支股票进行图表分析，返回解析后的 JSON 结果。
        """
        # 图片 base64
        img_b64 = self.image_to_base64(day_chart)
        img_url = f"data:image/png;base64,{img_b64}"

        # 构造请求体（智谱标准多模态格式）
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"股票代码：{code}"},
                    {"type": "image_url", "image_url": {"url": img_url}}
                ]
            }
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        # 发送请求
        resp = requests.post(self.url, headers=headers, json=data, timeout=120)

        if resp.status_code != 200:
            raise RuntimeError(f"智谱API调用失败：{resp.status_code} {resp.text}")

        result = resp.json()
        response_text = result["choices"][0]["message"]["content"]

        # 提取 JSON（复用你原来的方法）
        json_result = self.extract_json(response_text)
        json_result["code"] = code
        return json_result


def main():
    parser = argparse.ArgumentParser(description="智谱AI 图表复评")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help="配置文件路径（默认 config/zhipu_review.yaml）",
    )
    parser.add_argument("--candidates", default=None,
                        help="候选文件路径（覆盖配置中的 candidates）")
    parser.add_argument("--output-dir", default=None,
                        help="评分输出目录（覆盖配置中的 output_dir）")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.candidates:
        config["candidates"] = _resolve_cfg_path(args.candidates)
    if args.output_dir:
        config["output_dir"] = _resolve_cfg_path(args.output_dir)
    reviewer = ZhipuReviewer(config)
    reviewer.run()


if __name__ == "__main__":
    main()