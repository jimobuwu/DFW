"""
zhipu_news_review.py
~~~~~~~~~~~~~~~~
使用 智谱AI GLM-4-Flash（文本版）获取推荐股票的消息面信息，
并更新汇总推荐文件 suggestion.json。

用法：
    python agent/zhipu_news_review.py --config config/zhipu_review.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml

from base_reviewer import BaseReviewer

# 复用配置解析逻辑
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "zhipu_review.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "candidates": "data/candidates/candidates_latest.json",
    "output_dir": "data/review",
    "prompt_news_path": "agent/prompt_news.md",  # 消息面分析专用prompt
    "model": "glm-4-flash",  # 文本版GLM-4-Flash
    "request_delay": 5,
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
    cfg["candidates"] = _resolve_cfg_path(cfg["candidates"])
    cfg["output_dir"] = _resolve_cfg_path(cfg["output_dir"])
    cfg["prompt_news_path"] = _resolve_cfg_path(cfg.get("prompt_news_path", "agent/prompt_news.md"))

    return cfg


class ZhipuNewsReviewer(BaseReviewer):
    def __init__(self, config):
        super().__init__(config)
        
        # 智谱API Key（复用原有key）
        api_key = os.environ.get("ZHIPU_API_KEY", "")
        if not api_key:
            print("[ERROR] 未找到环境变量 ZHIPU_API_KEY，请先设置后重试。", file=sys.stderr)
            sys.exit(1)
        
        self.api_key = api_key
        self.model = config.get("model", "glm-4-flash")
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        # 加载消息面分析专用prompt
        self.news_prompt = self.load_prompt(Path(config["prompt_news_path"]))

    def get_news_analysis(self, code: str, prompt: str) -> dict:
        """
        调用GLM-4-Flash文本模型，获取单支股票的消息面分析
        """
        # 构造纯文本请求体（无图片）
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"请分析股票代码 {code} 的最新消息面情况，包括但不限于：\n"
                            "1. 公司最新公告、业绩预告、重大事项\n"
                            "2. 所属行业政策、行业动态\n"
                            "3. 市场舆情、资金流向相关消息\n"
                            "4. 潜在利好/利空因素\n"
                            "要求输出JSON格式，包含字段：news_score（消息面评分0-5）、news_summary（消息面总结）、"
                            "positive_factors（利好因素）、negative_factors（利空因素）"
            }
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "response_format": {"type": "json_object"}  # 强制返回JSON
        }

        # 发送请求
        resp = requests.post(self.url, headers=headers, json=data, timeout=120)

        if resp.status_code != 200:
            raise RuntimeError(f"智谱API调用失败：{resp.status_code} {resp.text}")

        result = resp.json()
        response_text = result["choices"][0]["message"]["content"]
        
        # 解析JSON结果
        json_result = self.extract_json(response_text)
        json_result["code"] = code
        return json_result

    def update_suggestion_with_news(self):
        """
        核心逻辑：
        1. 读取候选股票日期和原suggestion.json
        2. 为推荐股票获取消息面信息
        3. 更新suggestion.json，添加消息面字段
        """
        # 1. 加载候选数据获取日期
        candidates_data = self.load_candidates(Path(self.config["candidates"]))
        pick_date: str = candidates_data["pick_date"]
        out_dir = self.output_dir / pick_date
        
        # 2. 读取原推荐文件
        suggestion_file = out_dir / "suggestion.json"
        if not suggestion_file.exists():
            print(f"[ERROR] 未找到推荐文件：{suggestion_file}", file=sys.stderr)
            sys.exit(1)
        
        with open(suggestion_file, "r", encoding="utf-8") as f:
            suggestion_data = json.load(f)
        
        # 3. 遍历推荐股票，获取消息面信息
        recommendations: List[Dict] = suggestion_data["recommendations"]
        updated_recommendations = []
        
        for i, rec in enumerate(recommendations, 1):
            code = rec["code"]
            print(f"[{i}/{len(recommendations)}] 正在获取 {code} 消息面信息...", end=" ", flush=True)
            
            try:
                # 调用GLM-4-Flash获取消息面分析
                news_result = self.get_news_analysis(
                    code=code,
                    prompt=self.news_prompt
                )
                
                # 合并消息面信息到原推荐记录
                rec.update({
                    "news_score": news_result.get("news_score", 0),
                    "news_summary": news_result.get("news_summary", ""),
                    "positive_factors": news_result.get("positive_factors", []),
                    "negative_factors": news_result.get("negative_factors", [])
                })
                updated_recommendations.append(rec)
                print("完成")
                
                # 避免请求过快
                if i < len(recommendations):
                    time.sleep(self.config.get("request_delay", 5))
                    
            except Exception as e:
                print(f"失败 — {e}")
                # 保留原数据，消息面字段置空
                rec.update({
                    "news_score": 0,
                    "news_summary": "",
                    "positive_factors": [],
                    "negative_factors": []
                })
                updated_recommendations.append(rec)
        
        # 4. 更新推荐数据并写入文件
        suggestion_data["recommendations"] = updated_recommendations
        
        with open(suggestion_file, "w", encoding="utf-8") as f:
            json.dump(suggestion_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n[INFO] 消息面信息已更新到：{suggestion_file}")
        print(f"       共处理 {len(updated_recommendations)} 支推荐股票")


def main():
    parser = argparse.ArgumentParser(description="智谱AI 股票消息面分析")
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
    reviewer = ZhipuNewsReviewer(config)
    reviewer.update_suggestion_with_news()


if __name__ == "__main__":
    main()