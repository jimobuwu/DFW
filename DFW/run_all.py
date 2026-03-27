"""
run_all.py
~~~~~~~~~~
一键运行完整交易选股流程，分别对 B1 和砖型图两个策略执行独立流水线。

数据目录与代码目录分离：
  - DFW/          代码目录（本文件所在目录）
  - DATA_ROOT/    数据目录（通过 --data-root 指定，默认为 DFW 同级的 data/）

步骤：
  1  拉取最新 K 线数据（共享）
  2  对每个策略分别执行：
     2a  量化初选
     2b  导出候选 K 线图
     2c  AI 图表复评
  3  打印各策略推荐购买的股票

数据目录结构（DATA_ROOT 下）：
  raw/                      原始日线 CSV
  candidates/               候选列表 JSON
  kline/                    K 线图
  review_b1/{date}/         B1 策略复评结果
  review_brick/{date}/      砖型图策略复评结果
  logs/                     运行日志

用法：
    python run_all.py
    python run_all.py --data-root /path/to/data
    python run_all.py --skip-fetch
    python run_all.py --strategies b1 brick
    python run_all.py --reviewer zhipu
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DFW_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STRATEGY_LIST = ["b1", "brick"]

REVIEWER_MAP = {
    "zhipu": str(DFW_ROOT / "agent" / "zhipu_review.py"),
    "gemini": str(DFW_ROOT / "agent" / "gemini_review.py"),
}


def _run(step_name: str, cmd: list[str], cwd: str | None = None) -> None:
    """运行子进程，失败时终止整个流程。"""
    print(f"\n{'='*60}")
    print(f"[步骤] {step_name}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd or str(DFW_ROOT))
    if result.returncode != 0:
        print(f"\n[ERROR] 步骤「{step_name}」返回非零退出码 {result.returncode}，流程已中止。")
        sys.exit(result.returncode)


def _candidates_path(data_root: Path, strategy: str) -> Path:
    return data_root / "candidates" / f"candidates_latest_{strategy}.json"


def _review_dir(data_root: Path, strategy: str) -> Path:
    return data_root / f"review_{strategy}"


def _print_recommendations(data_root: Path, strategy: str) -> None:
    """读取指定策略的 suggestion.json，打印推荐购买的股票。"""
    cand_file = _candidates_path(data_root, strategy)
    if not cand_file.exists():
        print(f"[WARN] 找不到 {cand_file.name}，跳过 {strategy} 推荐打印。")
        return

    with open(cand_file, encoding="utf-8") as f:
        pick_date: str = json.load(f).get("pick_date", "")

    if not pick_date:
        print(f"[ERROR] {cand_file.name} 中未设置 pick_date。")
        return

    suggestion_file = _review_dir(data_root, strategy) / pick_date / "suggestion.json"
    if not suggestion_file.exists():
        print(f"[WARN] 找不到 {suggestion_file}，跳过 {strategy} 推荐打印。")
        return

    with open(suggestion_file, encoding="utf-8") as f:
        suggestion: dict = json.load(f)

    recommendations: list[dict] = suggestion.get("recommendations", [])
    min_score: float = suggestion.get("min_score_threshold", 0)
    total: int = suggestion.get("total_reviewed", 0)

    print(f"\n{'='*60}")
    print(f"  策略：{strategy.upper()}")
    print(f"  选股日期：{pick_date}")
    print(f"  评审总数：{total} 只   推荐门槛：score >= {min_score}")
    print(f"{'='*60}")

    if not recommendations:
        print("  暂无达标推荐股票。")
        return

    header = f"{'排名':>4}  {'代码':>8}  {'总分':>6}  {'信号':>10}  {'研判':>6}  备注"
    print(header)
    print("-" * len(header))
    for r in recommendations:
        rank        = r.get("rank",        "?")
        code        = r.get("code",        "?")
        score       = r.get("total_score", "?")
        signal_type = r.get("signal_type", "")
        verdict     = r.get("verdict",     "")
        comment     = r.get("comment",     "")
        score_str   = f"{score:.1f}" if isinstance(score, (int, float)) else str(score)
        print(f"{rank:>4}  {code:>8}  {score_str:>6}  {signal_type:>10}  {verdict:>6}  {comment}")

    print(f"\n  推荐购买 {len(recommendations)} 只股票（详见 {suggestion_file}）")


def _run_strategy_pipeline(
    strategy: str,
    data_root: Path,
    reviewer_script: str,
    reviewer_config: str,
    step_offset: int,
    total_steps: int,
) -> None:
    """对单个策略运行完整的 初选 → 导出图表 → AI复评 流水线。"""
    tag = strategy.upper()
    cand_file = str(_candidates_path(data_root, strategy))
    review_dir = str(_review_dir(data_root, strategy))
    raw_dir = str(data_root / "raw")
    candidates_dir = str(data_root / "candidates")
    kline_dir = str(data_root / "kline")
    log_dir = str(data_root / "logs")

    # 2a) 量化初选
    _run(
        f"{step_offset}/{total_steps}  [{tag}] 量化初选",
        [PYTHON, "-m", "pipeline.cli", "preselect",
         "--strategy", strategy,
         "--data", raw_dir,
         "--output", candidates_dir,
         "--log-dir", log_dir],
    )

    # 2b) 导出 K 线图
    _run(
        f"{step_offset + 1}/{total_steps}  [{tag}] 导出 K 线图",
        [PYTHON, str(DFW_ROOT / "dashboard" / "export_kline_charts.py"),
         "--candidates", cand_file,
         "--raw-dir", raw_dir,
         "--out-dir", kline_dir],
    )

    # 2c) AI 图表复评
    _run(
        f"{step_offset + 2}/{total_steps}  [{tag}] AI 图表复评",
        [PYTHON, reviewer_script,
         "--config", reviewer_config,
         "--candidates", cand_file,
         "--output-dir", review_dir,
         "--kline-dir", kline_dir],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentTrader 全流程自动运行脚本")
    parser.add_argument(
        "--data-root", default=None,
        help="数据根目录（默认为 DFW 同级的 data/ 目录）",
    )
    parser.add_argument(
        "--skip-fetch", action="store_true",
        help="跳过步骤 1（行情下载），直接从初选开始",
    )
    parser.add_argument(
        "--start-from", type=int, default=1, metavar="N",
        help="从第 N 步开始执行（1=下载, 2=初选+复评）",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=STRATEGY_LIST,
        choices=STRATEGY_LIST,
        help="要运行的策略列表（默认 b1 brick 全部运行）",
    )
    parser.add_argument(
        "--reviewer", default="zhipu", choices=list(REVIEWER_MAP.keys()),
        help="AI 复评模型（默认 zhipu）",
    )
    args = parser.parse_args()

    # 数据根目录：默认为 DFW 同级的 data/
    if args.data_root:
        data_root = Path(args.data_root).resolve()
    else:
        data_root = DFW_ROOT.parent / "data"

    # 确保数据子目录存在
    for sub in ["raw", "candidates", "kline", "logs"]:
        (data_root / sub).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 代码目录: {DFW_ROOT}")
    print(f"[INFO] 数据目录: {data_root}")

    start = args.start_from
    if args.skip_fetch and start == 1:
        start = 2

    strategies = args.strategies
    reviewer_script = REVIEWER_MAP[args.reviewer]
    reviewer_config_map = {
        "zhipu": str(DFW_ROOT / "config" / "zhipu_review.yaml"),
        "gemini": str(DFW_ROOT / "config" / "gemini_review.yaml"),
    }
    reviewer_config = reviewer_config_map[args.reviewer]

    n_strategies = len(strategies)
    total_steps = 1 + n_strategies * 3 + 1

    # ── 步骤 1：拉取 K 线数据 ─────────────────────────────────────────
    if start <= 1:
        raw_dir = str(data_root / "raw")
        _run(
            f"1/{total_steps}  拉取 K 线数据（fetch_kline）",
            [PYTHON, "-m", "pipeline.fetch_kline",
             "--out", raw_dir],
        )

    # ── 步骤 2~N：各策略独立流水线 ────────────────────────────────────
    if start <= 2:
        for i, strategy in enumerate(strategies):
            step_offset = 2 + i * 3
            _run_strategy_pipeline(
                strategy=strategy,
                data_root=data_root,
                reviewer_script=reviewer_script,
                reviewer_config=reviewer_config,
                step_offset=step_offset,
                total_steps=total_steps,
            )

    # ── 最后：打印各策略推荐结果 ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[步骤] {total_steps}/{total_steps}  推荐购买的股票")
    for strategy in strategies:
        _print_recommendations(data_root, strategy)

    print(f"\n{'='*60}")
    print("全部流程完成。")
    print(f"数据目录: {data_root}")


if __name__ == "__main__":
    main()
