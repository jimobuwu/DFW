#!/usr/bin/env bash
# =============================================================================
# run.sh — AgentTrader 一键运行脚本
#
# 目录结构：
#   StockTradebyZ-main/
#   ├── DFW/           代码目录（所有 .py / .yaml / .md 等源文件）
#   ├── StockData/     数据目录（运行时自动创建，存放所有中间和结果数据）
#   └── run.sh         本脚本
#
# 用法：
#   bash run.sh                      # 完整流程（拉数据 + B1 + 砖型图）
#   bash run.sh --skip-fetch         # 跳过数据下载
#   bash run.sh --strategies b1      # 仅运行 B1 策略
#   bash run.sh --reviewer gemini    # 使用 Gemini 做 AI 复评
#
# 所有额外参数会直接传递给 DFW/run_all.py
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DFW_DIR="$SCRIPT_DIR/DFW"
DATA_DIR="$SCRIPT_DIR/StockData"

# 创建数据目录结构
mkdir -p "$DATA_DIR/raw"
mkdir -p "$DATA_DIR/candidates"
mkdir -p "$DATA_DIR/kline"
mkdir -p "$DATA_DIR/logs"

echo "============================================================"
echo "  AgentTrader 选股系统"
echo "  代码目录: $DFW_DIR"
echo "  数据目录: $DATA_DIR"
echo "============================================================"

# 切换到代码目录（pipeline 模块依赖 cwd = DFW）
cd "$DFW_DIR"

# 运行主流程，将数据根目录指向 StockData
python run_all.py --data-root "$DATA_DIR" "$@"
