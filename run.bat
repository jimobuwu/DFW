@echo off
REM =============================================================================
REM run.bat — AgentTrader 一键运行脚本 (Windows)
REM
REM 目录结构：
REM   StockTradebyZ-main\
REM   ├── DFW\           代码目录
REM   ├── StockData\     数据目录（自动创建）
REM   └── run.bat        本脚本
REM
REM 用法：
REM   run.bat                          完整流程
REM   run.bat --skip-fetch             跳过数据下载
REM   run.bat --strategies b1          仅运行 B1 策略
REM   run.bat --reviewer gemini        使用 Gemini 做 AI 复评
REM =============================================================================

set SCRIPT_DIR=%~dp0
set DFW_DIR=%SCRIPT_DIR%DFW
set DATA_DIR=%SCRIPT_DIR%StockData

REM 创建数据目录结构
if not exist "%DATA_DIR%\raw" mkdir "%DATA_DIR%\raw"
if not exist "%DATA_DIR%\candidates" mkdir "%DATA_DIR%\candidates"
if not exist "%DATA_DIR%\kline" mkdir "%DATA_DIR%\kline"
if not exist "%DATA_DIR%\logs" mkdir "%DATA_DIR%\logs"

echo ============================================================
echo   AgentTrader 选股系统
echo   代码目录: %DFW_DIR%
echo   数据目录: %DATA_DIR%
echo ============================================================

REM 切换到代码目录
cd /d "%DFW_DIR%"

REM 运行主流程，将数据根目录指向 StockData
python run_all.py --data-root "%DATA_DIR%" %*
