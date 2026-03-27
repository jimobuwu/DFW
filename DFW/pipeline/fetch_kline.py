from __future__ import annotations

import datetime as dt
import logging
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import os
import threading

import pandas as pd
import tushare as ts
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --------------------------- pandas 兼容补丁 --------------------------- #
import pandas as _pd

_orig_fillna = _pd.DataFrame.fillna

def _patched_fillna(self, value=None, *, method=None, axis=None, inplace=False, limit=None, **kwargs):
    if method is not None:
        if method == "ffill":
            result = self.ffill(axis=axis, inplace=inplace, limit=limit)
        elif method == "bfill":
            result = self.bfill(axis=axis, inplace=inplace, limit=limit)
        else:
            raise ValueError(f"Unsupported fillna method: {method}")
        return result
    return _orig_fillna(self, value, axis=axis, inplace=inplace, limit=limit, **kwargs)

_pd.DataFrame.fillna = _patched_fillna  # type: ignore[method-assign]

_orig_series_fillna = _pd.Series.fillna

def _patched_series_fillna(self, value=None, *, method=None, axis=None, inplace=False, limit=None, **kwargs):
    if method is not None:
        if method == "ffill":
            result = self.ffill(axis=axis, inplace=inplace, limit=limit)
        elif method == "bfill":
            result = self.bfill(axis=axis, inplace=inplace, limit=limit)
        else:
            raise ValueError(f"Unsupported fillna method: {method}")
        return result
    return _orig_series_fillna(self, value, axis=axis, inplace=inplace, limit=limit, **kwargs)

_pd.Series.fillna = _patched_series_fillna  # type: ignore[method-assign]

# --------------------------- 全局日志配置 --------------------------- #
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _PROJECT_ROOT / "data" / "logs"

def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _PROJECT_ROOT) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)

def _default_log_path() -> Path:
    today = dt.date.today().strftime("%Y-%m-%d")
    return _DEFAULT_LOG_DIR / f"fetch_{today}.log"

def setup_logging(log_path: Optional[Path] = None) -> None:
    if log_path is None:
        log_path = _default_log_path()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )

logger = logging.getLogger("fetch_from_stocklist")

# --------------------------- 严格限流器（50次/分钟） --------------------------- #
class RateLimiter:
    def __init__(self, max_requests: int = 50, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            cutoff = now - self.window_seconds
            self.requests = [t for t in self.requests if t > cutoff]

            if len(self.requests) >= self.max_requests:
                oldest_in_window = min(self.requests)
                wait_time = oldest_in_window + self.window_seconds - now
                wait_time = max(wait_time, 0.1)

                logger.warning(f"分钟限流：已请求 {len(self.requests)} 次，需等待 {wait_time:.1f} 秒")

                self.lock.release()
                try:
                    time.sleep(wait_time)
                finally:
                    self.lock.acquire()

                now = time.time()
                cutoff = now - self.window_seconds
                self.requests = [t for t in self.requests if t > cutoff]

            self.requests.append(now)
            remaining = self.max_requests - len(self.requests)
            logger.debug(f"限流器：本次请求通过，窗口内 {len(self.requests)} 次，剩余额度 {remaining}")

_global_limiter = RateLimiter(max_requests=50, window_seconds=60)

# --------------------------- 限流/封禁处理配置 --------------------------- #
COOLDOWN_SECS = 600
BAN_PATTERNS = (
    "访问频繁", "请稍后", "超过频率", "频繁访问",
    "too many requests", "429",
    "forbidden", "403",
    "max retries exceeded",
    "每分钟最多访问"
)

def _looks_like_ip_ban(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return any(pat.lower() in msg for pat in BAN_PATTERNS)

class RateLimitError(RuntimeError):
    pass

def _cool_sleep(base_seconds: int) -> None:
    jitter = random.uniform(0.9, 1.2)
    sleep_s = max(1, int(base_seconds * jitter))
    logger.warning("疑似被限流/封禁，进入冷却期 %d 秒...", sleep_s)
    time.sleep(sleep_s)

# --------------------------- 本地缓存检查（强制更新：只要不是最新就拉取） --------------------------- #
def _should_skip_fetch(code: str, raw_dir: Path, end_date: str) -> tuple[bool, Optional[dt.date]]:
    """
    检查是否应该跳过拉取 —— 【严格模式】
    只有本地数据最新日期 == 目标日期 才跳过，否则一律拉取
    """
    csv_path = raw_dir / f"{code}.csv"

    if not csv_path.exists():
        return False, None

    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        if df.empty or "date" not in df.columns:
            logger.warning(f"{code}: 文件为空/格式异常，重新拉取")
            return False, None

        latest_date = df["date"].max().date()

        if end_date.lower() == "today":
            target_end = dt.date.today()
        else:
            target_end = dt.datetime.strptime(end_date, "%Y%m%d").date()

        if latest_date >= target_end:
            logger.info(f"{code}: 本地已最新 {latest_date}，跳过")
            return True, latest_date
        else:
            logger.info(f"{code}: 本地 {latest_date} < 目标 {target_end}，需要更新")
            return False, latest_date

    except Exception as e:
        logger.warning(f"{code}: 读取文件失败: {e}，重新拉取")
        return False, None

# --------------------------- 历史K线（Tushare 日线，固定qfq） --------------------------- #
pro: Optional[ts.pro_api] = None

def set_api(session) -> None:
    global pro
    pro = session

def _to_ts_code(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "9")):
        return f"{code}.SH"
    elif code.startswith(("4", "8")):
        return f"{code}.BJ"
    else:
        return f"{code}.SZ"

def _get_kline_tushare(code: str, start: str, end: str) -> pd.DataFrame:
    _global_limiter.acquire()

    ts_code = _to_ts_code(code)
    try:
        df = ts.pro_bar(
            ts_code=ts_code,
            adj="qfq",
            start_date=start,
            end_date=end,
            freq="D",
            api=pro
        )
    except Exception as e:
        if _looks_like_ip_ban(e):
            raise RateLimitError(str(e)) from e
        raise

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"trade_date": "date", "vol": "volume"})[
        ["date", "open", "close", "high", "low", "volume"]
    ].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)

def validate(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df

# --------------------------- 读取 stocklist.csv & 过滤板块 --------------------------- #
def _filter_by_boards_stocklist(df: pd.DataFrame, exclude_boards: set[str]) -> pd.DataFrame:
    ts = df["ts_code"].astype(str).str.upper()
    num = ts.str.extract(r"(\d{6})", expand=False).str.zfill(6)
    mask = pd.Series(True, index=df.index)

    if "bj" in exclude_boards:
        mask &= ~((ts.str.endswith(".BJ")) | num.str.startswith(("4", "8")))

    return df[mask].copy()

def load_codes_from_stocklist(stocklist_csv: Path, exclude_boards: set[str]) -> List[str]:
    df = pd.read_csv(stocklist_csv)
    df = _filter_by_boards_stocklist(df, exclude_boards)
    codes = df["symbol"].astype(str).str.zfill(6).tolist()
    codes = list(dict.fromkeys(codes))
    logger.info("从 %s 读取到 %d 只股票（排除板块：%s）",
                stocklist_csv, len(codes), ",".join(sorted(exclude_boards)) or "无")
    return codes

# --------------------------- 单只抓取 --------------------------- #
def fetch_one(
    code: str,
    start: str,
    end: str,
    out_dir: Path,
    skip_existing: bool = True,
):
    csv_path = out_dir / f"{code}.csv"

    if skip_existing:
        should_skip, latest_date = _should_skip_fetch(code, out_dir, end)
        if should_skip:
            return {
                "code": code,
                "status": "skipped",
                "latest_date": latest_date,
                "reason": "本地已是最新日期"
            }

    for attempt in range(1, 4):
        try:
            new_df = _get_kline_tushare(code, start, end)
            if new_df.empty:
                logger.debug("%s 无数据，生成空表。", code)
                new_df = pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
            new_df = validate(new_df)
            new_df.to_csv(csv_path, index=False)
            return {
                "code": code,
                "status": "success",
                "rows": len(new_df)
            }
        except Exception as e:
            if _looks_like_ip_ban(e):
                logger.error(f"{code} 第 {attempt} 次抓取疑似被封禁，沉睡 {COOLDOWN_SECS} 秒")
                _cool_sleep(COOLDOWN_SECS)
            else:
                silent_seconds = 30 * attempt
                logger.info(f"{code} 第 {attempt} 次抓取失败，{silent_seconds} 秒后重试：{e}")
                time.sleep(silent_seconds)
    else:
        logger.error("%s 三次抓取均失败，已跳过！", code)
        return {
            "code": code,
            "status": "failed",
            "error": "三次重试失败"
        }

# --------------------------- 配置加载 --------------------------- #
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "fetch_kline.yaml"

def _load_config(config_path: Path = _CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("已加载配置文件：%s", config_path.resolve())
    return cfg

# --------------------------- 主入口 --------------------------- #
def main(log_path: Optional[Path] = None, out_override: Optional[str] = None, force_update: bool = False):
    cfg = _load_config()

    if out_override:
        cfg["out"] = out_override

    if log_path is None:
        cfg_log = cfg.get("log")
        log_path = _resolve_cfg_path(cfg_log) if cfg_log else _default_log_path()
    setup_logging(log_path)
    logger.info("日志文件：%s", Path(log_path).resolve())

    os.environ["NO_PROXY"] = "api.waditu.com,.waditu.com,waditu.com"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    ts_token = os.environ.get("TUSHARE_TOKEN")
    if not ts_token:
        raise ValueError("请先设置环境变量 TUSHARE_TOKEN")
    ts.set_token(ts_token)
    global pro
    pro = ts.pro_api()

    raw_start = str(cfg.get("start", "20190101"))
    raw_end   = str(cfg.get("end",   "today"))
    start = dt.date.today().strftime("%Y%m%d") if raw_start.lower() == "today" else raw_start
    end   = dt.date.today().strftime("%Y%m%d") if raw_end.lower()   == "today" else raw_end

    out_dir = _resolve_cfg_path(cfg.get("out", "./data/raw"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"数据输出目录: {out_dir}")

    stocklist_path = _resolve_cfg_path(cfg.get("stocklist", "./pipeline/stocklist.csv"))
    exclude_boards = set(cfg.get("exclude_boards") or [])
    codes = load_codes_from_stocklist(stocklist_path, exclude_boards)

    if not codes:
        logger.error("stocklist 为空或被过滤后无代码，请检查。")
        sys.exit(1)

    if not force_update:
        codes_to_fetch = []
        skipped_codes = []
        for code in codes:
            should_skip, latest_date = _should_skip_fetch(code, out_dir, end)
            if should_skip:
                skipped_codes.append((code, latest_date))
            else:
                codes_to_fetch.append(code)

        logger.info(
            "开始抓取 %d 支股票 | 跳过 %d 支（已是最新）| 数据源:Tushare(日线,qfq) | 日期:%s → %s",
            len(codes_to_fetch), len(skipped_codes), start, end
        )
        if skipped_codes:
            logger.info(f"跳过的股票示例: {skipped_codes[:5]}...")
    else:
        codes_to_fetch = codes
        skipped_codes = []
        logger.info(f"强制更新模式：全部 {len(codes)} 支股票重新拉取")

    if not codes_to_fetch:
        logger.info("所有股票数据均已最新，无需拉取")
        return

    workers = int(cfg.get("workers", 4))
    results = {"success": 0, "skipped": len(skipped_codes), "failed": 0}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_one,
                code,
                start,
                end,
                out_dir,
                skip_existing=not force_update,
            ): code for code in codes_to_fetch
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
            code = futures[future]
            try:
                result = future.result()
                if result["status"] == "success":
                    results["success"] += 1
                elif result["status"] == "skipped":
                    results["skipped"] += 1
                else:
                    results["failed"] += 1
            except Exception as e:
                logger.error(f"{code} 任务异常: {e}")
                results["failed"] += 1

    logger.info(
        "全部任务完成 | 成功: %d | 跳过: %d | 失败: %d | 数据已保存至 %s",
        results["success"], results["skipped"], results["failed"], out_dir.resolve()
    )

if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="拉取 K 线数据")
    _parser.add_argument("--out", default=None, help="K 线数据输出目录（覆盖配置文件）")
    _parser.add_argument("--force", action="store_true", help="强制重新拉取所有数据")
    _args = _parser.parse_args()
    main(out_override=_args.out, force_update=_args.force)
