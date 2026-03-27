"""
pipeline/io.py
统一路径解析 + 原子写入 candidates*.json。

契约规则：
  - 按日期存档：candidates/candidates_YYYY-MM-DD.json
  - 唯一契约文件（下游只读）：candidates/candidates_latest.json
  - 写入采用"先写临时文件 → os.replace 原子替换"，防止下游读到半写文件。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Union

from schemas import CandidateRun

logger = logging.getLogger(__name__)

# 默认输出目录（相对于项目根）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CANDIDATES_DIR = _PROJECT_ROOT / "data" / "candidates"


def _resolve_path(path_like: Union[str, Path]) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (_PROJECT_ROOT / p)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, content: str) -> None:
    """原子写入：先写 .tmp，再 os.replace。"""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    logger.debug("写入完成: %s", path)


def save_candidates(
    run: CandidateRun,
    *,
    candidates_dir: Union[str, Path, None] = None,
    write_dated: bool = True,
    write_latest: bool = True,
    strategy_tag: str = "",
) -> dict[str, Path]:
    """
    将 CandidateRun 序列化为 JSON，写入磁盘。

    参数
    ----
    run             : CandidateRun 对象
    candidates_dir  : 输出目录，默认 data/candidates/
    write_dated     : 是否写 candidates_YYYY-MM-DD.json
    write_latest    : 是否覆盖 candidates_latest.json
    strategy_tag    : 策略标签（如 "b1" / "brick"），非空时文件名带策略后缀

    返回
    ----
    写入成功的路径字典，key 为 "dated" / "latest"。
    """
    out_dir = _resolve_path(candidates_dir) if candidates_dir else _DEFAULT_CANDIDATES_DIR
    _ensure_dir(out_dir)

    payload = json.dumps(run.to_dict(), ensure_ascii=False, indent=2)
    written: dict[str, Path] = {}

    suffix = f"_{strategy_tag}" if strategy_tag else ""

    if write_dated:
        dated_path = out_dir / f"candidates_{run.pick_date}{suffix}.json"
        _atomic_write(dated_path, payload)
        written["dated"] = dated_path
        logger.info("存档文件: %s", dated_path)

    if write_latest:
        latest_path = out_dir / f"candidates_latest{suffix}.json"
        _atomic_write(latest_path, payload)
        written["latest"] = latest_path
        logger.info("契约文件: %s", latest_path)

    return written


def load_latest(
    candidates_dir: Union[str, Path, None] = None,
) -> CandidateRun:
    """
    读取 candidates_latest.json，返回 CandidateRun。
    供 dashboard 或外部脚本调用。
    """
    out_dir = _resolve_path(candidates_dir) if candidates_dir else _DEFAULT_CANDIDATES_DIR
    latest_path = out_dir / "candidates_latest.json"

    if not latest_path.exists():
        raise FileNotFoundError(f"契约文件不存在: {latest_path}")

    data = json.loads(latest_path.read_text(encoding="utf-8"))
    return CandidateRun.from_dict(data)


def load_by_date(
    pick_date: str,
    candidates_dir: Union[str, Path, None] = None,
) -> CandidateRun:
    """读取指定日期的存档文件。"""
    out_dir = _resolve_path(candidates_dir) if candidates_dir else _DEFAULT_CANDIDATES_DIR
    path = out_dir / f"candidates_{pick_date}.json"
    if not path.exists():
        raise FileNotFoundError(f"存档文件不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return CandidateRun.from_dict(data)
