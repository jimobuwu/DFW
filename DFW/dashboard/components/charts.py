"""
dashboard/components/charts.py
K线 / 知行线 / 量能 / 砖型图 — Plotly 图表组件（亮色主题，双周期）
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# rangebreaks 工具
# ─────────────────────────────────────────────────────────────────────────────

def _calc_rangebreaks_daily(trade_dates: pd.DatetimeIndex) -> list[dict]:
    """
    根据实际交易日期动态计算 rangebreaks，彻底去除所有空缺（含节假日）。

    原理：
      1. 排除周末（bounds=["sat","mon"]）
      2. 找出 [min_date, max_date] 范围内所有工作日（周一~周五）中，
         实际不存在交易记录的日期 → 即节假日 + 停牌日 → 加入 values
    """
    if len(trade_dates) == 0:
        return [dict(bounds=["sat", "mon"])]

    min_d = trade_dates.min()
    max_d = trade_dates.max()
    biz_days = pd.bdate_range(min_d, max_d)          # 所有工作日（周一~周五）
    trade_set = set(trade_dates.normalize())          # 实际交易日集合
    missing = [d.strftime("%Y-%m-%d") for d in biz_days if d not in trade_set]

    breaks: list[dict] = [dict(bounds=["sat", "mon"])]
    if missing:
        breaks.append(dict(values=missing))
    return breaks


def _calc_rangebreaks_weekly(all_daily_dates: pd.DatetimeIndex) -> list[dict]:
    """
    根据完整日线交易日期计算周线 rangebreaks，去除因长节假日产生的空周。

    必须传入完整日线日期（而非截断后的周线日期），才能正确覆盖截断窗口
    边界之外的空周（如春节整周落在窗口起点之前一周）。

    原理：枚举 [min, max] 范围内所有理论周五，找出整周无交易日的空周，
    然后将该空周的每个工作日（Mon-Fri）逐一加入 values（而非用 dvalue=7天），
    避免与 bounds=["sat","mon"] 产生重叠扣除，导致显示间距异常。
    """
    if len(all_daily_dates) == 0:
        return [dict(bounds=["sat", "mon"])]

    min_d = all_daily_dates.min()
    max_d = all_daily_dates.max()
    all_fridays = pd.date_range(min_d, max_d, freq="W-FRI")

    # 构建「每个周五所在自然周」→「是否有交易」的映射
    # 自然周工作日：周一(fri-4天) ~ 周五(fri)
    trade_set = set(all_daily_dates.normalize())
    missing_workdays: list[str] = []
    for fri in all_fridays:
        week_workdays = pd.date_range(fri - pd.Timedelta(days=4), fri)  # Mon-Fri
        if not any(d in trade_set for d in week_workdays):
            # 整周无交易 → 逐日加入（只加工作日，与 bounds 无重叠）
            for wd in week_workdays:
                missing_workdays.append(wd.strftime("%Y-%m-%d"))

    breaks: list[dict] = [dict(bounds=["sat", "mon"])]
    if missing_workdays:
        breaks.append(dict(values=missing_workdays))
    return breaks


# ─────────────────────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────────────────────

def _calc_ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def _calc_kdj(
    df: pd.DataFrame,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算 KDJ 指标（通达信标准公式）。
    RSV = (close - LLV(low,n)) / (HHV(high,n) - LLV(low,n)) * 100
    K = EMA(RSV, alpha=1/m1)  （等价 SMA(RSV,m1,1)）
    D = EMA(K,   alpha=1/m2)
    J = 3K - 2D
    """
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    llv = low.rolling(n, min_periods=1).min()
    hhv = high.rolling(n, min_periods=1).max()
    denom = hhv - llv
    denom = denom.replace(0, 1e-6)
    rsv = (close - llv) / denom * 100.0

    alpha_k = 1.0 / m1
    alpha_d = 1.0 / m2
    k = rsv.ewm(alpha=alpha_k, adjust=False).mean()
    d = k.ewm(alpha=alpha_d, adjust=False).mean()
    j = 3 * k - 2 * d

    return k, d, j


def _calc_zx_lines(
    df: pd.DataFrame,
    zxdq_span: int = 10,
    m1: int = 14, m2: int = 28, m3: int = 57, m4: int = 114,
) -> tuple[pd.Series, pd.Series]:
    """
    知行短期线 (zxdq)  = double-EWM(span)
    知行多空线 (zxdkx) = 四均线均值 MA(m1,m2,m3,m4)
    应在完整历史数据上调用，确保预热期足够。
    """
    close = df["close"].astype(float)
    zxdq  = close.ewm(span=zxdq_span, adjust=False).mean().ewm(span=zxdq_span, adjust=False).mean()
    zxdkx = (
        close.rolling(m1, min_periods=m1).mean()
        + close.rolling(m2, min_periods=m2).mean()
        + close.rolling(m3, min_periods=m3).mean()
        + close.rolling(m4, min_periods=m4).mean()
    ) / 4.0
    return zxdq, zxdkx


def prepare_daily_indicators(
    df: pd.DataFrame,
    zx_params: Optional[dict] = None,
    brick_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    在完整日线 DataFrame 上预计算所有指标列，返回带指标列的 df。
    需在截断 bars 之前调用，保证指标预热期足够。

    新增列：
      _zxdq   — 知行短期线
      _zxdkx  — 知行多空线
      _brick  — 砖型图值
      _kdj_k  — KDJ K 值
      _kdj_d  — KDJ D 值
      _kdj_j  — KDJ J 值
    """
    zx_params    = zx_params    or {}
    brick_params = brick_params or {}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    zxdq, zxdkx = _calc_zx_lines(df, **zx_params)
    df["_zxdq"]  = zxdq.values
    df["_zxdkx"] = zxdkx.values
    df["_brick"] = _calc_brick(df, **brick_params).values

    k, d, j = _calc_kdj(df)
    df["_kdj_k"] = k.values
    df["_kdj_d"] = d.values
    df["_kdj_j"] = j.values

    return df


def _calc_brick(
    df: pd.DataFrame,
    n: int = 4, m1: int = 4, m2: int = 6, m3: int = 6,
    t: float = 4.0, shift1: float = 90.0, shift2: float = 100.0,
    sma_w1: int = 1, sma_w2: int = 1, sma_w3: int = 1,
) -> pd.Series:
    """
    计算砖型图的 raw 值（通达信 VAR6A = VAR5A - VAR2A，clip 于 t）。

    注意：此函数返回 raw（即通达信"砖型图"本身），而非差分后的 brick。
    raw[i] = max(VAR5A[i] - VAR2A[i] - t, 0)

    砖型图涨跌（通达信 STICKLINE）需要比较 raw[i] 与 raw[i-1]：
      - raw[i] > raw[i-1]：红柱（上涨）
      - raw[i] < raw[i-1]：绿柱（下跌）
    绘图时每根柱从 raw[i-1]（前日尾部）画到 raw[i]（本日头部）。
    """
    # 纯 numpy/pandas 实现（同 Selector._compute_brick_numba 逻辑，但返回 raw 而非 brick）
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    length = len(close)

    hhv = pd.Series(high).rolling(n, min_periods=1).max().values
    llv = pd.Series(low).rolling(n, min_periods=1).min().values

    a1 = sma_w1 / m1; b1 = 1.0 - a1
    var2a = np.empty(length, dtype=float)
    for i in range(length):
        rng = hhv[i] - llv[i]
        if rng == 0.0: rng = 0.01
        v1 = (hhv[i] - close[i]) / rng * 100.0 - shift1
        var2a[i] = (v1 + shift2) if i == 0 else (a1 * v1 + b1 * (var2a[i - 1] - shift2) + shift2)

    a2 = sma_w2 / m2; b2 = 1.0 - a2
    a3 = sma_w3 / m3; b3 = 1.0 - a3
    var4a = np.empty(length, dtype=float)
    var5a = np.empty(length, dtype=float)
    for i in range(length):
        rng = hhv[i] - llv[i]
        if rng == 0.0: rng = 0.01
        v3 = (close[i] - llv[i]) / rng * 100.0
        if i == 0:
            var4a[i] = v3; var5a[i] = v3 + shift2
        else:
            var4a[i] = a2 * v3 + b2 * var4a[i - 1]
            var5a[i] = a3 * var4a[i] + b3 * (var5a[i - 1] - shift2) + shift2

    raw = np.empty(length, dtype=float)
    for i in range(length):
        diff = var5a[i] - var2a[i]
        raw[i] = diff - t if diff > t else 0.0

    return pd.Series(raw, index=df.index)


def _build_weekly_df(df: pd.DataFrame) -> pd.DataFrame:
    """日线 DataFrame → 周线 OHLCV DataFrame。
    使用 'W-FRI'（以周五为周末）对齐，确保每周最后一个实际交易日落在正确的周桶内。
    dropna 保证只保留有完整 OHLCV 的周。
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.set_index("date").sort_index()
    weekly = d.resample("W-FRI").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open", "close"])
    weekly = weekly.reset_index()          # index -> date 列
    return weekly


# ─────────────────────────────────────────────────────────────────────────────
# 公共布局参数
# ─────────────────────────────────────────────────────────────────────────────

_LIGHT_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#1f2328", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.01,
        xanchor="right",  x=1,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0)",
    ),
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
)

_GRID_COLOR   = "rgba(0,0,0,0.07)"
_ZERO_COLOR   = "rgba(0,0,0,0.25)"


def _apply_axis_style(fig: go.Figure, n_rows: int, rangebreaks: list[dict]) -> None:
    """统一设置所有子图的坐标轴样式。"""
    for i in range(1, n_rows + 1):
        xname = "xaxis" if i == 1 else f"xaxis{i}"
        yname = "yaxis" if i == 1 else f"yaxis{i}"
        fig.update_layout(**{xname: dict(
            rangebreaks=rangebreaks,
            showgrid=False,
            linecolor="#d0d7de",
            tickfont=dict(color="#636c76"),
        )})
        fig.update_layout(**{yname: dict(
            showgrid=True,
            gridcolor=_GRID_COLOR,
            zeroline=False,
            linecolor="#d0d7de",
            tickfont=dict(color="#636c76"),
        )})


# ─────────────────────────────────────────────────────────────────────────────
# 日线图：K线 + 知行线 + 量能 + 砖型图
# ─────────────────────────────────────────────────────────────────────────────

def make_daily_chart(
    df: pd.DataFrame,
    code: str,
    volume_up_color: str = "rgba(220,53,69,0.7)",
    volume_down_color: str = "rgba(40,167,69,0.7)",
    bars: int = 120,
    height: int = 560,
    zx_params: Optional[dict] = None,
    show_brick: bool = False,
    brick_params: Optional[dict] = None,
) -> go.Figure:
    """
    日线图：K线 + 知行短期线 + 知行长期线 + 量能
    知行线在完整数据上预热后截断，保证均线正确性。
    """
    zx_params = zx_params or {}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 在全量数据上计算知行线，确保预热充分
    zxdq, zxdkx = _calc_zx_lines(df, **zx_params)
    df["_zxdq"]  = zxdq.values
    df["_zxdkx"] = zxdkx.values

    if bars > 0:
        df = df.tail(bars).reset_index(drop=True)

    x = df["date"]
    up_mask = df["close"] >= df["open"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
        subplot_titles=[f"{code}  日线", "成交量"],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}]],
    )

    # ── K 线 ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        increasing_line_color="#dc3545",
        decreasing_line_color="#28a745",
        increasing_fillcolor="#dc3545",
        decreasing_fillcolor="#28a745",
        name="K线",
        showlegend=False,
        line=dict(width=1),
    ), row=1, col=1)

    # ── 知行线 ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=df["_zxdq"],
        mode="lines",
        name="短期均线",
        line=dict(color="#e67e22", width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=df["_zxdkx"],
        mode="lines",
        name="长期均线",
        line=dict(color="#2980b9", width=1.5, dash="dot"),
    ), row=1, col=1)

    # ── 成交量 ────────────────────────────────────────────────────────
    vol_colors = np.where(up_mask, volume_up_color, volume_down_color)
    fig.add_trace(go.Bar(
        x=x, y=df["volume"],
        marker_color=vol_colors.tolist(),
        name="成交量",
        showlegend=False,
    ), row=2, col=1)

    # ── 布局 ──────────────────────────────────────────────────────────
    fig.update_layout(height=height, **_LIGHT_LAYOUT)
    _apply_axis_style(fig, 2, _calc_rangebreaks_daily(pd.DatetimeIndex(x)))
    for ann in fig.layout.annotations:
        ann.font = dict(color="#636c76", size=11)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 周线图：K线 + 四条 MA + 量能
# ─────────────────────────────────────────────────────────────────────────────

def make_weekly_chart(
    df: pd.DataFrame,
    code: str,
    ma_windows: List[int] = None,
    ma_colors: Dict[int, str] = None,
    volume_up_color: str = "rgba(220,53,69,0.7)",
    volume_down_color: str = "rgba(40,167,69,0.7)",
    bars: int = 60,
    height: int = 400,
) -> go.Figure:
    """
    周线图：K线 + 四条 MA 均线 + 量能（纯净版，仅用于看盘与导出）。
    df 为完整日线 DataFrame，内部聚合为周线后截取 bars 根展示。
    rangebreaks 基于完整日线日期计算，确保节假日空周被正确排除。
    """
    ma_windows = ma_windows or [5, 10, 20, 60]
    ma_colors  = ma_colors  or {5: "#e67e22", 10: "#27ae60", 20: "#2980b9", 60: "#8e44ad"}

    # 先用完整日线数据计算 rangebreaks（截断前）
    all_daily_dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    weekly_rangebreaks = _calc_rangebreaks_weekly(all_daily_dates)

    wdf = _build_weekly_df(df)

    if bars > 0:
        wdf = wdf.tail(bars).reset_index(drop=True)

    x = wdf["date"]
    up_mask = wdf["close"] >= wdf["open"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
        subplot_titles=[f"{code}  周线", "成交量(周)"],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}]],
    )

    # ── K 线 ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x,
        open=wdf["open"], high=wdf["high"],
        low=wdf["low"],   close=wdf["close"],
        increasing_line_color="#dc3545",
        decreasing_line_color="#28a745",
        increasing_fillcolor="#dc3545",
        decreasing_fillcolor="#28a745",
        name="K线(周)",
        showlegend=False,
        line=dict(width=1),
    ), row=1, col=1)

    # ── MA 均线 ───────────────────────────────────────────────────────
    for w in ma_windows:
        if len(wdf) >= w:
            ma = _calc_ma(wdf["close"], w)
            fig.add_trace(go.Scatter(
                x=x, y=ma,
                mode="lines",
                name=f"MA{w}(周)",
                line=dict(color=ma_colors.get(w, "#aaa"), width=1.4),
            ), row=1, col=1)

    # ── 成交量 ────────────────────────────────────────────────────────
    vol_colors = np.where(up_mask, volume_up_color, volume_down_color)
    fig.add_trace(go.Bar(
        x=x, y=wdf["volume"],
        marker_color=vol_colors.tolist(),
        name="成交量(周)",
        showlegend=False,
    ), row=2, col=1)

    # ── 布局 ──────────────────────────────────────────────────────────
    fig.update_layout(height=height, **_LIGHT_LAYOUT)
    _apply_axis_style(fig, 2, weekly_rangebreaks)
    for ann in fig.layout.annotations:
        ann.font = dict(color="#636c76", size=11)

    return fig
