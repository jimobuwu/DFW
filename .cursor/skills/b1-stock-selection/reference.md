# B1 策略技术参考

## 1. 指标计算公式

### KDJ 指标

```
RSV = (Close - LLV(Low, 9)) / (HHV(High, 9) - LLV(Low, 9) + 1e-9) × 100
K[0] = D[0] = 50
K[i] = 2/3 × K[i-1] + 1/3 × RSV[i]
D[i] = 2/3 × D[i-1] + 1/3 × K[i]
J = 3K - 2D
```

Numba 加速函数：`_kdj_core(rsv: ndarray) -> (K, D, J)`

### 知行线（ZX Lines）

- **zxdq**（快线）：`close.ewm(span=10).mean().ewm(span=10).mean()`
- **zxdkx**（慢线）：`(MA(m1) + MA(m2) + MA(m3) + MA(m4)) / 4`

函数：`compute_zx_lines(df, m1, m2, m3, m4, zxdq_span) -> (zxdq, zxdkx)`

### 周线均线

```python
compute_weekly_close(df)          # 日线 → 周线收盘（ISO周分组取最后交易日）
compute_weekly_ma_bull(df, (s,m,l))  # 周线 MA_s > MA_m > MA_l，ffill 回日线
```

### 砖型图（Brick Chart）

通达信公式的 Numba 实现，参数 `(n, m1, m2, m3, t, shift1, shift2, sma_w1, sma_w2, sma_w3)`：

```
HHV = 滚动 n 日最高价
LLV = 滚动 n 日最低价
var2a = SMA((HHV - close) / (HHV - LLV) × 100 - shift1, m1, sma_w1) + shift2
var4a = SMA((close - LLV) / (HHV - LLV) × 100, m2, sma_w2)
var5a = SMA(var4a, m3, sma_w3) + shift2
raw = max(var5a - var2a - t, 0)
brick[i] = raw[i] - raw[i-1]
```

`brick > 0` 为红柱（多头），`brick < 0` 为绿柱（空头）。

## 2. Filter 接口规范

每个 Filter 必须实现：

```python
@dataclass(frozen=True)
class MyFilter:
    param: float = 1.0

    def __call__(self, hist: pd.DataFrame) -> bool:
        """点查：给定截至某日的历史 DataFrame，返回是否通过。"""
        ...

    def vec_mask(self, df: pd.DataFrame) -> np.ndarray:
        """向量化：返回布尔数组，长度 == len(df)。"""
        ...
```

### 已有 Filter 清单

| Filter | 用途 | 关键参数 |
|--------|------|----------|
| `KDJQuantileFilter` | J 值低位 | j_threshold, j_q_threshold, kdj_n |
| `ZXConditionFilter` | 知行线多头 | zx_m1~m4, zxdq_span, require_close_gt_long, require_short_gt_long |
| `WeeklyMABullFilter` | 周线多头排列 | wma_short, wma_mid, wma_long |
| `MaxVolNotBearishFilter` | 量能健康 | n |
| `BrickPatternFilter` | 砖型图形态 | daily_return_threshold, brick_growth_ratio, min_prior_green_bars |
| `ZXDQRatioFilter` | 价格偏离度 | zxdq_ratio, zxdq_span |

## 3. Selector 接口规范

继承 `PipelineSelector`：

```python
class MySelector(PipelineSelector):
    def __init__(self, ...):
        # 创建 Filter 实例
        self._filters = [filter1, filter2, ...]
        super().__init__(filters=self._filters, min_bars=..., ...)

    def prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """预计算所有指标列 + _vec_pick 布尔列。"""
        df = df.copy()
        # 计算指标列...
        df["_vec_pick"] = _apply_vec_filters(df, self._filters)
        return df
```

`_apply_vec_filters` 对所有 Filter 的 `vec_mask` 取交集。

## 4. 选股流程 API

```python
from select_stock import run_preselect

pick_ts, candidates = run_preselect(
    config_path="config/rules_preselect.yaml",
    data_dir="data/raw",
    end_date="2026-03-27",
    pick_date="2026-03-27",
)
# candidates: List[Candidate]
```

## 5. Candidate 数据结构

```python
@dataclass
class Candidate:
    code: str              # "600519"
    date: str              # "2026-03-27"
    strategy: str          # "b1" / "brick"
    close: float
    turnover_n: float
    brick_growth: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
```

## 6. 配置文件结构（rules_preselect.yaml）

```yaml
global:
  data_dir: "./data/raw"
  output_dir: "./data/candidates"
  top_m: 5000              # 流动性池大小
  n_turnover_days: 43      # 滚动成交额窗口
  min_bars_buffer: 10

b1:
  enabled: true
  zx_m1: 14
  zx_m2: 28
  zx_m3: 57
  zx_m4: 114
  j_threshold: 15.0
  j_q_threshold: 0.10

brick:
  enabled: false
  n: 8
  m1: 3
  # ... (完整参数见 config/rules_preselect.yaml)
```

## 7. 性能优化要点

- `prepare_df()` 预计算所有指标列（一次 O(N)），后续 `vec_picks_from_prepared()` 直接查表
- Numba `@njit(cache=True)` 加速 KDJ 递推、砖型图核心、连续绿柱计数
- `ProcessPoolExecutor` 多进程并行数据预处理
- `ThreadPoolExecutor` 并行 selector 特征计算（pandas/numpy 释放 GIL）
- `prepare_df_brick_only()`：超参搜索时仅重算砖型图列，速度快 3-10×

## 8. AI 复评完整评分体系

### 信号类型（三选一）

| 信号 | 含义 |
|------|------|
| `trend_start` | 主升启动 |
| `rebound` | 跌后反弹 |
| `distribution_risk` | 出货风险 |

### 特殊规则

- `volume_behavior = 1` → 强制 FAIL
- 总分 >= 4.0 → PASS
- 3.2 ~ 4.0 → WATCH
- < 3.2 → FAIL

### 推理顺序

`趋势结构 → 价格位置 → 量价行为 → 前期建仓异动 → 信号类型` → JSON 输出
