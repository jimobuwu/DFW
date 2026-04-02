# B1/B2/砖型图 策略技术参考

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

### 知行线（双线战法 / ZX Lines）

- **白线 / zxdq**（快线/短期趋势线）：`close.ewm(span=10).mean().ewm(span=10).mean()`
  - 股价短期快速上涨时一般不跌破白线（"牵牛绳"）
  - 跌破白线 → 短期趋势走坏
- **黄线 / zxdkx**（慢线/知行多空线/大哥线）：`(MA(m1) + MA(m2) + MA(m3) + MA(m4)) / 4`
  - 主力入场的标志性参考线
  - 极限洗盘的极限位
  - 大哥控盘的成本线
  - 跌破黄线且次日无法收回 → 中期趋势走坏

函数：`compute_zx_lines(df, m1, m2, m3, m4, zxdq_span) -> (zxdq, zxdkx)`

### 双线关系判定

| 状态 | 含义 | 操作 |
|------|------|------|
| zxdq > zxdkx（白线在黄线上） | 多头排列 | 右侧交易有效区 |
| 白线金叉黄线 | 上涨趋势确立 | 等回踩 B1 入场 |
| 白线死叉黄线 | 趋势终结 | 最后离场时机 |
| 白线要死叉而未死叉 + J 值负 | 极限洗盘 | 极限买点（需止损） |
| 价格在白黄区间 | N 型上涨中继 | 容错率较高的买入区 |
| 放量金叉后缩量踩黄线 | 最后震仓 | 高价值 B2 买点 |

### 周线均线

```python
compute_weekly_close(df)             # 日线 → 周线收盘（ISO周分组取最后交易日）
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

`brick > 0` 为红砖（多头），`brick < 0` 为绿砖（空头）。

#### 砖型图大小判定标准

- 主板（10cm 涨跌幅）：有效红砖 ≥ 5%（即 ≥ 5 厘米）
- 创业板/科创板（20cm 涨跌幅）：有效红砖 ≥ 10%÷2 = 5%（即涨幅除以 2 判断）
- 阳线实际跌幅 2% 仍属有效红砖（以砖型图颜色为准，非 K 线视觉）

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

### 建议新增 Filter

| Filter | 用途 | 关键参数 |
|--------|------|----------|
| `B2VolumeReverseFilter` | B2 倍量反包 | b1_lookback, shrink_days, volume_ratio, zxdkx_tolerance |
| `BrickNTypeFilter` | N 型起跳检测 | n_lookback, min_brick_size |
| `BrickBreakoutFilter` | 横盘突破检测 | consolidation_days, range_ratio |
| `BrickUpperShadowFilter` | 大上影线排除 | shadow_body_ratio |
| `FourBrickExitSignal` | 四块砖止盈 | max_bricks |

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
    strategy: str          # "b1" / "brick" / "b2"（待实现）
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
  enabled: true
  n: 8
  m1: 3
  m2: 12
  m3: 12
  t: 8
  shift1: 92
  shift2: 114
  daily_return_threshold: 0.2
  brick_growth_ratio: 0.5
  min_prior_green_bars: 1
  zxdq_ratio: 1.47
  require_zxdq_gt_zxdkx: true
  require_weekly_ma_bull: true
  wma_short: 5
  wma_mid: 10
  wma_long: 20

# b2:                      # 待实现
#   enabled: false
#   b1_lookback: 30
#   shrink_days: 2
#   volume_ratio: 1.3
#   zxdkx_tolerance: 0.03
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

## 9. B2 策略实现参考

### B2 经典图形模式（来自 22 个实战案例）

```
B1 买点 → 缩量回调（踩黄线） → 倍量阳线反包 = B2 买点
```

关键量化指标：
- **缩量判定**：volume[i] < volume[i-1] 连续 >= 2 日
- **踩黄线判定**：|close - zxdkx| / zxdkx < tolerance（默认 3%）
- **放量判定**：volume >= prev_volume × 1.3（最低 30%，理想为倍量 100%+）
- **阳线判定**：close > open
- **反包判定**：close >= prev_close

### B2 反面案例特征

- 大长下影线的放量阳线（实体虚，不可靠）
- 异动阶段涨幅 > 50%（前期已充分拉升）
- 堆量 + 犬牙交错（"狗啃式"，主力出货）

## 10. 砖型图三大定式量化参考

### 定式一：N 型起跳

```python
# 检测 N 型结构：低→高→低→红砖起跳
# 1. 近 N 日存在局部高点 local_high
# 2. local_high 后出现回调低点 local_low
# 3. 当日红砖出现在 local_low 附近
# 4. 红砖大小 >= min_brick_size
# 5. K 线无大上影线（upper_shadow / body < shadow_ratio）
```

### 定式二：横盘突破

```python
# 检测横盘后红砖突破
# 1. 近 N 日振幅 < range_ratio（如 5%）
# 2. 当日出现红砖突破横盘上沿
# 3. 红砖实体足够大
# 4. 无大上影线
```

### 定式三：上升延续

```python
# 检测上涨中回调后的红砖延续
# 1. 近 N 日整体上涨趋势（MA 向上）
# 2. 中间仅 1-2 根绿砖回调
# 3. 当日红砖恢复上涨
# 4. 突破前高
```

### 四块砖止盈

```python
# 连续红砖计数
# 1. red_count = 连续 brick > 0 的根数
# 2. red_count >= 4 → 生成减仓/清仓信号
# 3. 任何 brick < 0 → 立即离场信号
# 4. 新绿砖后再红砖 → red_count 重新从 1 开始
```

## 11. 形态排除规则

以下形态应在选股阶段直接排除：

| 排除条件 | 判定方法 |
|----------|----------|
| 大上影线（"大苗子"） | upper_shadow > body × 1.5 |
| 犬牙交错（"狗啃式"） | 近 N 日阴阳交替频率 > 80% 且无趋势 |
| 连续红砖无调整 | 连续 > 6 根红砖且中间无绿砖 |
| 白线距离黄线过远 | (zxdq - zxdkx) / zxdkx > 0.15 |
| 三波未突破 | 第三波高点 < 第二波高点 |
