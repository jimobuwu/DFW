---
name: b1-stock-selection
description: >-
  A股 B1 波段选股策略的完整知识库：KDJ 分位过滤、知行均线多头排列、周线趋势过滤、
  成交量非阴线过滤、砖型图形态选股、AI 图表复评打分。
  用于修改或扩展 select_stock.py 中的选股逻辑、调参、新增策略、回测优化。
  当用户提及 B1 策略、选股、初选、知行线、KDJ、砖型图、量化选股时自动触发。
---

# B1 波段选股策略技能

## 策略总览

B1 是本项目的核心预选策略，目标是在 A 股中筛选出 **处于波段启动初期、具备主力建仓迹象** 的股票。完整流水线：

```
拉取日线数据 → 流动性池筛选 → B1 量化初选 → 导出 K 线图 → AI 图表复评 → 推荐输出
```

## 核心文件

| 文件 | 职责 |
|------|------|
| `pipeline/Selector.py` | 所有 Filter 和 Selector 的实现（B1Selector、BrickChartSelector） |
| `pipeline/select_stock.py` | 选股主入口，编排策略执行流程 |
| `pipeline/pipeline_core.py` | 数据预处理、流动性池构建、并行预计算 |
| `pipeline/schemas.py` | Candidate / CandidateRun 数据结构 |
| `config/rules_preselect.yaml` | 所有策略参数配置 |
| `agent/prompt.md` | AI 复评评分 prompt（趋势/位置/量价/异动四维度） |

## B1 策略四重过滤条件

B1Selector 由 4 个独立 Filter 组成，全部通过才选入：

### 1. KDJ 分位过滤（KDJQuantileFilter）

捕捉 J 值处于历史低位的股票（超跌信号）。

- **条件**：`J < j_threshold` OR `J <= 历史累积 j_q_threshold 分位`
- **默认参数**：`j_threshold=15.0`, `j_q_threshold=0.10`, `kdj_n=9`
- **向量化实现**：使用 `expanding().quantile()` 保证无未来信息泄漏
- **Numba 加速**：KDJ 递推 `K[i] = 2/3 * K[i-1] + 1/3 * RSV[i]`，`J = 3K - 2D`

### 2. 知行线条件过滤（ZXConditionFilter）

确认股价在长期均线之上，趋势向好。

- **条件**：`close > zxdkx`（价格在均线上）且 `zxdq > zxdkx`（快线在均线上）
- **zxdq**：double-EWM 平滑线，`ewm(span=10).ewm(span=10)`
- **zxdkx**：四均线平均 `(MA_m1 + MA_m2 + MA_m3 + MA_m4) / 4`
- **默认参数**：`zx_m1=14, zx_m2=28, zx_m3=57, zx_m4=114`

### 3. 周线均线多头排列过滤（WeeklyMABullFilter）

确认中长期趋势健康。

- **条件**：周线 `MA_short > MA_mid > MA_long`
- **默认参数**：`wma_short=10, wma_mid=20, wma_long=30`（周线周期）
- **实现**：日线 → ISO 周分组 → 周线收盘价 → 滚动均线 → ffill 回日线

### 4. 成交量最大日非阴线过滤（MaxVolNotBearishFilter）

排除主力出货迹象。

- **条件**：过去 n 日内成交量最大的那天 `close >= open`（非阴线）
- **默认参数**：`n=20`
- **Numba 加速**：O(N×n) 滚动窗口

## 砖型图策略（BrickChartSelector）

作为 B1 的补充策略，捕捉砖型图红绿柱转换信号：

### 五重过滤

1. **涨幅限制**：今日涨幅 < `daily_return_threshold`（排除涨停追高）
2. **红柱确认**：今日 brick > 0
3. **绿柱翻红**：昨日 brick < 0
4. **力度判断**：`今日红柱 >= brick_growth_ratio × |昨日绿柱|`
5. **连续绿柱**：红柱前至少 `min_prior_green_bars` 根绿柱

### 附加条件（可选）

- `close < zxdq × zxdq_ratio`：价格未过度偏离知行线
- `zxdq > zxdkx`：知行线多头
- 周线均线多头排列

## 流动性池构建

- 使用 `TopTurnoverPoolBuilder`
- 按每日滚动成交额（`turnover_n`，窗口 `n_turnover_days=43` 日）跨市场排名
- 取 top_m（默认 5000）只股票进入候选池
- `turnover = (open + close) / 2 × volume`

## 数据预处理流水线

`MarketDataPreparer` 执行以下步骤（多进程并行）：

1. 列名统一小写，日期转 datetime，排序
2. warmup 切片（确保均线有足够历史数据）
3. 计算 `turnover_n`
4. 设置 DatetimeIndex
5. 调用 `selector.prepare_df()` 预计算所有指标列

## AI 复评评分维度

量化初选后，导出 K 线图交给 AI（智谱 GLM-4V-Flash）做视觉评分：

| 维度 | 权重 | 评分标准 |
|------|------|----------|
| 趋势结构 | 0.20 | 均线多头排列、上拐、交叉 |
| 价格位置 | 0.20 | 中低位突破 > 接近前高 > 高位过热 |
| 量价行为 | 0.30 | 上涨放量+回调缩量最佳，放量大阴最差 |
| 前期建仓异动 | 0.30 | 异常放量中大阳+突破平台最佳 |

判定：`total_score >= 4.0` → PASS，`3.2 ~ 4.0` → WATCH，`< 3.2` → FAIL

## 调参指南

修改 `config/rules_preselect.yaml` 中的参数：

```yaml
b1:
  enabled: true
  zx_m1: 14         # 知行线短周期（越小越敏感）
  zx_m2: 28
  zx_m3: 57
  zx_m4: 114        # 知行线长周期（越大越稳定）
  j_threshold: 15.0  # J 值绝对阈值（越高选出越多）
  j_q_threshold: 0.10 # J 值历史分位阈值（越高选出越多）
```

- 想选更多股票：提高 `j_threshold` 或 `j_q_threshold`
- 想更严格：降低阈值，或调大 `zx_m4`
- `top_m` 越大，流动性池越宽，覆盖更多小盘股

## 新增策略模板

在 `Selector.py` 中新增策略的标准模式：

1. 创建 `@dataclass(frozen=True)` 的 Filter 类，实现 `__call__` + `vec_mask`
2. 创建继承 `PipelineSelector` 的 Selector 类，实现 `prepare_df`
3. 在 `select_stock.py` 中添加 `run_xxx()` 函数
4. 在 `rules_preselect.yaml` 中添加配置段
5. 在 `run_preselect()` 的策略执行段落中调用

## 详细技术规格

完整的指标计算公式、Numba 函数签名、Filter/Selector 接口定义见 [reference.md](reference.md)。
