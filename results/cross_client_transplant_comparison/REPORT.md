# Cross-client weight transplantation prestudy 结果

## 结论

实验观察到 **A-specific 的局部方向性响应**，但没有观察到强意义上的 endpoint recovery。

- 在更高统计功效的 `probe_fraction=0.5` 运行中，shadow training step 与 A 的正常 update 呈正相关：all `0.249`、conv `0.234`、classifier `0.481`。
- 同一个 shadow step 与 donor B 的 update 呈负相关：all `-0.074`、conv `-0.060`、classifier `-0.286`。分类器的 A-specific 方向信号最强。
- 但 recovery/pull score 的均值为负：all `-0.364`（95% CI `-0.384`–`-0.345`）、conv `-0.373`、classifier `-0.134`。`W_B→A` 通常比 donor 起点 `W_B` 更远离 A 的正常终点 `W_A`。
- 因此，数据支持“客户端数据会系统性改变输入模型的更新方向”，不支持“5 个 local epochs 足以把任意同轮 donor model 拉回 A 自己的正常 endpoint”。

## 数据完整性

| Run | Rounds | Probe events | Final accuracy | Best accuracy | Last-50 mean |
|---|---:|---:|---:|---:|---:|
| transplant, probe 0.2 | 500 | 1,000 | 73.86% | 76.51% | 73.50% |
| transplant, probe 0.5 | 500 | 2,500 | 73.86% | 76.51% | 73.50% |
| original FedAvg | 500 | 0 | 73.61% | 76.19% | 73.04% |

三份 stderr 均为空。两个 transplantation 运行的 500 点 global accuracy 逐点完全相同，最大绝对差为 0。这证明增加 shadow probe 数量没有改变 global trajectory。完整 500 轮未启用逐轮 full-state SHA-256；5-round smoke test 已完成该哈希隔离验证。

原生 FedAvg 与 transplantation 的总体性能接近，但不能把逐轮差异解释成 probe 效应。transplantation 脚本为了让 A baseline 与 shadow 使用相同 shuffle/dropout 轨迹，按 round/client 显式重置本地 RNG；原生 `main_fed.py` 使用连续全局 RNG。

## 五个核心问题

### 1. 同一个 target 换不同 donor 后，是否有相似 transformation pattern？

有弱到中等的 target signal，但不够强，不能称为 donor-invariant。`probe 0.5` 中每个 target 平均被 probe `25.0` 次，覆盖 `21.9` 个不同 donor。控制 donor dominant class 和阶段后，target identity 对 recovery 的 partial R² 为 `0.104`，对 shadow-step/A-update cosine 为 `0.108`。大部分变异仍来自同一 target 内的 event/donor/round 差异。

### 2. 不同 target 对同类 donor input 的处理是否明显不同？

差异可检测，但不占主导。这里“同类 donor”定义为 donor 本地标签分布的 dominant class 相同。target identity 在控制 donor 类型与阶段后解释约 9%–11% 的关键响应变异。

### 3. 同一个 target 的 response 在训练前中后期是否稳定？

存在中等稳定性，但不是强稳定。`probe 0.5` 中 target-level recovery 跨阶段 Pearson 相关为 `0.438`–`0.522`；shadow-step/A-update cosine 为 `0.407`–`0.458`。

阶段均值也在变化：all-parameter recovery 从前期 `-0.148`，变为中期 `-0.322`、后期 `-0.558`。后期 endpoint recovery 更弱。

### 4. Between-client variance 是否明显大于 within-client variance？

否，方向相反。`probe 0.5` 的 recovery between/within variance ratio 为 `0.099`，shadow-step/A-update cosine 为 `0.122`。within-client event variance 分别约为 between-target-means variance 的 `10.1` 倍和 `8.2` 倍。

### 5. Recovery/pull score 是否系统性非零？

是，但为系统性负值，不是期望的正 pull。all 参数仅 `6.0%` 事件为正，conv `5.4%`，classifier `48.2%`。classifier 的中位数接近 0，存在较强 A-specific 方向信号，但负尾部使均值显著为负。

## 参数范围差异

- Conv 参数主导 all-parameter 结果：recovery 明显为负，A-alignment 为正但较弱。
- Classifier 参数的转换最像 A：shadow step 与 A update cosine 为 `0.481`，与 B update 为 `-0.286`；endpoint 与 A update cosine `0.525`，高于与 B update 的 `0.260`。
- 即便如此，classifier recovery 均值仍为 `-0.134`，且正 recovery 事件比例为 `48.2%`。方向像 A 不等于端点距离一定更接近 `W_A`。

## 解释边界

- 每个事件只有一个 donor 输入和一次配对 stochastic trajectory；结论描述本实验定义下的平均现象。
- donor “同类”使用 dominant label class，是粗粒度 operational definition。
- `probe 0.2` 与 `0.5` 的总体统计高度一致；主要推断优先采用事件更多的 `probe 0.5`。
- 这些结果用于验证现象，不构成新方法或优化结论。
