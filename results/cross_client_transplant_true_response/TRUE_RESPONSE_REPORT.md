# True input-conditioned client response: post-hoc analysis

## Final judgment

**Weak/limited support**

Starting model 明显改变了 A 的 local training step，因此现象不能只用普通的 A-specific local gradient 解释。但 target-specific component 主要体现在 response magnitude；rotation/cancellation 的 target identity 解释度较弱，而且 donor identity 对部分方向指标的解释度更高。因此结果支持真实的 input-conditioned response，但不支持强、稳定、由 target identity 主导的 rich response operator。

## Exact derivation and QC

本分析没有重新训练，也没有加载模型权重。对每个 scope，现有日志严格提供了 `||a||`、`||b||`、`||s||`、`||b-a||`、`||e||`、`cos(s,a)` 和 `cos(s,b)`。由此精确恢复：

```text
<a,b> = (||a||² + ||b||² - ||b-a||²) / 2
<s,a> = ||s|| ||a|| cos(s,a)
<s,b> = ||s|| ||b|| cos(s,b)
||q||² = ||s||² + ||a||² - 2<s,a>
<q,b> = <s,b> - <a,b>
<q,a> = <s,a> - ||a||²
e = b + q
```

`||e||² = ||b+q||²` 的最大相对闭合误差为 all `1.38e-09`、conv `1.35e-09`、classifier `2.08e-08`。2,500 个事件均得到有限指标，没有负的 `||q||²`，也没有零 `||q||`。

## Overall metrics

| Scope | `||q||/||b||` | Cancellation | `||e||/||b||` | `cos(e,b)` | `cos(q,b)` | `cos(q,a)` |
|---|---:|---:|---:|---:|---:|---:|
| all | 2.066 | 0.150 | 2.286 | 0.462 | -0.081 | -0.401 |
| conv | 2.073 | 0.125 | 2.306 | 0.471 | -0.066 | -0.416 |
| classifier | 1.887 | 0.700 | 1.812 | 0.245 | -0.399 | -0.138 |

## Q1. `q` 是否显著非零？

是，而且不是小扰动。`response_change_ratio=||q||/||b||` 的均值为 all `2.066`（95% CI `1.987`–`2.145`）、conv `2.073`、classifier `1.887`。所有 2,500 个事件在三个 scope 上都严格大于 0。all 参数的中位数为 `1.680`，5%–95% 分位为 `0.717`–`4.450`。

这排除了“starting model 几乎不影响 A local step”的解释。

## Q2. `e` 是否只是 `b` 的简单缩放？

否。若只是正 scalar gain，`cos(e,b)` 应接近 1；实际均值为 all `0.462`、conv `0.471`、classifier `0.245`。classifier 中 `18.4%` 的事件甚至 `cos(e,b)<0`。与此同时 survival ratio 均值为 all `2.286`、classifier `1.812`，说明输出 perturbation 通常不是简单衰减，而是带有很大的旋转/新增分量。

`q` 几乎总是反向于 `b`：all `cos(q,b)` 均值 `-0.081`，classifier `-0.399`。cancellation coefficient 为 all `0.150`、classifier `0.700`。A 在 classifier 上抵消 donor 方向更强，但同时产生较大的非共线 response，所以 `||e||/||b||` 仍常大于 1。

## Q3. Target identity 解释多少 response variation？

Target identity 对 response magnitude 的解释明显高于旧 recovery 指标：

- all `response_change_ratio` target partial R² = `0.285`；between/within ratio = `0.383`。
- all `survival_ratio` 的 target partial R² 与 magnitude 结果相近（详见 CSV）。
- 方向/旋转指标较弱：`survival_cosine` target partial R² = `0.157`，`response_cosine` = `0.129`；对应 between/within ratio 分别为 `0.167` 和 `0.136`。

Donor identity 对部分方向指标更强：all cancellation donor partial R² = `0.300`，response cosine donor partial R² = `0.413`。所以 response magnitude 有清晰 target-specific component，但 rotation/cancellation 不是由 target identity 单独主导。

## Q4. Response 是否跨 participation 稳定？

Target-specific magnitude 的数值具有较强线性延续性：all `response_change_ratio` 的 target-level 跨阶段 Pearson 为 `0.787`–`0.854`；但直接衡量排序的 Spearman 只有 `0.338`–`0.591`，所以不能称为强 rank stability。阶段均值分别为 `1.496`、`1.961`、`2.573`。

方向 ordering 更弱：target-level `survival_cosine` Pearson `0.411`–`0.556`、Spearman `0.297`–`0.478`；`response_cosine` Pearson `0.343`–`0.485`、Spearman `0.201`–`0.384`。相反，donor-level `response_cosine` 更稳定，Pearson `0.719`–`0.834`、Spearman `0.644`–`0.751`。

因此稳定的是“某个 target 对 starting-model change 有多敏感”，不是一个完全稳定的 target-specific rotation map。

## Q5. Classifier signal 是普通 local gradient 还是 conditioned response？

不是普通 gradient 就能解释完。Classifier 的 `||q||/||b||` 均值 `1.887`，cancellation `0.700`，`cos(q,b)` `-0.399`；starting point 对 A 的 update 有大幅影响。Classifier response magnitude 的 target partial R² 为 `0.291`，并具有稳定 target ordering。

但 classifier rotation 的 target-specific component 仍有限：`survival_cosine` target partial R² `0.059`，`response_cosine` `0.139`；donor identity 对这两个方向指标分别解释 `0.316` 和 `0.384`。

## Interpretation boundary

- 所有新指标都由已保存几何量解析恢复，没有近似模型权重或重新训练。
- Partial R² 是 categorical fixed-effect descriptive statistic，不应解释为因果效应。
- Between variance 指 entity mean 的方差；within variance 是同一 entity 内 event residual 的 pooled variance。
- “Strong support”需要 target identity 同时主导 response magnitude 和稳定 rotation。本结果只满足前者，因此最终判断为 **Weak/limited support**。
