# Same-experienced-direction shadow transfer diagnostic

This is a seed-1 falsification-style diagnostic. The main trajectory is ordinary FedAvg; every shadow result is excluded from aggregation.

## Isolation and parity

- 10-round parity passed: **True**
- Selected clients match: True
- Global-state SHA-256 matches every round: True
- Maximum accuracy absolute delta: 3.896484344068085e-07
- Shadow events exercised during parity: 10

## Ratio summary

| Ratio | Events | Cosine median | NRE median | Norm ratio median | Sign agreement | Pearson | Spearman | Active count | Helpful fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 273 | 0.0064 | 0.9205 | 11.1392 | 0.6044 | 0.4046 | 0.2581 | 58 | 0.5345 |
| 0.05 | 273 | 0.0117 | 0.8611 | 5.6775 | 0.6923 | 0.3784 | 0.4910 | 58 | 0.6034 |
| 0.10 | 273 | 0.0162 | 0.8096 | 3.6404 | 0.6996 | 0.3277 | 0.5360 | 58 | 0.5862 |
| 0.20 | 273 | 0.0242 | 0.7552 | 2.3376 | 0.7509 | 0.2214 | 0.6286 | 58 | 0.6724 |
| 0.40 | 273 | 0.0336 | 0.7124 | 1.5494 | 0.7509 | 0.3417 | 0.6925 | 58 | 0.7069 |

## Participation-gap breakdown

| Ratio | Gap | Count | Cosine median | NRE median | Sign agreement | Active count | Helpful fraction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.02 | gap_le_5 | 113 | 0.0071 | 0.9045 | 0.6106 | 20 | 0.5000 |
| 0.02 | gap_6_10 | 66 | 0.0067 | 0.9231 | 0.6364 | 14 | 0.6429 |
| 0.02 | gap_11_20 | 60 | 0.0046 | 0.9337 | 0.5833 | 14 | 0.3571 |
| 0.02 | gap_gt_20 | 34 | 0.0029 | 0.9436 | 0.5588 | 10 | 0.7000 |
| 0.05 | gap_le_5 | 113 | 0.0130 | 0.8398 | 0.7876 | 20 | 0.6500 |
| 0.05 | gap_6_10 | 66 | 0.0117 | 0.8556 | 0.6818 | 14 | 0.5714 |
| 0.05 | gap_11_20 | 60 | 0.0116 | 0.8738 | 0.5833 | 14 | 0.4286 |
| 0.05 | gap_gt_20 | 34 | 0.0086 | 0.8841 | 0.5882 | 10 | 0.8000 |
| 0.10 | gap_le_5 | 113 | 0.0175 | 0.7835 | 0.7522 | 20 | 0.5000 |
| 0.10 | gap_6_10 | 66 | 0.0180 | 0.8007 | 0.6667 | 14 | 0.6429 |
| 0.10 | gap_11_20 | 60 | 0.0133 | 0.8188 | 0.6167 | 14 | 0.4286 |
| 0.10 | gap_gt_20 | 34 | 0.0098 | 0.8259 | 0.7353 | 10 | 0.9000 |
| 0.20 | gap_le_5 | 113 | 0.0283 | 0.7378 | 0.7611 | 20 | 0.7000 |
| 0.20 | gap_6_10 | 66 | 0.0260 | 0.7495 | 0.7273 | 14 | 0.6429 |
| 0.20 | gap_11_20 | 60 | 0.0227 | 0.7668 | 0.7500 | 14 | 0.5714 |
| 0.20 | gap_gt_20 | 34 | 0.0215 | 0.7733 | 0.7647 | 10 | 0.8000 |
| 0.40 | gap_le_5 | 113 | 0.0338 | 0.7104 | 0.7434 | 20 | 0.6500 |
| 0.40 | gap_6_10 | 66 | 0.0355 | 0.7099 | 0.7576 | 14 | 0.6429 |
| 0.40 | gap_11_20 | 60 | 0.0353 | 0.7191 | 0.6833 | 14 | 0.6429 |
| 0.40 | gap_gt_20 | 34 | 0.0295 | 0.7244 | 0.8824 | 10 | 1.0000 |

## Interpretation boundary

This experiment does not ask whether the next natural global displacement resembles the old direction. It actively reuses the old experienced direction at a future nearby global state and tests whether the old response observation remains predictive. That is the direct test of Assumption 3.

## Required questions

**Q1. Is old h directionally consistent with future same-direction h? No meaningful full-vector consistency.** Median cosine only ranges from 0.0064 to 0.0336, i.e. nearly orthogonal even though the sign is slightly positive.

**Q2. Does magnitude transfer? No at the small local ratios.** At ratio 0.02/0.05, median norm ratios are 11.1392/5.6775 and median NRE is 0.9205/0.8611. Even ratio 0.40 retains NRE 0.7124 and norm ratio 1.5494.

**Q3. Does predicted support predict realized support? Weakly as a one-dimensional projection, but not as a reusable response vector.** Sign agreement rises from 0.6044 to 0.7509; Spearman rises from 0.2581 to 0.6925. This scalar signal coexists with near-zero full-vector cosine.

**Q4. Are controller-relevant predicted-active actions actually helpful? Not reliably in the most local regime.** All ratios have 58 predicted-active events. Helpful fraction is 0.5345 at 0.02 and 0.6034 at 0.05, versus 0.7069 at 0.40. The smallest step is near chance and the 0.05 result is only modest.

**Q5. Is transfer more reliable at smaller ratios?** No consistent joint cosine-up/NRE-down advantage is established for 0.02/0.05 over 0.20/0.40.

**Q6. Does transfer decay with participation gap? Only a mild vector-quality trend, not a clean action-level decay.** Cosine generally falls and NRE generally rises with gap, but sign agreement and predicted-active helpfulness are non-monotonic; the >20 active subset has only 10 events per ratio.

## Accounting and resources

- Valid scalar events: 1365 (attempted 1365)
- Skipped events: 0
- Maximum diagnostic CPU history: 4.136 GiB
- Representative upper-quartile alignment identity error bound across ratios: 0.00000002
- Context-only final/best FedAvg accuracy: 73.1200 / 75.1400

## Pre-registered decision rule

STRONG SUPPORT requires both ratios 0.02 and 0.05 to have at least 30 events and 10 predicted-active events, median cosine >=0.30, median NRE <=0.70, sign agreement >=0.60, helpful fraction >=0.60, and at least one support correlation >=0.20. LIMITED SUPPORT requires at least one of those ratios to have 20/5 events, cosine >0.10, NRE <0.90, sign/helpful fractions >0.50, and a positive support correlation. Otherwise the result is NOT SUPPORTED.

## Final judgment: NOT SUPPORTED

Neither fixed small ratio satisfies even the pre-registered limited-support combination of response, support, helpfulness, and count criteria.

Accuracy is not part of this judgment, and no InteractionDispatch or FedPhoenix behavior was modified.