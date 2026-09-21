# FedPhoenix longitudinal repeated-interaction diagnostic

This is a read-only seed-1 diagnostic. Accuracy is reported as context only and is not used to judge response transfer.

## Main comparison

| Metric | FedAvg | FedPhoenix |
|---|---:|---:|
| Final accuracy | 73.1200 | 73.9300 |
| Best accuracy | 75.1400 | 78.3500 |
| Median input change norm | 1.0893 | 2.5002 |
| Input change q25/q75 | 0.7975 / 1.5093 | 2.0128 / 4.7345 |
| Median dispatch-global norm | 0.0000 | 1.2311 |
| Median response change norm | 1.4103 | 1.5864 |
| Median response change ratio | 0.6579 | 0.6272 |
| Median secant gain | 1.2710 | 0.5418 |

## Similar-input transfer

| Subset | Algorithm | Count | Response cosine median | NRE median | Sign agreement | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| input_cos_ge_0_5 | FedAvg | 0 | NA | NA | NA | NA | NA |
| input_cos_ge_0_5 | FedPhoenix | 0 | NA | NA | NA | NA | NA |
| input_cos_ge_0_7 | FedAvg | 0 | NA | NA | NA | NA | NA |
| input_cos_ge_0_7 | FedPhoenix | 0 | NA | NA | NA | NA | NA |

## Participation-gap transfer

| Gap | Algorithm | Count | Response cosine median | NRE median | Sign agreement |
|---|---|---:|---:|---:|---:|
| gap_le_5 | FedAvg | 126 | -0.4951 | 0.8800 | 0.3968 |
| gap_le_5 | FedPhoenix | 126 | -0.4607 | 0.8616 | 0.4206 |
| gap_6_10 | FedAvg | 56 | -0.5059 | 0.8698 | 0.4107 |
| gap_6_10 | FedPhoenix | 56 | -0.5023 | 0.8667 | 0.4821 |
| gap_11_20 | FedAvg | 84 | -0.4394 | 0.8656 | 0.4881 |
| gap_11_20 | FedPhoenix | 84 | -0.4933 | 0.8649 | 0.5000 |
| gap_gt_20 | FedAvg | 21 | -0.4490 | 0.8726 | 0.5238 |
| gap_gt_20 | FedPhoenix | 21 | -0.5098 | 0.8772 | 0.5714 |

## Required questions

**Q1. Does FedPhoenix strengthen returning-client input variation? Yes.** The median input-change norm is 2.2953x FedAvg, and the median dispatch-global norm changes from 0.0000 to 1.2311. This establishes stronger excitation only.

**Q2. Does it create stronger local-update response change? Only in absolute norm, not proportionally.** The absolute response-change norm is 1.1249x, while normalized response-change ratio is 0.9533x and secant gain is 0.4263x FedAvg. The much larger input therefore does not produce a proportionally stronger response.

**Q3. Are adjacent observations reusable for similar input directions? Not testable at the required thresholds.** Both algorithms have 0 events at input cosine >=0.5 (and therefore >=0.7). Across all 287 pairs, response cosine is -0.4860/-0.4914 and NRE is 0.8726/0.8643 for FedAvg/FedPhoenix, which is poor unconstrained transfer but cannot replace the missing similar-direction test.

**Q4. Is reuse stronger under FedPhoenix? Not supported.** There are no critical similar-direction events. Even the weak input-cosine >0 subset has only 48 FedAvg and 14 FedPhoenix events, with median input cosine 0.0486/0.0865; those directions are not close enough for the intended transfer claim.

**Q5. Does previous support predict the next support sign better? No reliable evidence.** Overall sign agreement is 0.4355 for FedAvg and 0.4669 for FedPhoenix, both below 0.5; Pearson correlations are -0.9100/-0.9731. The required >=0.5 and >=0.7 subsets are empty.

**Q6. Does reuse decay rapidly with participation gap? No clear monotonic decay is visible.** Transfer is already poor in the <=5 bucket (negative response cosine and NRE near 0.86-0.88), and the longer-gap buckets do not worsen monotonically. This is weak-at-origin behavior rather than evidence for a useful signal that merely becomes stale.

**Q7. Final interpretation: C.** Critical similar-input subsets do not all contain at least 10 events.

A larger dispatch/input perturbation is explicitly not treated as evidence of longitudinal predictive value. FedPhoenix accuracy is likewise not part of the A/B/C rule.