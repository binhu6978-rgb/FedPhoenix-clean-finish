# TargetedFedPhoenix overnight focused screening

- Full regression: True
- rho=0 strict parity: True
- E5 rounds 1-150 FedPhoenix hash parity: True

## Complete results

| Rank | Run | Peak | Round | Final | Last20 | Last50 | R201-300 | Best roll5 | Roll5 end |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | E5_start151_gap10_ratio025_stronglayers | 78.9100 | 232 | 74.2700 | 72.1315 | 71.8220 | 71.7644 | 76.0700 | 262 |
| 2 | E2_gap10_ratio0125_all | 78.5600 | 218 | 75.1600 | 71.8830 | 71.8016 | 71.9883 | 75.7160 | 261 |
| 3 | E1_fresh_gap10_ratio025_all | 78.3400 | 290 | 73.7900 | 71.5510 | 71.3232 | 71.4089 | 74.8800 | 297 |
| 4 | E6_residual_gap10_ratio025_stronglayers | 78.3100 | 290 | 74.6700 | 72.1120 | 71.7852 | 71.7959 | 75.1760 | 233 |
| 5 | E3_gap20_ratio025_stronglayers | 78.2500 | 284 | 71.3400 | 71.8335 | 71.2170 | 71.3741 | 75.4180 | 261 |
| 6 | E4_gap10_ratio025_stronglayers | 77.8800 | 232 | 73.2800 | 71.1860 | 71.3232 | 71.2720 | 75.1880 | 261 |

## Strict mechanism conclusions

- A gap<=10 vs gap<=20: {'all_layers_delta_E1_vs_correctedV1': 0.0099945068359375, 'strong_layers_delta_E4_vs_E3': -0.37000274658203125}
- B rho=.125 vs .25 peak delta: 0.2200 pp
- C strong layers vs all peak delta: -0.0800 pp
- D fresh+strong combination: {'E4_vs_E1': -0.45999908447265625, 'E4_vs_E3': -0.37000274658203125}
- E delayed vs immediate peak delta: 1.0300 pp
- F residual vs update_norm peak delta: 0.4300 pp
- G any peak > 78.35: True ['E2_gap10_ratio0125_all', 'E5_start151_gap10_ratio025_stronglayers']
- H next dimension if none exceeded: None

No additional ratios, gaps, seeds, or long runs were launched.

## Protocol and execution checks

VGG/CIFAR-10, Dirichlet α=0.3, seed=1, 300 rounds per run. Peak test accuracy is primary; all accuracy values are percentages. The six fixed configurations ran as E1/E2, E3/E4, and E5/E6, with two simultaneous processes per batch. All six completed on their first attempt. No CUDA OOM, sequential fallback, failed client update, traceback, or runtime error occurred.

The full regression suite passed 80 tests, including 43 TargetedFedPhoenix tests. The rho=0 strict 10-round parity gate matched selected clients, task seeds, reset masks, accuracy, all global hashes, and final global hash. The SHA256 active smoke passed. E5's first 150 global state hashes exactly matched the existing observer-only FedPhoenix reference. Every run preserved global RNG state, kept persistent history on CPU as scalar kernel arrays, and used at most 2,112,800 bytes of history.

## Reference deltas and mechanism

Original FedPhoenix: peak 78.35 at round 218, final 73.93, last20 69.665. Corrected V1: peak 78.33 at round 284, final 73.54, last20 70.7595. Deltas below are percentage points.

| Run | Peak Δ vs FedPhoenix / V1 | Last20 Δ vs FedPhoenix / V1 | Eligible / targeted clients | Stale / cold fallback | Targeted / replaced slots | Actual target fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | −0.01 / +0.01 | +1.8860 / +0.7915 | 1,923 / 1,923 | 977 / 100 | 28,845 / 28,405 | 0.1495 |
| E2 | +0.21 / +0.23 | +2.2180 / +1.1235 | 1,923 / 1,923 | 977 / 100 | 11,538 / 11,364 | 0.0598 |
| E3 | −0.10 / −0.08 | +2.1685 / +1.0740 | 2,577 / 2,577 | 323 / 100 | 18,039 / 17,755 | 0.0935 |
| E4 | −0.47 / −0.45 | +1.5210 / +0.4265 | 1,923 / 1,923 | 977 / 100 | 13,461 / 13,249 | 0.0698 |
| E5 | +0.56 / +0.58 | +2.4665 / +1.3720 | 1,923 / 985 | 515 / 0* | 6,895 / 6,792 | 0.0357 |
| E6 | −0.04 / −0.02 | +2.4470 / +1.3525 | 1,923 / 1,923 | 977 / 100 | 13,461 / 13,249 | 0.0698 |

Each run had 2,900 returning client participations and 192,930 baseline reset slots. A targeted slot includes a history-selected position already present in the baseline; replaced slots count actual position changes. Per-layer targeted slots: E1 used features.14/.17/.20 (1,923 each) and .24/.27/.30/.34/.37/.40 (3,846 each); E2 used .24/.27/.30/.34/.37/.40 (1,923 each); E3 used .20 (2,577) and .24/.27/.30 (5,154 each); E4 and E6 used .20 (1,923) and .24/.27/.30 (3,846 each); E5 used .20 (985) and .24/.27/.30 (1,970 each).

*E5 recorded 1,500 tasks in rounds 1–150 as `before_start_round`, which takes precedence over the cold-start fallback counter. Its first 150 rounds still collected history. E6's valid residual history fraction was 98.4775%; 192,930 own-reset kernels were invalid and zero kernels lacked a clean peer in these ten-client rounds. Residual consensus excluded the client itself and reset peers, with data-size weights.

## Interpretation of the predefined questions

A. Gap ≤10 did **not consistently** improve peak over gap ≤20. E1 and corrected V1 were essentially tied for all layers (+0.01 pp); E4 was 0.37 pp below E3 for strong layers.

B. The conservative ρ=.125 improved peak by 0.22 pp over ρ=.25 in the matched gap10/all-layer comparison (E2 vs E1). It also improved final accuracy by 1.37 pp and last20 by 0.332 pp.

C. Restricting to the four predefined strong-history layers did not improve peak: E3 was 0.08 pp below corrected V1 at gap20, and E4 was 0.46 pp below E1 at gap10.

D. Fresh history plus strong layers showed no combined peak gain: E4 was 0.37 pp below E3 and 0.46 pp below E1.

E. Delaying targeting until round 151 was the strongest matched change. E5 exceeded immediate targeting E4 by 1.03 pp peak and 0.9455 pp last20. Its first 150 global hashes exactly matched FedPhoenix.

F. Clean residual improved peak by 0.43 pp and last20 by 0.926 pp over update_norm in the matched immediate gap10/strong-layer comparison (E6 vs E4). It did not exceed the best E5 or E2 peak.

G. E2 reached 78.56% (+0.21 pp vs FedPhoenix), and E5 reached 78.91% (+0.56 pp). E5 also led this screen on last20 and best rolling-5.

H. The conditional “if none exceeds” question does not apply. Within this fixed screen, intervention timing produced the largest matched peak improvement. No further run is part of this task.

E1 and E6 peaks differed by 0.03 pp: E6 had higher best rolling-5, rounds 201–300, and last20. E3 and E6 peaks differed by 0.06 pp: E3 had the higher best rolling-5, while E6 led on rounds 201–300 and last20. This is one seed, so small rank differences and all apparent improvements are descriptive rather than multi-seed statistical evidence.
