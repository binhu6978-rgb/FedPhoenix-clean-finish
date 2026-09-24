# SymmetryFedPhoenix horizontal-flip probe (seed 1)

Code commit: `8995b20d87028a2481ecc6c5f86bbc61e691a055` on `codex/symmetry-fedphoenix`.

All three comparisons use CIFAR-10/VGG, 100 clients, 10% participation, Dirichlet beta 0.3, seed 1, five local epochs, local batch 50, SGD with learning rate 0.01 and momentum 0.5, and the original FedPhoenix task resets (`reset=1/64`, `FP_conv=1000`, `FP_fc=0`, `remethod=ori_normal`). The existing B0 CSV is read only through round 300; B0 was not rerun. The two new 300-round experiments ran concurrently on GPU 0.

| Experiment | Peak (%) | Peak round | Top-10 mean (%) | Rounds 251–300 mean (%) | Round 300 (%) |
|---|---:|---:|---:|---:|---:|
| B0, original FedPhoenix | 78.35 | 218 | 77.557 | 70.749 | 73.93 |
| FedPhoenix + input HFlip | 83.12 | 290 | 82.300 | 75.773 | 77.09 |
| SymmetryFedPhoenix, model-view alt | 81.14 | 255 | 80.534 | 73.640 | 76.50 |

Input HFlip adds only `RandomHorizontalFlip(p=0.5)` to the CIFAR-10 training transform. Model-view alt has no input augmentation. Both new experiments have exactly the same selected clients and task seeds as B0 in rounds 1–300. Model-view alt used F in 1,502 of 3,000 client participations (50.07%); its reported and independently checked alternation violations are zero.

The model-view alt advantage over B0 is +2.79 percentage points in peak accuracy and +2.89 points in the last-50-round mean. Input HFlip remains higher by 1.98 and 2.13 points on those respective measures. These are single-seed results; the accuracy curve shows substantial round-to-round variation. No 500- or 1000-round continuation was launched.

`comparison_summary.json` gives full-precision metrics and protocol checks. `round_accuracy_comparison.csv` and `accuracy_curve_300.png` give the three aligned trajectories. `summarize_300.py` regenerates these from the experiment CSVs and B0. Each formal run directory contains its CSV, config, stdout, and stderr. The `smoke_none` and `smoke_alt` directories record the short correctness runs.
