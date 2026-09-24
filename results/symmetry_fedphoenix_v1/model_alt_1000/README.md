# SymmetryFedPhoenix-alt, 1000 rounds, seed 1

Diagnostics-only code commit: `286dce3766fbb39549c2de6171452a18f0d1d79d`.

This run uses the same CIFAR-10/VGG, client partition, client sampling, FedPhoenix reset schedule and task seeds as the stored B0 run. It uses `sym_view=alt` and `input_hflip=0`; the training transform has no input augmentation. B0 was read from its existing CSV through round 1000 and was not rerun. The first 300 accuracy values and view decisions are exactly equal to the earlier SymmetryFedPhoenix-alt 300-round run.

| Metric | B0 | Model-alt | Difference (pp) |
|---|---:|---:|---:|
| Peak accuracy | 82.740% (round 953) | 86.660% (round 810) | +3.920 |
| Top-10 accuracy mean | 82.433% | 86.452% | +4.019 |
| Mean, rounds 1–300 | 65.703% | 67.908% | +2.206 |
| Mean, rounds 301–500 | 73.603% | 76.437% | +2.834 |
| Mean, rounds 501–900 | 75.865% | 80.468% | +4.604 |
| Mean, rounds 901–1000 | 79.842% | 83.560% | +3.718 |
| Round 1000 | 76.140% | 85.540% | +9.400 |

All 1000 rounds and 10,000 client events are present. Selected clients and task seeds match B0 in every round. F was used in 50.02% of participations, the returning-client rate was 99.00%, and there were no reported or independently detected alternation violations. The visit-index distribution is 100 events each at visits 1, 2, 3 and 4; 9,600 events at visit 5 or later, including 9,100 at visit 10 or later.

| Reset-kernel diagnostic | I view | F view |
|---|---:|---:|
| Reset action norm, mean | 0.87575 | 0.88247 |
| Reset response norm, mean | 0.23782 | 0.23830 |
| Recovery coefficient, mean / median | 0.001602 / 0.001301 | 0.001538 / 0.001396 |
| Recovery cosine, mean / median | 0.005307 / 0.005692 | 0.005144 / 0.006068 |
| Residual ratio, mean / median | 1.11097 / 1.07512 | 1.11524 / 1.07360 |

The reset-response summaries for I and F are close. These particular kernel-level diagnostics do not by themselves explain the accuracy gain. They are observational statistics from a single seed, without a separate causal intervention for each view.

`cifar10_vgg_SymmetryFedPhoenix_seed1_model_alt_1000.csv` contains all per-round accuracy, interaction and I/F diagnostic summaries. `model_alt_1000_client_events.csv` contains the 10,000 scalar client events. The parent directory's `model_alt_1000_summary.json` includes full precision overall and training-stage summaries; `model_alt_1000_round_comparison.csv` and `model_alt_1000_visit_response.csv` hold aligned data for later figures. `stdout.log`, `stderr.log` and the config JSON preserve the actual run output and settings.
