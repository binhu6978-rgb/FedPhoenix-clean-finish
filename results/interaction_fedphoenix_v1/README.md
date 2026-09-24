# InteractionFedPhoenix V1 results

Formal setting: CIFAR10/VGG, 100 clients, participation fraction 0.1, seed 1,
1000 rounds, fixed `data/cifar10_100_noniidCase5_beta0.3.json` partition,
no input augmentation, local epochs 5, local batch size 50, SGD with learning
rate 0.01 and momentum 0.5, `FP_conv=1000`, reset rate `1/64`,
`remethod=ori_normal`. Both V1 runs use `rho=0.05` and `max_gap=20`; only
`ifp_space` differs. The per-run config JSON files contain all arguments.

| Method | Peak test accuracy | Peak round | Round 1000 accuracy |
| --- | ---: | ---: | ---: |
| Historical FedPhoenix, first 1000 rounds | 82.61% | 997 | 80.59% |
| InteractionFedPhoenix reset-aware | 82.34% | 994 | 73.76% |
| InteractionFedPhoenix head | 82.51% | 783 | 80.56% |

The historical baseline is read from
[`result_other_method/cifar10/0.3/vgg_FedPhoenix.log`](../../result_other_method/cifar10/0.3/vgg_FedPhoenix.log),
which contains 1200 rounds. It was not rerun. That log has no saved config or
partition-file identifier, so partition identity cannot be independently
verified from the log alone.

Formal outputs:

- Reset-aware: [`metrics CSV`](formal_reset_aware_1000/metrics/cifar10_vgg_InteractionFedPhoenix_seed1_ifp_reset_aware_1000.csv), [`config`](formal_reset_aware_1000/metrics/cifar10_vgg_InteractionFedPhoenix_seed1_ifp_reset_aware_1000_config.json), [`stdout`](formal_reset_aware_1000/stdout.log).
- Head: [`metrics CSV`](formal_head_1000/metrics/cifar10_vgg_InteractionFedPhoenix_seed1_ifp_head_1000.csv), [`config`](formal_head_1000/metrics/cifar10_vgg_InteractionFedPhoenix_seed1_ifp_head_1000_config.json), [`stdout`](formal_head_1000/stdout.log), [`stderr`](formal_head_1000/stderr.log).
- A 30-round reset-aware active smoke and its metrics are in [`formal_smoke`](formal_smoke/).

Across 10,000 client selections per formal run, reset-aware recorded 1,388
active interventions (13.88%) and head recorded 1,611 (16.11%). The full
round-level diagnostics, including masks, action sizes, and history memory,
are in the metrics CSV files.
