# FedPhoenix client-specialization diagnostic

This observer-only run preserves the original FedPhoenix training trajectory.

## Accuracy

- Peak: 78.349998% at round 218
- Final: 73.930000%
- Last-20 mean: 69.665000%

## Persistence

| Score | Scope | Events | Same overlap | Cross overlap | Random | Identity advantage | Same Spearman | Cross Spearman | Rank advantage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| update_norm | all_layers | 37700 | 0.145325 | 0.072318 | 0.016069 | 0.073007 | 0.475545 | 0.302155 | 0.173390 |
| update_norm | reset_eligible_layers | 33320 | 0.152859 | 0.073877 | 0.016125 | 0.078982 | 0.495815 | 0.306680 | 0.189135 |
| residual | all_layers | 37700 | 0.141157 | 0.078028 | 0.016069 | 0.063129 | 0.475496 | 0.330093 | 0.145402 |
| residual | reset_eligible_layers | 33320 | 0.148308 | 0.079591 | 0.016125 | 0.068717 | 0.496552 | 0.337642 | 0.158911 |
| norm_residual | all_layers | 37700 | 0.028571 | 0.018277 | 0.016069 | 0.010294 | 0.056083 | 0.003905 | 0.052179 |
| norm_residual | reset_eligible_layers | 33320 | 0.029911 | 0.017990 | 0.016125 | 0.011920 | 0.062201 | 0.004308 | 0.057893 |
| angular | all_layers | 37700 | 0.021960 | 0.018130 | 0.016069 | 0.003830 | 0.000950 | 0.003467 | -0.002517 |
| angular | reset_eligible_layers | 33320 | 0.022460 | 0.017945 | 0.016125 | 0.004515 | 0.001042 | 0.003794 | -0.002753 |

## Scope

The diagnostic measures longitudinal predictiveness of historical kernel rankings. It does not establish that resetting high-scoring kernels improves utility.
