# Corrected TargetedFedPhoenix V1 rho=0.25

- Full regression tests passed: True
- rho=0 exact parity passed: True
- Active deterministic smoke passed: True

## Accuracy

| Method | Peak | Peak round | Final | Last20 |
| --- | ---: | ---: | ---: | ---: |
| FedPhoenix | 78.3500 | 218 | 73.9300 | 69.6650 |
| Biased V1 rho=.25 | 78.2700 | 236 | 72.3000 | 72.2430 |
| Corrected V1 rho=.25 | 78.3300 | 284 | 73.5400 | 70.7595 |

## Mechanism

- Targeted/random/replaced slots: 38655/154275/38074
- Actual target fraction: 0.200358
- Cold/stale/no-valid: 100/323/0
- Baseline-random overlap: 0.015030
- Maximum history memory: 2112800 bytes

## Index-bias sanity

- Prefix-retention fraction: 0.106228
- Donor normalized index mean/median/q25/q75: 0.496988/0.493151/0.244618/0.749511
- Kept normalized index mean/median/q25/q75: 0.499208/0.499022/0.248532/0.751468

## Strict conclusion

late stability improvement persists after removing index-selection bias.
