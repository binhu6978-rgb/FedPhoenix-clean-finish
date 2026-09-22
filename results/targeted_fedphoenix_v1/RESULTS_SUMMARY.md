# TargetedFedPhoenix V1 screening

- Unit tests passed: True
- rho=0 exact parity passed: True
- CUDA OOM: False
- Sequential fallback: False
- Launcher wall time: 3564.3 seconds

## Accuracy versus fixed FedPhoenix reference

| Ratio | Peak | Peak round | Final | Last20 | Peak delta | Final delta | Last20 delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 78.2700 | 236 | 72.3000 | 72.2430 | -0.0800 | -1.6300 | +2.5780 |
| 0.50 | 78.1100 | 230 | 73.4000 | 72.1535 | -0.2400 | -0.5300 | +2.4885 |

## Mechanism statistics

### ratio=0.25

- Returning clients with history: 2900
- Clients targeted: 2577
- Clients with changed dispatch: 2577
- Actual target fraction: 0.200358
- Baseline-random overlap: 0.015729
- Replaced slots: 38047
- Cold/stale/no-valid/Q=0 client counts: 100/323/0/0
- Maximum history memory: 2112800 bytes

### ratio=0.50

- Returning clients with history: 2900
- Clients targeted: 2577
- Clients with changed dispatch: 2577
- Actual target fraction: 0.424289
- Baseline-random overlap: 0.015185
- Replaced slots: 80615
- Cold/stale/no-valid/Q=0 client counts: 100/323/0/0
- Maximum history memory: 2112800 bytes

## Strict conclusion

History-informed targeted reset did not improve peak accuracy over the fixed FedPhoenix reference in this 300-round screen.

This result evaluates reset targeting; it does not by itself establish that high update norm is a harmful-kernel score.
