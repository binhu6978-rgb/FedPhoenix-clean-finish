# InteractionDispatchV2 current results

Protocol: VGG16 on CIFAR-10, 100 clients, 10% participation, Dirichlet
non-IID alpha 0.3, local SGD for five epochs, seed 1, 300 communication
rounds. The V2 controller uses directional response only as an action gate and
uses `alpha = id2_step_ratio * obs_input_norm` for its dispatch magnitude.

## Correctness gates

- 13 V2 unit tests passed; the combined V2 and legacy InteractionDispatch
  suites passed 19 tests.
- In the 10-round no-action test (`id2_step_ratio=0`), client selections,
  accuracy values, and all ten global-model SHA256 hashes exactly matched
  FedAvg.

## Completed 300-round runs

| Method | Peak | Peak round | Final | Last-20 mean |
| --- | ---: | ---: | ---: | ---: |
| FedAvg | 75.14 | 212 | 73.12 | 70.59 |
| InteractionDispatch, old best (`ratio=0.02`) | 75.34 | 255 | 74.06 | 70.65 |
| InteractionDispatchV2, `ratio=0.10` | 74.90 | 212 | 73.39 | 70.80 |
| InteractionDispatchV2, `ratio=0.20` | 75.00 | 260 | 72.52 | 69.99 |

Neither completed V2 run reached the fixed promotion thresholds: peak 75.64
or last-20 mean 71.148. Therefore no 600-round promotion run was launched.

Both completed V2 runs had 531 active interventions (17.7% of 3,000 client
interactions), zero failed client updates, and a maximum persistent-history
footprint of 40,375,999,200 bytes (37.6 GiB).

## Coarse parameter groups

Across observation-bearing decisions, `features` dominated absolute support:
90.0% of events for ratio 0.10 and 89.6% for ratio 0.20. `classifier` and
`fc` were dominant in only about 6% and 4% of events respectively. Thus the
full-model support gate was mostly driven by convolutional feature parameters.

## Interrupted run

The `ratio=0.40` run was stopped by user request after 29 of 300 rounds. Its
partial trajectory is retained for reproducibility but is not used in the
formal comparison or ranking.
