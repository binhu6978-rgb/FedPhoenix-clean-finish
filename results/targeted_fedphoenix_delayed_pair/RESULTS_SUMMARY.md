# TargetedFedPhoenix delayed confirmation pair

Both runs use VGG/CIFAR-10, Dirichlet alpha=0.3, seed=1, and 300 rounds.

## Correctness

- Full regression, rho=0 parity, active smoke: passed.
- P1/P2 rounds 1-150 selected clients, task seeds, and global hashes: exactly matched FedPhoenix.
- OOM: False; sequential fallback: False.

## Complete metrics

| Run | Peak | Peak round | Final | Last20 | Last50 | R151-300 | R201-300 | Best rolling-5 (end) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E5 | 78.9100 | 232 | 74.2700 | 72.1315 | 71.8220 | 71.2820 | 71.7644 | 76.0700 (262) |
| P1_start151_gap10_ratio0125_all | 77.9200 | 232 | 74.1500 | 71.8200 | 71.2418 | 71.0371 | 71.5411 | 75.5080 (262) |
| P2_start151_gap10_ratio025_all | 78.3400 | 230 | 73.4600 | 71.1875 | 71.1316 | 70.9051 | 71.3383 | 75.1880 (261) |

## Matched differences (left minus right; percentage points)

- P1_vs_E2_delayed_at_ratio0125: peak -0.6400, final -1.0100, last20 -0.0630, rolling-5 -0.2080.
- P2_vs_E1_delayed_at_ratio025: peak +0.0000, final -0.3300, last20 -0.3635, rolling-5 +0.3080.
- P2_vs_E5_all_vs_strong_layers: peak -0.5700, final -0.8100, last20 -0.9440, rolling-5 -0.8820.
- P1_vs_P2_conservative_ratio: peak -0.4200, final +0.6900, last20 +0.6325, rolling-5 +0.3200.

## Answers to the predefined questions

1. P1 exceeds original FedPhoenix peak: no (-0.4300 pp).
2. P1 exceeds E2 peak: no (-0.6400 pp).
3. P1 exceeds E5 peak: no (-0.9900 pp).
4. P2 exceeds E1 peak: no (+0.0000 pp).
5. P2 exceeds E5 peak: no (-0.5700 pp).
6. Delayed targeting is not consistently beneficial at either intensity: P1 trails E2 on peak, final, last20 and rolling-5; P2 and E1 tie on peak at displayed precision, while P2 improves rolling-5 but trails on final and last20.
7. All-layer targeting does not outperform strong-layer restriction in the delayed setting: P2 trails E5 on peak, rolling-5, final and last20.
8. Candidate for a later 1200-round run: E5. This is a single-seed screening recommendation, not a significance claim; no long run is started here.

## Mechanism

- P1_start151_gap10_ratio0125_all: history/eligible/targeted 2900/1923/985; cold/stale/before-start 0/515/1500; reset/targeted/replaced slots 192930/5910/5812; actual fraction 0.030633; maximum history 2112800 bytes; failed clients 0; layers: features.24=985, features.27=985, features.30=985, features.34=985, features.37=985, features.40=985.
- P2_start151_gap10_ratio025_all: history/eligible/targeted 2900/1923/985; cold/stale/before-start 0/515/1500; reset/targeted/replaced slots 192930/14775/14558; actual fraction 0.076582; maximum history 2112800 bytes; failed clients 0; layers: features.14=985, features.17=985, features.20=985, features.24=1970, features.27=1970, features.30=1970, features.34=1970, features.37=1970, features.40=1970.

## Candidate for a later long run

Primary peak ranking among E5/P1/P2: E5, P2, P1.
Top-two peak gap: 0.5700 pp. The primary peak separates the top two by at least 0.10 pp.
This report uses one seed and does not initiate a long run.
