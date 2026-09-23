# TargetedFedPhoenix 1000-round candidate comparison

VGG/CIFAR-10, Dirichlet alpha=0.3, seed=1; L2 and L3 each completed 1000 rounds.
FedPhoenix is an existing historical log, not a newly generated paired run.

## Correctness

- Full regression tests, rho=0 strict parity, and active deterministic smoke passed.
- L2 rounds 1-150 exactly matched the FedPhoenix diagnostic in selected clients, task seeds, and global state hashes.
- OOM: False; sequential fallback: False.

## Accuracy

| Run | Peak (round) | Final | Last20 | Last50 | Last100 | Best rolling5 (end) | Best rolling10 (end) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical FedPhoenix | 82.6100 (997) | 80.5900 | 80.5935 | 79.5490 | 79.2956 | 81.7900 (971) | 81.5380 (974) |
| L2 / E5 | 82.6200 (997) | 79.1900 | 80.9620 | 79.9966 | 78.6558 | 82.1900 (990) | 81.6720 (973) |
| L3 / E2 | 82.1500 (783) | 81.8600 | 79.8825 | 79.6274 | 77.5759 | 81.7020 (906) | 81.2340 (973) |

## Stage accuracy (L2 minus L3 in pp)

| Rounds | L2 | L3 | Difference |
| --- | ---: | ---: | ---: |
| 1-150 | 60.5306 | 60.6313 | -0.1007 |
| 151-300 | 71.2820 | 71.3654 | -0.0834 |
| 301-500 | 73.7749 | 73.5415 | +0.2334 |
| 501-750 | 74.4618 | 74.6470 | -0.1852 |
| 751-1000 | 77.5637 | 77.1926 | +0.3711 |

Longer late-period means (L2 / L3 / difference): rounds 501-1000 76.0127 / 75.9198 / +0.0930; rounds 771-1000 77.4132 / 77.0089 / +0.4043; rounds 901-1000 78.6558 / 77.5759 / +1.0799.

## L2 - historical FedPhoenix (percentage points)

peak_accuracy +0.0100, final_accuracy -1.4000, last20_mean +0.3685, last50_mean +0.4476, last100_mean -0.6398, best_rolling5_mean +0.4000, best_rolling10_mean +0.1340.

## L3 - historical FedPhoenix (percentage points)

peak_accuracy -0.4600, final_accuracy +1.2700, last20_mean -0.7110, last50_mean +0.0784, last100_mean -1.7197, best_rolling5_mean -0.0880, best_rolling10_mean -0.3040.

## L2 - L3 (percentage points)

peak_accuracy +0.4700, final_accuracy -2.6700, last20_mean +1.0795, last50_mean +0.3692, last100_mean +1.0799, best_rolling5_mean +0.4880, best_rolling10_mean +0.4380.

## Stage mechanism

### L2 / E5

| Rounds | Targeted clients | Targeted slots | Replaced slots | Actual fraction | Mean gap | Accuracy mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 151-300 | 985 | 6895 | 6792 | 0.072840 | 4.7076 | 71.2820 |
| 301-500 | 1295 | 9065 | 8920 | 0.079490 | 4.6942 | 73.7749 |
| 501-750 | 1622 | 7535 | 7415 | 0.070898 | 4.5863 | 74.4618 |
| 751-1000 | 130 | 260 | 253 | 0.006007 | 4.7000 | 77.5637 |

Per-layer targeted slots: features.20=2541, features.24=6072, features.27=7078, features.30=8064.
Total clients with history / history eligible / targeted: 9900 / 6476 / 4032.
Total reset / targeted / random / replaced slots: 456530 / 23755 / 432775 / 23380.
Last round with targeting: 770; targeted slots in rounds 771-1000: 0.
Maximum history memory: 2112800 bytes; failed client updates: 0.

### L3 / E2

| Rounds | Targeted clients | Targeted slots | Replaced slots | Actual fraction | Mean gap | Accuracy mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-150 | 938 | 5628 | 5542 | 0.057271 | 4.4947 | 60.6313 |
| 151-300 | 985 | 5910 | 5822 | 0.062434 | 4.7076 | 71.3654 |
| 301-500 | 1295 | 7770 | 7635 | 0.068134 | 4.6942 | 73.5415 |
| 501-750 | 1622 | 8503 | 8363 | 0.080006 | 4.5863 | 74.6470 |
| 751-1000 | 1636 | 3509 | 3445 | 0.081077 | 4.5954 | 77.1926 |

Per-layer targeted slots: features.24=3974, features.27=4477, features.30=4970, features.34=5457, features.37=5966, features.40=6476.
Total clients with history / history eligible / targeted: 9900 / 6476 / 6476.
Total reset / targeted / random / replaced slots: 456530 / 31320 / 425210 / 30807.
Last round with targeting: 1000; targeted slots in rounds 771-1000: 2989.
Maximum history memory: 2112800 bytes; failed client updates: 0.

## Answers and selection

1. Existing FedPhoenix reference: peak 82.6100 at 997; final 80.5900; last20/50/100 80.5935/79.5490/79.2956.
2. E5: peak 82.6200 at 997; final 79.1900; last20/50/100 80.9620/79.9966/78.6558.
3. E2: peak 82.1500 at 783; final 81.8600; last20/50/100 79.8825/79.6274/77.5759.
4. E5 exceeds the historical FedPhoenix peak by +0.0100 pp. The margin is only 0.01 pp and should be treated as essentially tied.
5. E2 exceeds the historical FedPhoenix peak: no (-0.4600 pp).
6. Higher E5/E2 peak: E5 (L2); E5 minus E2 +0.4700 pp.
7. E2 has the higher final by +2.6700 pp; E5 has the higher last100 by +1.0799 pp.
8. E5's relative accuracy advantage is present in rounds 751-1000 (+0.3711 pp) and 901-1000 (+1.0799 pp). E5 performs no targeting after round 770; the late difference cannot be attributed to ongoing targeted resets.
9. E2 continues sparse targeting through round 1000, but its last100 is -1.0799 pp relative to E5. This run does not show better sustained late accuracy.
10. Recommended seed=1 candidate for a later multi-seed evaluation: E5 (L2). E5 leads peak, rolling means and last100; E2 leads at the single final checkpoint. This is a selection from one seed, not a statistical significance claim.
No additional training is started by this launcher.
