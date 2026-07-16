# Calibrated spatial-Prithvi ensemble: exact MARS-S2L v3 benchmark

Transparent post-test architecture evaluation against the exact reconstructed v3 paper comparator.

| View | Exact v3 AP | Candidate AP | AP delta (95% CI) | Matched-FPR recall delta (95% CI) | Exact v3 IoU | Candidate IoU | IoU delta (95% CI) | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full | 0.641020 | 0.676102 | +0.035082 ([+0.016224, +0.049491]) | +0.031440 ([+0.019979, +0.048110]) | 0.324365 | 0.379964 | +0.055598 ([+0.034660, +0.077298]) | PASS |
| test_only_sites | 0.450274 | 0.467027 | +0.016753 ([-0.023301, +0.052032]) | +0.022026 ([-0.008066, +0.051119]) | 0.171562 | 0.292462 | +0.120900 ([+0.085772, +0.155426]) | FAIL |

Official paper revision: [v3](https://arxiv.org/html/2511.21777v3).

The frozen ensemble does not beat the exact MARS-S2L v3 comparator on every required gate.
