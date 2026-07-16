# UNEP-augmented successor: exact MARS-S2L paper benchmark

Transparent post-test replay, not an untouched confirmation. Candidate scores were computed from the label-free cache before comparator outcomes were opened; dense masks remain the unchanged promoted v3 branch.

| View | AP | AP delta (95% CI) | Matched-FPR recall delta (95% CI) | FPR delta | IoU delta (95% CI) | Gates |
|---|---:|---:|---:|---:|---:|---|
| full | 0.67429 | +0.03327 ([+0.01409, +0.04795]) | +0.03309 ([+0.01961, +0.04624]) | -0.03740 | +0.05560 ([+0.03531, +0.07744]) | PASS |
| test_only_sites | 0.44823 | -0.00204 ([-0.04284, +0.03440]) | +0.01762 ([-0.01149, +0.04611]) | -0.04032 | +0.12090 ([+0.08528, +0.15537]) | FAIL |

At least one exact MARS-S2L v3 paper gate remains unresolved.
