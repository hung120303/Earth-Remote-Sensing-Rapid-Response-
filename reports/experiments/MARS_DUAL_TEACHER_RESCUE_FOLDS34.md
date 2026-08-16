# Recall-anchored dual-teacher rescue: folds 3/4

The candidate can only raise the frozen Gaussian+DOFA score inside the released-detector rescue route.

- Anchor: `conservative_dense`
- Rescue weight: 0.333333
- Raised rows: 581 (19 positives, 562 negatives)
- AP delta: -0.000347
- Matched-FPR recall delta: -0.004593
- Paired 25 km-group AP interval: [-0.001139, +0.000689]

| Fold | AP delta | Recall delta |
|---|---:|---:|
| 3 | -0.000054 | -0.003958 |
| 4 | -0.000716 | -0.003916 |

Decision: **reject_deterministic_dual_teacher_rescue**
