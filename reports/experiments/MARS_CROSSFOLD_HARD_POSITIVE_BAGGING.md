# Recall-focused crossfold-bagged MARS scene head

Only current-head hard positives receive extra fitting weight; all OOF members exclude the held fold.

- Hard-positive multiplier: 2.0
- Aggregation: `mean_logit`
- Bagged weight around current head: 0.100
- AP delta vs current: +0.00152
- Recall delta vs current: +0.00026
- Paired-site AP interval vs current: [+0.00102, +0.00201]

| Fold | AP delta vs current | Recall delta vs current |
|---|---:|---:|
| 0 | +0.00086 | +0.00000 |
| 1 | +0.00104 | +0.00000 |
| 2 | +0.00166 | +0.00000 |
| 3 | +0.00258 | +0.00264 |
| 4 | +0.00189 | +0.00000 |

Reject hard-positive crossfold bagging before paper-cache scoring.
