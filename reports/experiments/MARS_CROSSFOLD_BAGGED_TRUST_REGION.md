# Crossfold-bagged scene-head trust region

The frozen bagged ensemble is blended around the current v3 head to protect held-fold recall.

- Aggregation: `mean_logit`
- Bagged weight: 0.100
- AP delta vs current: +0.00152
- Recall delta vs current: +0.00000
- Paired-site AP interval vs current: [+0.00101, +0.00201]

| Fold | AP delta vs current | Recall delta vs current |
|---|---:|---:|
| 0 | +0.00085 | +0.00000 |
| 1 | +0.00108 | +0.00000 |
| 2 | +0.00175 | +0.00000 |
| 3 | +0.00266 | +0.00264 |
| 4 | +0.00181 | +0.00000 |

Reject the bagged trust region before paper-cache scoring.
