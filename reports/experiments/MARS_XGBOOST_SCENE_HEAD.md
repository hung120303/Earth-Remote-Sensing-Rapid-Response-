# Regularized XGBoost MARS scene head

Every prediction is cross-fitted by complete physical-site fold; no held-label early stopping is used.

- Model specification: `depth3_lr004`
- Current/XGBoost logit blend: 0.100
- AP delta vs current: +0.00245
- Recall delta vs current: +0.00079
- Paired-site AP interval vs current: [+0.00151, +0.00348]

| Fold | AP delta vs current | Recall delta vs current |
|---|---:|---:|
| 0 | +0.00208 | +0.00000 |
| 1 | +0.00221 | +0.00134 |
| 2 | +0.00242 | +0.00125 |
| 3 | +0.00289 | +0.00132 |
| 4 | +0.00216 | +0.00000 |

Freeze the XGBoost head and its artifact for one exact-paper replay.
