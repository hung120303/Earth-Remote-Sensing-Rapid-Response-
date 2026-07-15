# Target-adapted XGBoost consensus

Each held fold is unlabeled to its density-ratio estimator; both component learners exclude that fold's labels.

- Logit weights (current / target / XGBoost): 0.65 / 0.30 / 0.05
- AP delta vs current: +0.00543
- Recall delta vs current: +0.00262
- Paired-site AP interval vs current: [+0.00318, +0.00750]

| Fold | AP delta | Recall delta |
|---|---:|---:|
| 0 | +0.00291 | +0.00000 |
| 1 | +0.00443 | +0.00268 |
| 2 | +0.00564 | +0.00502 |
| 3 | +0.00993 | +0.00264 |
| 4 | +0.00495 | -0.00131 |

Reject consensus before paper adaptation.
