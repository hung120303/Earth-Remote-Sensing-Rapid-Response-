# UNEP-positive augmented MARS XGBoost scene head

- Selected auxiliary multiplier: **1.0**.
- Selected candidate blend: **0.200**.
- Selection gates: **PASS**.

| Partition | AP delta vs current | Recall delta at 7.13% FPR | AP 95% CI | Gates |
|---|---:|---:|---:|---|
| fold 2 selection | +0.00412 | +0.00251 | not used | PASS |
| fold 0 confirmation | +0.00392 | +0.00134 | [+0.00052, +0.00808] | PASS |
| fold 1 confirmation | +0.00325 | +0.00537 | [+0.00067, +0.00754] | PASS |

Freeze the augmented scene head for exact paper-cache evaluation; dense masks remain unchanged.
