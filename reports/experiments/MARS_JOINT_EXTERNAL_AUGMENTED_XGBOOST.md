# Joint external positive/negative augmented MARS scene head

- Positive multiplier: **2.7185**.
- Negative multiplier: **1.0000**.
- Complement blend: **0.200**.

| Partition | AP delta | Recall delta at 7.13% FPR | AP 95% CI | Gates |
|---|---:|---:|---:|---|
| folds 2/3/4 selection | +0.00345 | +0.00302 | [+0.00094, +0.00628] | PASS |
| fold 0 confirmation | +0.00396 | +0.00134 | [+0.00055, +0.00788] | PASS |
| fold 1 confirmation | +0.00329 | +0.00671 | [+0.00074, +0.00787] | PASS |

Reject the joint external-data scene head before paper-cache replay.
