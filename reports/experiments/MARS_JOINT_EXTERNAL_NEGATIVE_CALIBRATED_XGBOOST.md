# Joint external positive/negative augmented MARS scene head

- Positive multiplier: **2.7185**.
- Negative multiplier: **4.0000**.
- Complement blend: **0.200**.

| Partition | AP delta | Recall delta at 7.13% FPR | AP 95% CI | Gates |
|---|---:|---:|---:|---|
| folds 2/3/4 selection | +0.00363 | +0.00345 | [+0.00095, +0.00674] | PASS |
| fold 0 confirmation | +0.00374 | +0.00134 | [+0.00021, +0.00739] | PASS |
| fold 1 confirmation | +0.00382 | +0.00403 | [+0.00109, +0.00862] | PASS |

Reject the joint external-data scene head before paper-cache replay.
