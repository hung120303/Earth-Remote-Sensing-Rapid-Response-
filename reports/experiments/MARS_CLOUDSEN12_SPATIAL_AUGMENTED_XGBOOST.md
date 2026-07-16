# CloudSEN12+ spatial-negative augmented MARS scene head

- Selected negative multiplier: **1.00**.
- Selected complement blend: **0.200**.
- Cross-fitted folds 2/3/4 gates: **PASS**.

| Partition | AP delta | Recall delta at 7.13% FPR | AP 95% CI | Gates |
|---|---:|---:|---:|---|
| folds 2/3/4 selection | +0.00229 | +0.00517 | [+0.00023, +0.00525] | PASS |
| fold 0 confirmation | +0.00337 | +0.00000 | [-0.00025, +0.00711] | FAIL |
| fold 1 confirmation | +0.00295 | +0.00537 | [+0.00079, +0.00692] | PASS |

Reject the spatial-negative scene complement before paper-cache replay.
