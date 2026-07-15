# Scene stacker v3: post-test paper-cache diagnosis

Exploratory only. The stacker failed its inner non-regression gate before this paper cache was loaded.

| View | AP delta | AP 95% CI | Recall delta at matched FPR | Recall 95% CI |
|---|---:|---:|---:|---:|
| full | +0.03637 | [+0.01780, +0.05077] | +0.03696 | [+0.02429, +0.05428] |
| test_only_sites | +0.01763 | [-0.01744, +0.04845] | +0.03084 | [-0.00538, +0.06542] |

Reject the stacker; it does not solve the exact paper scene gates.
