# MARS residual fold-0 trust region

Development-only architecture selection; the paper test and fold 1 were not loaded.

| Alpha | AP delta | Recall delta at <=7.13% FPR | IoU delta | Worst primary delta |
|---:|---:|---:|---:|---:|
| 0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| 0.03125 | +0.00020 | +0.00000 | +0.00063 | +0.00020 |
| 0.06250 | +0.00037 | +0.00000 | +0.00130 | +0.00037 |
| 0.12500 | +0.00035 | +0.00000 | +0.00255 | +0.00035 |
| 0.25000 | +0.00092 | -0.00134 | +0.00518 | +0.00092 |
| 0.37500 | +0.00102 | +0.00000 | +0.00760 | +0.00102 |
| 0.50000 | +0.00246 | +0.00000 | +0.01034 | +0.00246 |
| 0.75000 | +0.00186 | +0.00000 | +0.01283 | +0.00186 |
| 1.00000 | -0.00046 | -0.00134 | -0.00979 | -0.00979 |

Selected alpha: **0.50000**.

Reject the trust-region blend on fold 0; proceed to source-aligned fitting.
