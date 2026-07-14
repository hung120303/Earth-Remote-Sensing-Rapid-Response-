# Development-only mask-threshold analysis

The paper test was not loaded. Scene ranking is unchanged; this analysis only calibrates the dense-mask decision rule.

| Threshold | Pooled IoU | Delta | Worst fold delta | Worst sensor delta |
|---:|---:|---:|---:|---:|
| 0.5000 | 0.48872 | +0.00000 | +0.00000 | +0.00000 |
| 0.7000 | 0.53703 | +0.04832 | +0.04737 | +0.00910 |

Selected threshold: **0.7000**.

Mask threshold 0.70 passed the separate folds 0-1 development confirmation.
