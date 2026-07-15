# Development-only mask-threshold analysis

The paper test was not loaded. Scene ranking is unchanged; this analysis only calibrates the dense-mask decision rule.

| Threshold | Pooled IoU | Delta | Worst fold delta | Worst sensor delta |
|---:|---:|---:|---:|---:|
| 0.5000 | 0.48872 | +0.00000 | +0.00000 | +0.00000 |
| 0.6500 | 0.52800 | +0.03928 | +0.03834 | +0.01010 |
| 0.7000 | 0.53703 | +0.04832 | +0.04737 | +0.00910 |
| 0.7500 | 0.54366 | +0.05494 | +0.05356 | +0.00520 |
| 0.8000 | 0.54633 | +0.05761 | +0.05676 | -0.00160 |

Selected threshold: **0.8000**.

Do not advance this mask rule; at least one fold or sensor IoU gate failed.
