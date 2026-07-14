# Development-only mask-threshold analysis

The paper test was not loaded. Scene ranking is unchanged; this analysis only calibrates the dense-mask decision rule.

| Threshold | Pooled IoU | Delta | Worst fold delta | Worst sensor delta |
|---:|---:|---:|---:|---:|
| 0.5000 | 0.52726 | +0.00000 | +0.00000 | +0.00000 |
| 0.5500 | 0.53937 | +0.01211 | +0.01031 | +0.00604 |
| 0.6000 | 0.54964 | +0.02238 | +0.01796 | +0.01030 |
| 0.6500 | 0.55786 | +0.03060 | +0.02310 | +0.01219 |
| 0.7000 | 0.56413 | +0.03687 | +0.02665 | +0.01228 |
| 0.7500 | 0.56737 | +0.04011 | +0.02602 | +0.00861 |
| 0.8000 | 0.56736 | +0.04010 | +0.02183 | +0.00122 |

Selected threshold: **0.7000**.

Advance this mask threshold to a separate development-fold confirmation.
