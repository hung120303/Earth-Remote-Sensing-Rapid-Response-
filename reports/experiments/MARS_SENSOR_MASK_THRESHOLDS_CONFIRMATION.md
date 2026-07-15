# Sensor-specific MARS mask-threshold confirmation

Frozen rule: Sentinel-2 uses 0.80; Landsat retains 0.70. The paper test was not loaded.

| Partition | Rows | Baseline IoU | Routed IoU | Delta | 95% CI | Gates |
|---|---:|---:|---:|---:|---:|---|
| selection_folds_2_3_4 | 26,578 | 0.56413 | 0.57228 | +0.00815 | [+0.00070, +0.01857] | PASS |
| confirmation_folds_0_1 | 17,785 | 0.53703 | 0.55135 | +0.01431 | [+0.00691, +0.02344] | PASS |
| all_five_folds | 44,363 | 0.55338 | 0.56404 | +0.01066 | [+0.00479, +0.01754] | PASS |

Advance the frozen sensor-specific mask rule to a transparent post-test benchmark evaluation.
