# XGBoost successor: exact MARS-S2L paper benchmark

Transparent post-test replay; this is not an untouched confirmation cohort. XGBoost scores were computed from a separate label-free cache, and the dense-mask gate remains driven by the unchanged v3 scene score.

| View | AP | AP delta (95% CI) | Recall delta (95% CI) | FPR delta | IoU delta (95% CI) | Gates |
|---|---:|---:|---:|---:|---:|---|
| full | 0.67599 | +0.03497 ([+0.01565, +0.04933]) | +0.03530 ([+0.02087, +0.04865]) | -0.03754 | +0.05560 ([+0.03499, +0.07755]) | PASS |
| test_only_sites | 0.46400 | +0.01373 ([-0.02262, +0.04562]) | +0.02203 ([-0.00913, +0.04965]) | -0.04233 | +0.12090 ([+0.08584, +0.15621]) | FAIL |

Reject the XGBoost complement as the final successor; at least one exact paper gate fails.
