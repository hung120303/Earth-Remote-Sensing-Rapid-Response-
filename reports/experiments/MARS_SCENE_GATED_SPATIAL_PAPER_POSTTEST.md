# Scene-gated spatial successor: exact MARS-S2L paper benchmark

Transparent post-test cache replay; it is not an untouched confirmation cohort. Scene metrics come unchanged from the frozen ordinary-BCE spatial classifier; only the independently development-confirmed dense-mask gate changes.

| View | AP delta (95% CI) | Recall delta (95% CI) | IoU | IoU delta (95% CI) | Gates |
|---|---:|---:|---:|---:|---|
| full | +0.03108 ([+0.01254, +0.04520]) | +0.03034 ([+0.01893, +0.04337]) | 0.37996 | +0.05560 ([+0.03492, +0.07788]) | PASS |
| test_only_sites | +0.01747 ([-0.01431, +0.04764]) | +0.01762 ([-0.01117, +0.04603]) | 0.29246 | +0.12090 ([+0.08591, +0.15621]) | FAIL |

Dense segmentation now passes both exact paper views, but at least one scene-level superiority gate remains unresolved.
