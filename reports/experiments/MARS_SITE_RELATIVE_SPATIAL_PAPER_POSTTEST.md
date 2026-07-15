# Site-relative spatial successor: exact MARS-S2L paper benchmark

Transparent post-test replay; this is not an untouched confirmation cohort. Site templates use no labels, and the dense-mask gate remains driven by the unchanged v3 scene score.

| View | AP | AP delta (95% CI) | Recall delta (95% CI) | FPR delta | IoU delta (95% CI) | Gates |
|---|---:|---:|---:|---:|---:|---|
| full | 0.67511 | +0.03409 ([+0.01576, +0.04860]) | +0.03034 ([+0.01843, +0.04556]) | -0.03807 | +0.05560 ([+0.03502, +0.07783]) | PASS |
| test_only_sites | 0.46712 | +0.01685 ([-0.01769, +0.04794]) | +0.01762 ([-0.01105, +0.04545]) | -0.04246 | +0.12090 ([+0.08601, +0.15526]) | FAIL |

Reject the site-relative spatial model as the final successor; at least one exact paper gate fails.
