# Prithvi scene probe: exact MARS-S2L paper benchmark

Transparent post-test replay; this is not an untouched confirmation cohort. The dense-mask gate remains driven by the separately frozen v3 scene score.

| View | AP | AP delta (95% CI) | Recall delta (95% CI) | FPR delta | IoU delta (95% CI) | Gates |
|---|---:|---:|---:|---:|---:|---|
| full | 0.67338 | +0.03236 ([+0.01481, +0.04637]) | +0.03089 ([+0.01857, +0.04661]) | -0.03682 | +0.05560 ([+0.03482, +0.07747]) | PASS |
| test_only_sites | 0.46779 | +0.01752 ([-0.01780, +0.05003]) | +0.02203 ([-0.00870, +0.04938]) | -0.04168 | +0.12090 ([+0.08599, +0.15544]) | FAIL |

Reject the Prithvi complement as the final successor; at least one exact paper gate fails.
