# Adaptive Prithvi: exact MARS-S2L paper benchmark

Transparent post-test replay; this is not an untouched confirmation cohort. Model scores came from a separately sealed label-free adaptation.

| View | AP | AP delta (95% CI) | Recall delta (95% CI) | FPR delta | IoU delta (95% CI) | Gates |
|---|---:|---:|---:|---:|---:|---|
| full | 0.67700 | +0.03598 ([+0.01633, +0.05062]) | +0.03420 ([+0.02059, +0.04871]) | -0.03296 | +0.05560 ([+0.03536, +0.07742]) | PASS |
| test_only_sites | 0.46550 | +0.01523 ([-0.02630, +0.05233]) | +0.02643 ([-0.00671, +0.05833]) | -0.03746 | +0.12090 ([+0.08612, +0.15472]) | FAIL |

Reject adaptive Prithvi as final successor; at least one exact paper gate fails.
