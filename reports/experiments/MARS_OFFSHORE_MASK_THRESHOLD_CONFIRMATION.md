# Offshore MARS mask-threshold reverse validation

Baseline: Sentinel-2 0.80 / Landsat 0.70. Candidate: offshore scenes use 0.90. The paper test was not loaded.

| Partition | Rows | Offshore positive | Baseline IoU | Candidate IoU | Delta | 95% CI | Gates |
|---|---:|---:|---:|---:|---:|---:|---|
| selection_folds_2_3_4 | 26,578 | 102 | 0.57228 | 0.56996 | -0.00232 | [-0.00728, +0.00039] | FAIL |
| confirmation_folds_0_1 | 17,785 | 0 | 0.55135 | 0.55152 | +0.00017 | [+0.00000, +0.00064] | PASS |
| all_five_folds | 44,363 | 102 | 0.56404 | 0.56268 | -0.00136 | [-0.00404, +0.00030] | FAIL |

Reject the offshore 0.90 mask threshold because development reverse validation failed.
