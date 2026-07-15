# Scene domain-routing reverse validation

The rule was diagnosed after opening the paper test, then evaluated unchanged on development folds.

| Partition | AP delta vs primary | Recall delta | AP 95% CI | Offshore positive | Gates |
|---|---:|---:|---:|---:|---|
| cross_fitted_folds_2_3_4 | +0.00895 | -0.00215 | [-0.01942, +0.02685] | 102 | FAIL |
| held_fold_0 | +0.02100 | +0.00268 | [+0.00799, +0.03679] | 0 | PASS |
| held_fold_1 | +0.02907 | +0.01208 | [+0.01453, +0.04544] | 0 | FAIL |

Reject the post-test scene routing rule because development reverse validation failed.
