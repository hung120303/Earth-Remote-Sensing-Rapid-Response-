# MARS sensor-aware ordinal folds 3/4

- Promotion gates pass: **False**
- Pooled AP delta: **-0.786371**
- AP bootstrap lower bound: **-0.835886**
- Matched-FPR recall delta: **-0.776772**
- Dense IoU delta: **-0.636630**
- Dense bootstrap lower bound: **-0.682844**

## Endpoint checkpoints

- Held fold 3: epoch 11, inner-training cutpoints [95.39521026611328, 331.55938720703125, 709.936279296875]
- Held fold 4: epoch 23, inner-training cutpoints [75.24365425109863, 307.33689880371094, 642.2248992919922]

## Gate checks

- `pooled_ap_delta_gte_0_003`: **False**
- `each_fold_ap_positive`: **False**
- `each_sensor_ap_positive`: **False**
- `matched_fpr_recall_nonnegative`: **False**
- `ap_bootstrap_lower_positive`: **False**
- `dense_iou_delta_positive`: **False**
- `dense_bootstrap_lower_positive`: **False**

Enhancement values are used only as producer-supplied ordinal ordering; no physical-unit claim is made.
