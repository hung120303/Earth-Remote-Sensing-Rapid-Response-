# ERSRR grouped baseline results

Generated: `2026-07-10T00:18:09.648882+00:00`

All metrics are 5-fold group-held-out results with thresholds calibrated on disjoint inner groups. Legacy imagery and targets are evaluated on the same 128-pixel (~20 m) grid as the U-Net. Groups are connected components sharing an MGRS tile, Sentinel-2 acquisition, or EMIT granule. AUPRC and AUROC are threshold-free; F1/IoU are also reported as macro per-scene means.

## legacy_quality_cohort

### Target > 300 ppm·m

| Model | Features | Pixel AUPRC | Pixel AUROC | Scene F1 | Scene IoU | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| prior_dummy | 5 | 0.3548 | 0.5000 | 0.4988 | 0.3548 | 2.61 |
| raw_logistic | 5 | 0.4481 | 0.5995 | 0.4781 | 0.3398 | 5.59 |
| physics_logistic | 11 | 0.4350 | 0.5805 | 0.4606 | 0.3268 | 8.65 |
| physics_histgb | 11 | 0.3621 | 0.5004 | 0.4938 | 0.3507 | 20.32 |

### Target > 1000 ppm·m

| Model | Features | Pixel AUPRC | Pixel AUROC | Scene F1 | Scene IoU | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| prior_dummy | 5 | 0.0601 | 0.5000 | 0.2234 | 0.2133 | 2.34 |
| raw_logistic | 5 | 0.1243 | 0.6384 | 0.1614 | 0.1266 | 5.81 |
| physics_logistic | 11 | 0.1109 | 0.5833 | 0.1397 | 0.1077 | 8.64 |
| physics_histgb | 11 | 0.0591 | 0.4605 | 0.1752 | 0.1652 | 21.14 |

## v002_physical_mask_holdout

| Model | Features | Pixel AUPRC | Pixel AUROC | Scene F1 | Scene IoU | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| prior_dummy | 5 | 0.1708 | 0.5000 | 0.2819 | 0.1708 | 0.33 |
| raw_single_logistic | 5 | 0.2410 | 0.6367 | 0.2294 | 0.1406 | 0.48 |
| physics_single_logistic | 11 | 0.2415 | 0.6397 | 0.2292 | 0.1404 | 0.44 |
| physics_single_histgb | 11 | 0.1941 | 0.5861 | 0.2939 | 0.1777 | 15.51 |
| bitemporal_logistic | 33 | 0.2271 | 0.6212 | 0.1917 | 0.1152 | 0.72 |
| bitemporal_histgb | 33 | 0.2070 | 0.5995 | 0.2673 | 0.1635 | 45.93 |
