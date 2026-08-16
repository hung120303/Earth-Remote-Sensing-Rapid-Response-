# MARS released-detector recall-anchor diagnostic

Development-only folds 3/4 diagnostic; no candidate was fitted or promoted.

| Score | AP | Recall at <=7.13% FPR | Realized FPR |
|---|---:|---:|---:|
| current_v3 | 0.875284 | 0.952100 | 0.071266 |
| spatial_prithvi | 0.904076 | 0.961942 | 0.071266 |
| gaussian_dofa_champion | 0.906525 | 0.961942 | 0.071266 |
| released_detector | 0.871465 | 0.946194 | 0.071266 |

## Matched-FPR decision complementarity

`first` is the Gaussian+DOFA champion; `second` is the released detector.

| Cell | Rows | Positives | Negatives | Precision |
|---|---:|---:|---:|---:|
| both | 2369 | 1433 | 936 | 0.604897 |
| first_only | 253 | 33 | 220 | 0.130435 |
| second_only | 229 | 9 | 220 | 0.039301 |
| neither | 14894 | 49 | 14845 | 0.003290 |

Decision: **continue_to_recall_anchored_architecture**

## Strongest released-rescue TP/FP feature contrasts

| Feature | Standardized mean difference |
|---|---:|
| primary_area_above_0.9 | +1.3838 |
| input_0_top_100_mean | -1.2014 |
| primary_connected_score | +0.8507 |
| primary_area_above_0.5 | +0.8506 |
| primary_top_200_mean | +0.8410 |
| logit_delta_valid_mean | +0.8187 |
| primary_top_500_mean | +0.7876 |
| primary_top_100_mean | +0.7548 |
| primary_top_50_mean | +0.7122 |
| primary_top_25_mean | +0.7082 |
