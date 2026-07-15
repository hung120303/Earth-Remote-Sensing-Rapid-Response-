# Prithvi-EO-2.0 MARS scene probe

Frozen foundation features were selected with cross-fitted folds 2/3/4; the chosen probe and blend were then evaluated once on held folds 0 and 1.

- Selected: `dropout-0.0_epochs-20_feature_set-cls_plus_base_hidden-0_learning_rate-0.001_weight_decay-0.01_weighting-uniform`, blend 0.05
- Inner AP delta vs current stronger head: +0.00003

| Partition | AP delta vs primary | Recall delta | AP 95% CI | AP delta vs current head | Gates |
|---|---:|---:|---:|---:|---|
| fold0 | +0.02506 | +0.00537 | [+0.00955, +0.04544] | -0.00033 | PASS |
| fold1 | +0.03765 | +0.00537 | [+0.02019, +0.05879] | +0.00130 | PASS |

Freeze the Prithvi scene complement for one transparent paper benchmark.
