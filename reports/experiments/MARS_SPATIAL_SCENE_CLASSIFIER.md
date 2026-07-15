# Physics-guided spatial MARS scene classifier

Selection used cross-fitted folds 2/3/4; the selected model and blend were frozen before folds 0/1 were scored.

- Selected model: `dropout-0.2_epochs-8_feature_set-physics_spatial_learning_rate-0.0003_weight_decay-0.001_weighting-site_cell`
- Spatial blend weight: 0.100
- Inner AP delta vs primary: +0.03018
- Inner AP delta vs stronger head: +0.00060
- Inner AP interval vs primary: [+0.01997, +0.04190]

| Partition | AP delta vs primary | Recall delta | AP 95% CI | AP delta vs new | Gates |
|---|---:|---:|---:|---:|---|
| fold0 | +0.02564 | +0.00537 | [+0.01049, +0.04537] | +0.00025 | PASS |
| fold1 | +0.03735 | +0.00671 | [+0.01986, +0.05812] | +0.00101 | PASS |

Freeze the spatial classifier for a transparent post-test paper benchmark.
