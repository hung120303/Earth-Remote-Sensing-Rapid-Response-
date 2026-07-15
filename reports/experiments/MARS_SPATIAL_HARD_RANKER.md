# Hard-negative pairwise MARS spatial ranker

Every selected score was cross-fitted on folds 2/3/4, then frozen before folds 0/1 confirmation.

- Selected model: `dropout-0.2_epochs-10_feature_set-physics_spatial_hard_negative_fraction-1.0_learning_rate-0.0003_pairwise_weight-0.5_weight_decay-0.001_weighting-site_cell`
- Spatial blend weight: 0.100
- Inner AP delta vs primary: +0.02984
- Inner AP delta vs stronger head: +0.00027
- Inner AP interval vs stronger head: [-0.00137, +0.00174]

| Partition | AP delta vs primary | Recall delta | AP 95% CI | AP delta vs new | Gates |
|---|---:|---:|---:|---:|---|
| fold0 | +0.02574 | +0.00537 | [+0.01051, +0.04590] | +0.00036 | PASS |
| fold1 | +0.03748 | +0.00805 | [+0.02079, +0.05824] | +0.00114 | PASS |

Freeze the hard ranker for a transparent post-test paper benchmark.
