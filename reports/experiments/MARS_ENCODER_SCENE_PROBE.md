# Frozen-encoder MARS scene probe

Selection used cross-fitted folds 2/3/4; the selected probe was frozen before folds 0/1 were scored.

- Selected probe: `dropout-0.2_epochs-20_feature_set-level5_plus_base_hidden-64_learning_rate-0.001_weight_decay-0.001_weighting-site_cell`
- Inner AP delta vs primary: -0.03136
- Inner AP delta vs stronger head: -0.06093
- Inner paired AP interval vs primary: [-0.06477, -0.00646]

| Partition | AP delta vs primary | Recall delta | AP 95% CI | AP delta vs new | Gates |
|---|---:|---:|---:|---:|---|
| fold0 | -0.07970 | -0.06040 | [-0.15232, -0.03606] | -0.10508 | FAIL |
| fold1 | -0.08566 | -0.07114 | [-0.18987, -0.02554] | -0.12200 | FAIL |

Reject the encoder scene probe before paper-test feature extraction.
