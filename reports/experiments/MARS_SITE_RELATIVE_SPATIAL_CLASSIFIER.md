# Site-relative spatial MARS scene classifier

Every spatial template is label-free and computed only from other observations of the same physical site.

- Selected model: `dropout-0.2_epochs-8_feature_set-original_residual_learning_rate-0.0003_weight_decay-0.001_weighting-group`
- Blend weight: 0.100
- Inner AP delta vs current head: +0.00266
- Inner AP interval vs current head: [+0.00062, +0.00511]

| Partition | AP delta vs primary | AP delta vs current | Recall delta vs current | Gates |
|---|---:|---:|---:|---|
| fold0 | +0.02669 | +0.00130 | +0.00268 | PASS |
| fold1 | +0.03791 | +0.00156 | +0.00403 | PASS |

Pooled folds-0/1 paired-site AP interval versus current head: [+0.00091, +0.00210].

Freeze the site-relative spatial model for one transparent exact-paper replay.
