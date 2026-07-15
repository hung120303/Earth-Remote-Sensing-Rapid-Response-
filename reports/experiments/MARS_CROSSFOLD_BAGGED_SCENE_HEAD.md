# Crossfold-bagged MARS scene head

Every OOF score averages four ExtraTrees members trained on distinct subsets that exclude the held fold.

- Aggregation: `median_probability`
- Primary/head blend: 0.875
- AP delta vs current head: +0.00820
- Recall delta vs current head: +0.00236
- Paired-site AP interval vs current: [+0.00447, +0.01157]

| Fold | AP delta vs current | Recall delta vs current |
|---|---:|---:|
| 0 | +0.00418 | -0.00403 |
| 1 | +0.00615 | +0.00268 |
| 2 | +0.00918 | +0.01380 |
| 3 | +0.01560 | +0.00396 |
| 4 | +0.00656 | -0.00261 |

Reject crossfold bagging before paper-cache scoring.
