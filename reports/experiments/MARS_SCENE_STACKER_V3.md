# Cross-fitted MARS scene stacker v3

The stacker was selected only from OOF folds 2/3/4, then frozen before folds 0/1 were scored.

- Selected model: `C-1.0_family-logistic_feature_set-scores_weighting-group`
- Inner AP delta vs primary: +0.02121
- Inner AP delta vs stronger head: -0.00836
- Inner AP interval vs stronger head: [-0.01252, -0.00410]

| Partition | AP delta vs primary | Recall delta | AP 95% CI | AP delta vs new | Gates |
|---|---:|---:|---:|---:|---|
| fold0 | +0.02550 | +0.00268 | [+0.00984, +0.04607] | +0.00012 | PASS |
| fold1 | +0.03802 | +0.00940 | [+0.02006, +0.05899] | +0.00167 | PASS |

Reject the stacker before any paper-test evaluation.
