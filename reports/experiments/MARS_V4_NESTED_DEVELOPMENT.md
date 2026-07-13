# ERSRR v4 nested development experiment

Development result only. The former v3 strict cohort was already opened for v3 evaluation and is now used with group-nested cross-fitting; it is not an untouched v4 test set.

- Scenes/groups: 4,401 / 150
- Plume/no-plume: 67 / 4,334
- Released MARS-S2L: AP 0.352, AUROC 0.818, recall 0.642, FPR 0.095
- Nested v4 ranking: AP 0.183, AUROC 0.699
- Nested v4 at 5% FPR target: recall 0.507, observed FPR 0.052, precision 0.130
- Nested v4 at 9.5% FPR target: recall 0.582, observed FPR 0.128, precision 0.066
- Final development fit: `all:hist_gradient_boosting` with 48 features

## Spatial-fold constraint

The largest indivisible 25 km component contains 25 positive scenes. With five folds, at least one held-out fold must therefore contain at least 25 positives; the observed 25/10/11/11/10 allocation reaches that lower bound.

## Outer-fold architecture selections

| Fold | Candidate selected on inner OOF | Inner AP | Held-out AP | Held-out AUROC | Held-out positives |
|---:|---|---:|---:|---:|---:|
| 1 | `all:hist_gradient_boosting` | 0.557 | 0.080 | 0.638 | 25 |
| 2 | `all:hist_gradient_boosting` | 0.213 | 0.680 | 0.963 | 10 |
| 3 | `released_plus_v3:logistic` | 0.270 | 0.333 | 0.842 | 11 |
| 4 | `all:hist_gradient_boosting` | 0.258 | 0.155 | 0.602 | 11 |
| 5 | `all:hist_gradient_boosting` | 0.249 | 0.495 | 0.854 | 10 |

## Interpretation boundary

Reject the score/physics cascade as the v4 architecture because it did not exceed released MARS-S2L on every development gate. Do not carry its fitted artifact into confirmatory evaluation. Redirect v4 toward simulation-trained segmentation, explicit hard negatives, and domain-aware sampling; keep this result as a documented negative experiment.

The final serialized verifier and feature cache remain ignored because they are rejected derived artifacts. The numerical results were not recomputed for this interpretation-only amendment; they retain clean provenance from commit `42775d8c`, while rejection logic and fold-concentration documentation are fixed in commit `3d41ae51`.
