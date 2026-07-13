# MARS v3 connected-proposal validation

Stage-two classifier fit/calibration uses only internal-training groups; operating thresholds use disjoint internal validation.

- Proposals: 10,834 total / 10,634 labeled / 200 ambiguous excluded from fitting
- Validation recall / specificity / FPR: 0.820 / 0.951 / 0.049
- Validation AUROC / AP: 0.927 / 0.794
- Selected neural-presence blend weight: 1.00
- Proposal-only recall / FPR: 0.812 / 0.048
- Neural-only recall / FPR: 0.820 / 0.049
- Artifact SHA-256: `c3811c9880a485fc19678540c7a876ea6c7cbe3cba8e87214aa6ea32190a9360`

## Decision

Freeze the calibrated component classifier with the v3 checkpoint before strict evaluation.
