# MARS v3 connected-proposal validation

Stage-two classifier fit/calibration uses only internal-training groups; operating thresholds use disjoint internal validation.

- Proposals: 13,760 total / 13,598 labeled / 162 ambiguous excluded from fitting
- Validation recall / specificity / FPR: 0.808 / 0.953 / 0.047
- Validation AUROC / AP: 0.902 / 0.587
- Artifact SHA-256: `478d9c5b0f654fe00681a02e79a401cfc6d421f446b18a8b347a7f28476d06ff`

## Decision

Freeze the calibrated component classifier with the v3 checkpoint before strict evaluation.
