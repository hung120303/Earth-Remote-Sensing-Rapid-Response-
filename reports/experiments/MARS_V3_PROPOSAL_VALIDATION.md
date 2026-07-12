# MARS v3 connected-proposal validation

Stage-two classifier fit/calibration uses only internal-training groups; operating thresholds use disjoint internal validation.

- Proposals: 13,760 total / 13,598 labeled / 162 ambiguous excluded from fitting
- Validation recall / specificity / FPR: 0.826 / 0.951 / 0.049
- Validation AUROC / AP: 0.924 / 0.812
- Selected neural-presence blend weight: 1.00
- Proposal-only recall / FPR: 0.808 / 0.047
- Neural-only recall / FPR: 0.826 / 0.049
- Artifact SHA-256: `aa15e4294c9d1931de510778761b61098966069ab5c5d48e1721342b1b882caa`

## Decision

Freeze the calibrated component classifier with the v3 checkpoint before strict evaluation.
