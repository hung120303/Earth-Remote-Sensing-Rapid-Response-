# MARS v3 connected-proposal validation

Stage-two classifier fit/calibration uses only internal-training groups; operating thresholds use disjoint internal validation.

- Proposals: 13,530 total / 13,364 labeled / 166 ambiguous excluded from fitting
- Validation recall / specificity / FPR: 0.832 / 0.951 / 0.049
- Validation AUROC / AP: 0.923 / 0.796
- Selected neural-presence blend weight: 1.00
- Proposal-only recall / FPR: 0.824 / 0.048
- Neural-only recall / FPR: 0.832 / 0.049
- Artifact SHA-256: `317d79ebe9352e12f9590b42d5e60567cd8827ebedbc67fddc6fd96e65a8e28d`

## Decision

Freeze the calibrated component classifier with the v3 checkpoint before strict evaluation.
