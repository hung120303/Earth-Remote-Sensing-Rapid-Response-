# MARS v3 connected-proposal validation

Stage-two classifier fit/calibration uses only internal-training groups; operating thresholds use disjoint internal validation.

- Proposals: 13,017 total / 12,833 labeled / 184 ambiguous excluded from fitting
- Validation recall / specificity / FPR: 0.848 / 0.951 / 0.049
- Validation AUROC / AP: 0.940 / 0.828
- Selected neural-presence blend weight: 1.00
- Proposal-only recall / FPR: 0.776 / 0.030
- Neural-only recall / FPR: 0.848 / 0.049
- Artifact SHA-256: `f6c33a0e407e5b7e073bbecd69d7af2abdc1d12b2731c41a952d82de3d89eb18`

## Decision

Freeze the calibrated component classifier with the v3 checkpoint before strict evaluation.
