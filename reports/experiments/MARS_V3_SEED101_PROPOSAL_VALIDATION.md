# MARS v3 connected-proposal validation

Stage-two classifier fit/calibration uses only internal-training groups; operating thresholds use disjoint internal validation.

- Proposals: 13,419 total / 13,264 labeled / 155 ambiguous excluded from fitting
- Validation recall / specificity / FPR: 0.834 / 0.951 / 0.049
- Validation AUROC / AP: 0.931 / 0.822
- Selected neural-presence blend weight: 0.50
- Proposal-only recall / FPR: 0.826 / 0.044
- Neural-only recall / FPR: 0.832 / 0.050
- Artifact SHA-256: `3173f519780139e65be1d28f4cd11dafd7056468a4494f97eef769a8095dfb6c`

## Decision

Freeze the calibrated component classifier with the v3 checkpoint before strict evaluation.
