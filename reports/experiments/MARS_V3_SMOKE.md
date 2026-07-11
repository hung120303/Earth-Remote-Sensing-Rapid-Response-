# ERSRR MARS v3 validation result

Pipeline smoke test only; not an accuracy result.

- Model: `ersrr_mars_full_unet_proposal_v3` / 14,268,915 parameters
- Samples: 64 train / 32 validation
- Best epoch: 1 / 1
- Validation recall at FPR <= 0.05: 0.000 (FPR 0.000)
- Validation AUROC / AP: 0.465 / 0.490
- Validation mask Dice: 0.082
- Checkpoint SHA-256: `83a2464265cb1c07f1d1552c53d400928960243b4e4ebbbfbab54692e75f251c`

## Decision

Pipeline contract passes. Do not interpret smoke metrics; acquire the frozen full fit/validation corpus before architecture selection.
