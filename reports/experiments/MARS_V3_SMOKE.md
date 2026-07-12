# ERSRR MARS v3 validation result

Pipeline smoke test only; not an accuracy result.

- Model: `ersrr_mars_full_unet_proposal_v3` / 14,268,915 parameters
- Samples: 64 train / 32 validation
- Best epoch: 1 / 1
- Validation recall at FPR <= 0.05: 0.000 (FPR 0.000)
- Validation AUROC / AP: 0.477 / 0.494
- Validation mask Dice: 0.091
- Checkpoint SHA-256: `3002dea0d445928a154b1b398811992e60e67fb9b2c2db6a4243fb64376187d8`

## Decision

Pipeline contract passes. Do not interpret smoke metrics; acquire the frozen full fit/validation corpus before architecture selection.
