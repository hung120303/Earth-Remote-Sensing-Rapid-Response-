# ERSRR MARS v3 validation result

Pipeline smoke test only; not an accuracy result.

- Model: `ersrr_mars_full_unet_proposal_v3` / 14,268,915 parameters
- Samples: 64 train / 32 validation
- Best epoch: 1 / 1
- Validation recall at FPR <= 0.05: 0.062 (FPR 0.000)
- Validation AUROC / AP: 0.268 / 0.454
- Validation mask Dice: 0.053
- Checkpoint SHA-256: `26bb3688db0403a13cd0269035c1d91f08115706f70e726564f7335fc185680b`

## Decision

Pipeline contract passes. Do not interpret smoke metrics; acquire the frozen full fit/validation corpus before architecture selection.
