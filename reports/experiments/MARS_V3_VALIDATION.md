# ERSRR MARS v3 validation result

Validation-selected result on the frozen internal groups; strict test remains untouched.

- Model: `ersrr_mars_full_unet_proposal_v3` / 14,268,915 parameters
- Samples: 23763 train / 5945 validation
- Best epoch: 23 / 30
- Validation recall at FPR <= 0.05: 0.826 (FPR 0.049)
- Validation AUROC / AP: 0.924 / 0.812
- Validation mask Dice: 0.559
- Checkpoint SHA-256: `86c1bc2e55b73de21fa4e8831b009f50a1d75ea3da16f03cfc152c64fff53426`

## Decision

Freeze this validation-selected checkpoint before any strict-spatial evaluation.
