# ERSRR MARS v3 validation result

Validation-selected result on the frozen internal groups; strict test remains untouched.

- Model: `ersrr_mars_full_unet_proposal_v3` / 14,268,915 parameters
- Samples: 23763 train / 5945 validation
- Best epoch: 25 / 32
- Validation recall at FPR <= 0.05: 0.832 (FPR 0.049)
- Validation AUROC / AP: 0.923 / 0.796
- Validation mask Dice: 0.592
- Checkpoint SHA-256: `09c4614d2e7d6374aaf060cdb4aded744b4f86d9d69490029be0aa23a73fdc26`

## Decision

Freeze this validation-selected checkpoint before any strict-spatial evaluation.
