# ERSRR MARS v3 validation result

Validation-selected result on the frozen internal groups; strict test remains untouched.

- Model: `ersrr_mars_full_unet_proposal_v3` / 14,268,915 parameters
- Samples: 23763 train / 5945 validation
- Best epoch: 22 / 29
- Validation recall at FPR <= 0.05: 0.848 (FPR 0.049)
- Validation AUROC / AP: 0.940 / 0.828
- Validation mask Dice: 0.577
- Checkpoint SHA-256: `61b8ae482c85dc244aa151e4d91c797e9095c69dbb15f51b1afdb08f936ed8dc`

## Decision

Freeze this validation-selected checkpoint before any strict-spatial evaluation.
