# ERSRR MARS v3 validation result

Validation-selected result on the frozen internal groups; strict test remains untouched.

- Model: `ersrr_mars_full_unet_proposal_v3` / 14,268,915 parameters
- Samples: 23763 train / 5945 validation
- Best epoch: 27 / 34
- Validation recall at FPR <= 0.05: 0.820 (FPR 0.049)
- Validation AUROC / AP: 0.927 / 0.794
- Validation mask Dice: 0.549
- Checkpoint SHA-256: `fd100e622f49798764ef8497966d0b2741cd20196969c101d1a6e794deec8bd8`

## Decision

Freeze this validation-selected checkpoint before any strict-spatial evaluation.
