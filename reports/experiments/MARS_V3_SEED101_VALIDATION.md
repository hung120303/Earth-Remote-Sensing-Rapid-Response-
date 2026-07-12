# ERSRR MARS v3 validation result

Validation-selected result on the frozen internal groups; strict test remains untouched.

- Model: `ersrr_mars_full_unet_proposal_v3` / 14,268,915 parameters
- Samples: 23763 train / 5945 validation
- Best epoch: 14 / 21
- Validation recall at FPR <= 0.05: 0.832 (FPR 0.050)
- Validation AUROC / AP: 0.931 / 0.821
- Validation mask Dice: 0.609
- Checkpoint SHA-256: `688ea56ecabc5f4020c94a45ab8feef03e5fa556ddeed07f244e53eb33437803`

## Decision

Freeze this validation-selected checkpoint before any strict-spatial evaluation.
