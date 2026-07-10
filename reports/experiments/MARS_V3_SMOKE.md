# ERSRR MARS v3 validation result

Pipeline smoke test only; not an accuracy result.

- Model: `ersrr_mars_full_unet_proposal_v3` / 14,381,987 parameters
- Samples: 64 train / 32 validation
- Best epoch: 1 / 1
- Validation recall at FPR <= 0.05: 0.062 (FPR 0.000)
- Validation AUROC / AP: 0.551 / 0.564
- Validation mask Dice: 0.085
- Checkpoint SHA-256: `4d1de5aff29b1e2f1fbfdae11112fdbeb715fcb4533820a6c390ee49d1edc86d`

## Decision

Pipeline contract passes. Do not interpret smoke metrics; acquire the frozen full fit/validation corpus before architecture selection.
