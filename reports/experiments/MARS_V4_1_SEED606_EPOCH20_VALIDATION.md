# ERSRR MARS v4.1 simulation-first validation

Development result on spatially isolated internal groups; the opened strict cohort is not used for selection.

- Model: `ersrr_temporal_siamese_simulation_v4` / 13,621,491 parameters
- Samples: 23,763 fit / 5,945 validation
- Simulation library: 257 real enhancement rasters
- Best epoch: 12 / 20
- Validation AP / AUROC: 0.787 / 0.952
- Recall at <=8% FPR: 0.875 (FPR 0.079)
- Five-seed v3 reference AP / AUROC: 0.810 / 0.929
- Five-seed v3 reference recall@5% FPR / pixel Dice: 0.831 / 0.577
- Positive pixel Dice: 0.704

## Decision

Reject this v4 pilot before strict-cohort evaluation because it does not match the five-seed v3 internal reference on every gate. Preserve the negative result and revise the objective; do not spend the opened strict cohort on this checkpoint.
