# ERSRR MARS v4.2 simulation-first validation

Pipeline smoke test only; not an accuracy result.

- Model: `ersrr_temporal_siamese_simulation_v4` / 13,621,491 parameters
- Samples: 32 fit / 16 validation
- Simulation library: 256 real enhancement rasters
- Best epoch: 1 / 1
- Validation AP / AUROC: 0.491 / 0.438
- Recall at <=8% FPR: 0.000 (FPR 0.000)
- Positive pixel Dice: 0.002

## Decision

Pipeline contract passes. Do not interpret smoke metrics.
