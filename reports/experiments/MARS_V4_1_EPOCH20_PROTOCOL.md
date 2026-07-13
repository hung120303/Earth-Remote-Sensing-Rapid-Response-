# ERSRR MARS v4.1 capped schedule-extension protocol

Status: predeclared before the 20-epoch run.

## Rationale

The sealed 10-epoch seed-606 pilot selected epoch 10, and internal-validation AP increased at every evaluation: 0.473, 0.597, 0.712, 0.766, and 0.781. Its learning rate reached zero at the selected endpoint. The checkpoint exceeded the five-seed v3 mean on AUROC and positive-pixel Dice, while narrowly missing AP and recall at 5% FPR. This supports one controlled test of under-training without changing the architecture, data, loss, score, or seed.

## Frozen run

- Git parent: `17d21d61` plus this protocol commit; the training implementation remains byte-identical to commit `6bf9097d`.
- Architecture/objective: v4.1 temporal-siamese segmentation, observable weighted BCE (positive weight 20) plus 0.5 soft Dice.
- Seed: 606.
- Training draws: 4,096 per epoch for at most 20 epochs.
- Batch/workers: 12 / 4.
- Optimizer schedule: existing AdamW and 20-epoch cosine annealing implementation.
- Validation cadence: every 2 epochs; patience 4 validation checks.
- Model selection: lexicographic validation AP, recall at 8% FPR, then positive-pixel Dice, as already implemented.
- Strict spatial cohort: must not be loaded during this run.

## Decision rule

Promote to one frozen strict-cohort evaluation only if the selected checkpoint is not below the five-seed v3 internal mean on all four existing gates: AP 0.810261, AUROC 0.928784, recall 0.831287 at no more than 5% FPR, and positive-pixel Dice 0.576951. Otherwise reject v4.1 after this cap; do not add another schedule extension based on the same validation cohort.

This is an adaptive development experiment prompted by the sealed 10-epoch trajectory, not an independent confirmatory result.
