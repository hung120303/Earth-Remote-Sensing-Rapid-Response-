# ERSRR v4.2 three-seed development protocol

Status: predeclared before training any v4.2 seed.

## Rationale

The frozen seed-606 v4.1 checkpoint learned substantially better plume masks than v3, but its original top-0.5%-plus-max scene score missed the AP and 5%-FPR recall gates. A predeclared 14-formula, five-fold 25 km group audit found that broader top-logit extent improved ranking. The group-held adaptive selector achieved AP 0.862, AUROC 0.964, FPR 0.0491, and recall 0.8277; it missed the v3 recall reference by two scenes and was therefore rejected under its original gate. On all internal-validation groups, the fixed top-2% mean was the best declared formula and passed all four v3 point-estimate gates. This campaign tests whether that fixed architectural choice is reproducible across training seeds rather than weakening the failed group-held gate.

## Frozen architecture and runs

- Architecture: v4.2 temporal-Siamese simulation-trained segmentation model.
- Scene score: `sigmoid(mean(top 2% observable segmentation logits))` (the sigmoid is monotonic and does not alter ranking).
- Objective: observable weighted BCE (positive weight 20) plus 0.5 soft Dice; no scene-loss gradient.
- Seeds: 606, 707, and 808.
- Schedule per seed: at most 20 epochs, 4,096 balanced draws per epoch, batch 12, validation every 2 epochs, and patience 4 validation checks.
- Simulation: 50% of requested positive draws from fit-only real CH4 fields, equivalent to roughly 25% of balanced draws.
- Checkpoint selection: lexicographic internal-validation AP, recall at 8% FPR, then positive-pixel Dice.
- Strict spatial cohort: must not be loaded during training or internal selection.

## Promotion rule

Promote v4.2 to one frozen comparison on the already-opened strict MARS cohort only if the three-seed internal-validation mean is not below the five-seed v3 mean on all four existing metrics: AP 0.810261, AUROC 0.928784, recall 0.831287 at no more than 5% FPR, and positive-pixel Dice 0.576951. Each seed's 5%-target operating threshold must be selected on internal validation and frozen before strict scoring.

The later strict comparison is a development benchmark because v3 results from that cohort informed v4 research. It cannot serve as a new untouched superiority test for a paper. A publishable superiority claim still requires a newly sealed same-sensor plume/no-plume cohort.
