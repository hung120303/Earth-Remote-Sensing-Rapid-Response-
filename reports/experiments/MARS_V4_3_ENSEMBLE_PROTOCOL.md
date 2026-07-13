# ERSRR v4.3 frozen ensemble protocol

Status: predeclared before ensemble inference or metric calculation.

## Hypothesis

The three v4.2 seeds reproducibly improve AUROC and pixel Dice over v3, but their AP and recall at a 5% false-positive-rate target vary enough to fail the three-seed mean gate. A fixed ensemble should reduce seed-specific ranking variance without changing the learned segmentation architecture or consulting the strict cohort.

## Frozen inputs and rule

- Inputs: the selected v4.2 checkpoints for seeds 606, 707, and 808, checksum-bound by their frozen validation reports.
- Per-seed scene score: sigmoid of the mean of the top 2% observable segmentation logits.
- Calibration: convert each seed score to an empirical percentile using only the applicable internal-validation fitting partition.
- Ensemble scene score: arithmetic mean of the three calibrated percentiles. No learned combiner, morphology selector, seed weighting, or additional candidate will be evaluated.
- Segmentation: arithmetic mean of the three per-pixel plume probabilities.
- Pixel threshold: choose from 0.1 through 0.9 on all internal validation, maximizing positive-pixel Dice with the lower threshold as the deterministic tie-break.
- Group-held estimate: five fixed 25 km outer folds. In each fold, empirical score distributions and scene thresholds are fitted only on the four training folds and applied to the held-out fold.
- Final development rule: after the group-held audit, fit the three empirical calibrators and operating thresholds on all internal-validation scenes for a possible frozen strict comparison.

## Promotion rule

Authorize one evaluation on the already-opened strict MARS cohort only if both the five-fold group-held estimate and the final all-validation rule are not below the five-seed v3 internal mean on AP 0.810261, AUROC 0.928784, recall 0.831287 at no more than 5% FPR, and the ensemble positive-pixel Dice is not below 0.576951. The aggregated group-held FPR must also be at most 5%.

If any gate fails, preserve the ensemble result and do not load the strict cohort. Even if promoted, the strict comparison is development evidence because earlier v3 strict results informed v4 research. A publication-grade superiority claim still requires a newly sealed same-sensor plume/no-plume cohort.
