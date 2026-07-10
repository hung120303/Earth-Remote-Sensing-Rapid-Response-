# CH4Net released CH4Net_20250329 on the frozen ERSRR strict cohort

Inference-only reproduction using the authors' fixed 0.5 / 100-pixel rule; no ERSRR threshold tuning.

- Cohort: 579 scenes / 150 frozen 25 km groups; 67 plume / 512 no plume
- Scene recall / specificity / FPR: 0.164 / 0.912 / 0.088
- Scene AUROC / AP: 0.597 / 0.158
- Recall 95% CI: 0.040-0.361
- Specificity 95% CI: 0.877-0.941
- Validity-aware pixel AP / IoU / Dice: 0.0069 / 0.0075 / 0.0150
- Checkpoint SHA-256: `fbcdcad062fa199d7c66631f3607eb79d92f81650fbac21ce64332f2ad2b7a34`

## Interpretation

This is a checkpoint baseline on ERSRR's stricter spatially disjoint cohort, not a reproduction of the paper's official aggregate test metric. The released checkpoint trained on the official training split, so ERSRR does not use its internal official-train validation subset to recalibrate this model.
