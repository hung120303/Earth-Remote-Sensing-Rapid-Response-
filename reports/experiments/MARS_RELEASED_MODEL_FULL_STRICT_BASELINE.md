# MARS-S2L released MARSS2L_20250326 on the frozen ERSRR strict cohort

Inference-only reproduction using the authors' fixed 0.5 / 100-pixel rule; no ERSRR threshold tuning.

- Cohort: 4401 scenes / 150 frozen 25 km groups; 67 plume / 4334 no plume
- Scene recall / specificity / FPR: 0.642 / 0.905 / 0.095
- Scene AUROC / AP: 0.818 / 0.352
- Recall 95% CI: 0.480-0.861
- Specificity 95% CI: 0.885-0.924
- Validity-aware pixel AP / IoU / Dice: 0.2481 / 0.1329 / 0.2346
- Checkpoint SHA-256: `be634fb9e24dc4877f44c1ff9f69972e6f0453e30d70c0dc03677876340ef246`

## Interpretation

This is a checkpoint baseline on ERSRR's stricter spatially disjoint cohort, not a reproduction of the paper's official aggregate test metric. The released checkpoint trained on the official training split, so ERSRR does not use its internal official-train validation subset to recalibrate this model.
