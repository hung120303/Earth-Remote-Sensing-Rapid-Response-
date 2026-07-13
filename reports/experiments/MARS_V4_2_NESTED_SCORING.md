# ERSRR v4.2 morphology scene-ranker audit

Development-only result on spatially isolated internal groups; the strict cohort was not loaded.

- Frozen segmentation checkpoint: epoch 12
- Candidate formulas: 14
- Nested AP / AUROC: 0.862 / 0.964
- Nested recall at 5% FPR target: 0.828 (observed FPR 0.049)
- Final development formula: `top_logits_2pct_mean`
- Final full-validation AP / AUROC: 0.817 / 0.956

## Decision

Reject the v4.2 morphology ranker before strict evaluation because its group-held estimate does not clear every v3 development gate. Preserve the result and do not select a formula from strict behavior.

The final formula is selected for a possible frozen benchmark run only; nested held-out metrics are the development estimate.
