# Frozen ERSRR MARS v3 strict-spatial evaluation

Checkpoint and all operating rules were selected on internal validation before this run.

- Primary scene score: validation-selected neural/proposal blended probability
- Cohort: 4401 scenes / 150 frozen 25 km groups
- Scene recall / specificity / FPR: 0.299 / 0.970 / 0.030
- Scene AUROC / AP: 0.744 / 0.195
- Recall 95% CI: 0.088-0.628
- Specificity 95% CI: 0.961-0.980
- Pixel AP / IoU / Dice: 0.1028 / 0.0966 / 0.1763
- Promotion gate: FAIL

## Decision

V3 does not clear the frozen full-MARS gate. Preserve this result; do not retune from strict-test behavior.
