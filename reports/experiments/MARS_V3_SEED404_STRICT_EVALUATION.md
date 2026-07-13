# Frozen ERSRR MARS v3 strict-spatial evaluation

Checkpoint and all operating rules were selected on internal validation before this run.

- Primary scene score: validation-selected neural/proposal blended probability
- Cohort: 4401 scenes / 150 frozen 25 km groups
- Scene recall / specificity / FPR: 0.388 / 0.956 / 0.044
- Scene AUROC / AP: 0.733 / 0.197
- Recall 95% CI: 0.149-0.709
- Specificity 95% CI: 0.943-0.970
- Pixel AP / IoU / Dice: 0.0819 / 0.0653 / 0.1226
- Promotion gate: FAIL

## Decision

V3 does not clear the frozen full-MARS gate. Preserve this result; do not retune from strict-test behavior.
