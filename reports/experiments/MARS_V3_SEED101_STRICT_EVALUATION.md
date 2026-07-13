# Frozen ERSRR MARS v3 strict-spatial evaluation

Checkpoint and all operating rules were selected on internal validation before this run.

- Primary scene score: validation-selected neural/proposal blended probability
- Cohort: 4401 scenes / 150 frozen 25 km groups
- Scene recall / specificity / FPR: 0.358 / 0.949 / 0.051
- Scene AUROC / AP: 0.762 / 0.118
- Recall 95% CI: 0.152-0.636
- Specificity 95% CI: 0.933-0.964
- Pixel AP / IoU / Dice: 0.0876 / 0.0819 / 0.1513
- Promotion gate: FAIL

## Decision

V3 does not clear the frozen full-MARS gate. Preserve this result; do not retune from strict-test behavior.
