# Frozen ERSRR MARS v3 strict-spatial evaluation

Checkpoint and all operating rules were selected on internal validation before this run.

- Primary scene score: validation-selected neural/proposal blended probability
- Cohort: 4401 scenes / 150 frozen 25 km groups
- Scene recall / specificity / FPR: 0.284 / 0.965 / 0.035
- Scene AUROC / AP: 0.727 / 0.096
- Recall 95% CI: 0.102-0.575
- Specificity 95% CI: 0.953-0.976
- Pixel AP / IoU / Dice: 0.0717 / 0.1062 / 0.1919
- Promotion gate: FAIL

## Decision

V3 does not clear the frozen full-MARS gate. Preserve this result; do not retune from strict-test behavior.
