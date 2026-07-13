# Frozen ERSRR MARS v3 strict-spatial evaluation

Checkpoint and all operating rules were selected on internal validation before this run.

- Primary scene score: validation-selected neural/proposal blended probability
- Cohort: 4401 scenes / 150 frozen 25 km groups
- Scene recall / specificity / FPR: 0.269 / 0.976 / 0.024
- Scene AUROC / AP: 0.752 / 0.112
- Recall 95% CI: 0.095-0.587
- Specificity 95% CI: 0.969-0.983
- Pixel AP / IoU / Dice: 0.0848 / 0.0794 / 0.1471
- Promotion gate: FAIL

## Decision

V3 does not clear the frozen full-MARS gate. Preserve this result; do not retune from strict-test behavior.
