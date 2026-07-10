# ERSRR MARS joint MIL v2 development result

Validation-first architecture experiment; strict-spatial results appear only when the explicitly frozen test flag was used.

- Model: `ersrr_mars_joint_mil_v2` / 2,752,107 parameters
- Best epoch: 10 / 15
- Validation recall at FPR <= 0.05: 0.398 (observed FPR 0.047)
- Validation AUROC/AP: 0.685 / 0.627
- Validation mask Dice: 0.319
- Checkpoint SHA-256: `fb8e75287f6ab07a5ab181137b6d6171dd6eae009a4c450113100ce8cf4d07b9`

## Frozen strict-spatial evaluation

- Scene recall/specificity/FPR: 0.149 / 0.988 / 0.012
- Scene AUROC / unweighted AP: 0.752 / 0.368
- Group-bootstrap recall 95% CI: 0.044-0.350
- Group-bootstrap specificity 95% CI: 0.978-0.996
- Pixel AP/IoU/Dice: 0.0611 / 0.0692 / 0.1295
- Selective weighted coverage / accepted no-plume NPV: 0.810 / 0.991

## Decision

MIL v2 does not clear the strict-spatial gate. Relative to joint v1, the frozen result improves
strict scene recall by 10x while reducing FPR from 0.086 to 0.012, and improves pixel AP by 1.7x.
Freeze this result and use only validation/external data for the next decision. The released
MARS-S2L model remains the next required baseline; no claim of publication-grade performance is
warranted from this development tranche.
