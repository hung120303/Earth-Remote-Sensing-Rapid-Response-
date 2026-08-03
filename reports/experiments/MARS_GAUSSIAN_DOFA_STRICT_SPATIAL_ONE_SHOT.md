# Gaussian+DOFA strict-spatial candidate-specific replay

> This is a candidate-specific post-test diagnostic, not a fresh project-level holdout. Earlier ERSRR candidates had already opened this MARS exact-paper outcome view.

Rows: 4401; positives: 67; negatives: 4334; strict 25 km components: 150.

| Model | AP | AUROC | Recall | FPR | Precision |
|---|---:|---:|---:|---:|---:|
| Released MARS-S2L v3 | 0.352123 | 0.817546 | 0.641791 | 0.095062 | 0.094505 |
| Gaussian+DOFA | 0.374894 | 0.870049 | 0.611940 | 0.056991 | 0.142361 |

Primary superiority gate: **FAIL**.

Gaussian+DOFA changes only the frozen scene score. No pixel-IoU or localization-improvement claim is made.

Dataset provenance: UNEP-IMEO/MARS-S2L revision `c26b1d7e31a0c5241fa37c9140802622c215eb32`, CC-BY-NC-SA-4.0.
