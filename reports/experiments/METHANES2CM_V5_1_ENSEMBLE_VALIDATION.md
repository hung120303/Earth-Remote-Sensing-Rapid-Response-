# MethaneS2CM v5.1 three-seed ensemble validation

Frozen internal-development result. Location-test imagery remains sealed.

| Evaluation | Scene AP | AUROC | Recall @ 5% FPR target | Realized FPR | Pixel Dice | Pixel IoU |
|---|---:|---:|---:|---:|---:|---:|
| Five-fold 25 km group-held calibration | 0.8647 | 0.8622 | 0.4687 | 0.0519 | 0.4389 | 0.2811 |
| Final all-development frozen rule | 0.8658 | 0.8655 | 0.4671 | 0.0499 | 0.4424 | 0.2840 |

- Seeds: 1101, 2202, 3303
- Frozen final scene threshold at 5% target FPR: 0.733579
- Frozen pixel threshold: 0.40
- Ensemble pixel AP: 0.3090
- Bootstrap intervals resample the 64 frozen 25 km groups.
- Checkpoint selection still used this development cohort; the sealed location test is the external estimate.
