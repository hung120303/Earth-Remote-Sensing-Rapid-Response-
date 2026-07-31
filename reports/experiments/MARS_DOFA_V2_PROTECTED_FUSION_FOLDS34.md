# Recall-protected DOFA-v2 fusion - folds 3/4

The fusion leaves every current-model score below a fixed confidence gate unchanged and maps every affected score back above that gate. DOFA-v2 can therefore rerank likely-plume scenes without altering the no-plume operating region.

- Selected gate / DOFA weight: 0.50 / 0.05
- AP delta vs current: +0.001208
- Recall delta at FPR 0.0713: +0.000000
- Paired-site AP 95% interval: [+0.000362, +0.002204]
- Operating confusion counts preserved: yes

| Gate | Weight | AP delta | Recall delta | AP CI lower | Promoted |
|---:|---:|---:|---:|---:|:---:|
| 0.25 | 0.05 | +0.001230 | +0.000000 | +0.000328 | yes |
| 0.25 | 0.10 | +0.002038 | +0.000000 | +0.000020 | yes |
| 0.25 | 0.20 | +0.001955 | +0.000000 | -0.002879 | no |
| 0.50 | 0.05 | +0.001208 | +0.000000 | +0.000362 | yes |
| 0.50 | 0.10 | +0.002073 | +0.000000 | +0.000259 | yes |
| 0.50 | 0.20 | +0.002413 | +0.000000 | -0.001851 | no |
| 0.75 | 0.05 | +0.001074 | +0.000000 | +0.000283 | yes |
| 0.75 | 0.10 | +0.001882 | +0.000000 | +0.000223 | yes |
| 0.75 | 0.20 | +0.002510 | +0.000000 | -0.001103 | no |

Freeze the selected recall-protected DOFA-v2 fusion for one-shot fold-2 extraction.
