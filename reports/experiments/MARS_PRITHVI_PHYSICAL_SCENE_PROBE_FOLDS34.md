# Physically scaled Prithvi scene probe — folds 3/4

MARS raw reflectance DN were restored with the correct x5,000 conversion before the pinned Prithvi normalization. Selection is a two-way physical-site cross-fit on development folds 3 and 4; no fold-2, folds-0/1, or paper-test outcome was used.

- Feature set / C / blend: `temporal_change` / 0.01 / 0.025
- AP delta vs current: +0.000091
- Recall delta at FPR 0.0713: +0.000000
- Paired-site AP 95% interval: [-0.000299, +0.000601]

| Fold | AP delta vs current | Recall delta vs current |
|---:|---:|---:|
| 3 | -0.000050 | +0.000000 |
| 4 | +0.000106 | +0.000000 |

Reject the physical-radiometry Prithvi candidate before fold-2 extraction.
