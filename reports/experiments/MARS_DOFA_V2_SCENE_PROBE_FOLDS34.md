# DOFA-v2 sensor-aware scene probe — folds 3/4

Frozen wavelength-conditioned target/reference features were projected with a fixed sparse random map and scored by a regularized linear probe. Selection is a two-way physical-site cross-fit on folds 3 and 4 only.

- Feature set / C / blend: `change_extreme` / 0.01 / 0.050
- AP delta vs current: +0.001087
- Recall delta at FPR 0.0713: +0.000000
- Paired-site AP 95% interval: [-0.000223, +0.002588]

| Fold | AP delta vs current | Recall delta vs current |
|---:|---:|---:|
| 3 | +0.000910 | +0.001319 |
| 4 | +0.000964 | +0.001305 |

Reject the DOFA-v2 scene candidate before fold-2 extraction.
