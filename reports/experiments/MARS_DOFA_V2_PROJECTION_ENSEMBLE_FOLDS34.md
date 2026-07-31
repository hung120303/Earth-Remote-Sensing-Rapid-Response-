# DOFA-v2 projection-seed confirmation — folds 3/4

This preregistered confirmation holds the feature family, logistic regularization, blend, and folds fixed. It averages five newly seeded sparse-projection probes in log-odds space; the seed used during initial selection is excluded.

- Fixed feature set / C / blend: `change_extreme` / 0.01 / 0.050
- New projection seeds: `[20260780, 20260781, 20260782, 20260783, 20260784]`
- Aggregate AP delta vs current: +0.001251
- Aggregate recall delta at FPR 0.0713: -0.001312
- Paired-site AP 95% interval: [+0.000216, +0.002410]

| Projection seed | AP delta | Recall delta | Stable point gates |
|---:|---:|---:|:---:|
| 20260780 | +0.000374 | -0.001969 | no |
| 20260781 | +0.001605 | -0.002625 | no |
| 20260782 | +0.000980 | -0.000656 | no |
| 20260783 | +0.001442 | -0.001969 | no |
| 20260784 | +0.001171 | +0.000000 | yes |

Reject the DOFA-v2 projection ensemble before fold-2 extraction.
