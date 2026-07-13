# MethaneS2CM v5 signal and mask audit

Exploratory analysis on the frozen internal-development groups only; location-test imagery remains sealed.

| Evidence | Scene AP | AUROC | Recall at <=5% FPR | Pixel AP |
|---|---:|---:|---:|---:|
| `T_over_Tminus365_low` | 0.5509 | 0.5773 | 0.0479 | 0.1692 |
| `reference_consensus_low` | 0.5476 | 0.5745 | 0.0486 | 0.1809 |
| `reference_mean_low` | 0.5428 | 0.5710 | 0.0464 | 0.1685 |
| `reference_max_low` | 0.5427 | 0.5671 | 0.0456 | 0.1591 |
| `T_over_Tminus90_low` | 0.5399 | 0.5641 | 0.0564 | 0.1690 |
| `reference_max_absolute_change` | 0.5238 | 0.5541 | 0.0331 | 0.1380 |
| `T_over_Tminus365_high` | 0.5208 | 0.5544 | 0.0286 | 0.1734 |
| `T_over_Tminus90_high` | 0.5116 | 0.5457 | 0.0205 | 0.1729 |

- Positive-mask median area: 226 / 1024 pixels
- Positive crops with >=50% mask coverage: 2,552
- Positive crops with 100% mask coverage: 103
- Observable-pixel median: 1024 / 1024
- Baselines are diagnostic reference-only ratios, not fitted models or test results.
