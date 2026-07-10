# ERSRR compact residual U-Net experiment

Generated: `2026-07-10T00:24:03.612398+00:00`

Cohort: 65 scenes / 32 leakage-safe scene components; target > 1000 ppm·m.

| Architecture | Channels | Parameters | Sampled AUPRC | Sampled AUROC | Sampled F1 | Sampled IoU | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_resunet | 5 | 144,433 | 0.0953 | 0.5793 | 0.1203 | 0.1016 | 96.6 |
| physics_resunet | 11 | 144,913 | 0.0767 | 0.5504 | 0.0868 | 0.0622 | 93.7 |

Outer test groups are never used for normalization or early stopping. Each outer training fold contains a disjoint group-held-out inner validation subset, which also calibrates the decision threshold.
