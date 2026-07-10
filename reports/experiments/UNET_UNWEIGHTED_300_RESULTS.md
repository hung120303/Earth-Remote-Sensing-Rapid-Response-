# ERSRR compact residual U-Net experiment

Generated: `2026-07-10T00:19:54.068542+00:00`

Cohort: 65 scenes / 32 leakage-safe scene components; target > 300 ppm·m.

| Architecture | Channels | Parameters | Sampled AUPRC | Sampled AUROC | Sampled F1 | Sampled IoU | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_resunet | 5 | 144,433 | 0.4168 | 0.5659 | 0.4988 | 0.3546 | 103.4 |
| physics_resunet | 11 | 144,913 | 0.4079 | 0.5556 | 0.4939 | 0.3502 | 105.6 |

Outer test groups are never used for normalization or early stopping. Each outer training fold contains a disjoint group-held-out inner validation subset, which also calibrates the decision threshold.
