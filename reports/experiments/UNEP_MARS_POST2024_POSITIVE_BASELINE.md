# UNEP MARS post-2024 positive baseline

This positive-only result supports recall and mask-overlap conclusions only. It cannot estimate AP, false-positive rate, precision, specificity, or AUROC.

| Role | Endpoint | Rows | Groups | Recall | 95% group CI | Pixel IoU | 95% group CI |
|---|---|---:|---:|---:|---:|---:|---:|
| auxiliary_training | released_fixed_0_5 | 135 | 27 | 0.6444 | [0.5030, 0.8061] | 0.4440 | [0.3743, 0.5199] |
| auxiliary_training | current_sentinel_mask | 135 | 27 | 0.5778 | [0.4279, 0.7576] | 0.3945 | [0.3154, 0.4806] |
| development | released_fixed_0_5 | 4 | 4 | 0.7500 | [0.2500, 1.0000] | 0.4308 | [0.0917, 0.7507] |
| development | current_sentinel_mask | 4 | 4 | 0.7500 | [0.2500, 1.0000] | 0.3235 | [0.0281, 0.7031] |

The auxiliary rows are a training-domain diagnostic. The four development groups are isolated confirmation and did not select either endpoint.
