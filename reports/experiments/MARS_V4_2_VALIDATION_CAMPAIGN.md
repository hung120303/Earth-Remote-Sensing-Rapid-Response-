# ERSRR v4.2 three-seed validation campaign

Frozen internal-development result. No strict-cohort imagery or labels were loaded.

| Seed | Best epoch | AP | AUROC | Recall @ <=5% FPR | FPR | Pixel Dice |
|---:|---:|---:|---:|---:|---:|---:|
| 606 | 20 | 0.8014 | 0.9529 | 0.8257 | 0.0500 | 0.7155 |
| 707 | 18 | 0.7833 | 0.9460 | 0.7980 | 0.0500 | 0.7131 |
| 808 | 16 | 0.8011 | 0.9486 | 0.8139 | 0.0494 | 0.7188 |

## Mean-gate result

| Metric | v4.2 mean +/- sample SD | v3 five-seed mean | Delta | Gate |
|---|---:|---:|---:|:---:|
| AP | 0.7953 +/- 0.0103 | 0.8103 | -0.0150 | fail |
| AUROC | 0.9492 +/- 0.0035 | 0.9288 | +0.0204 | pass |
| Recall @ <=5% FPR | 0.8125 +/- 0.0139 | 0.8313 | -0.0187 | fail |
| Positive-pixel Dice | 0.7158 +/- 0.0029 | 0.5770 | +0.1389 | pass |

## Decision

Do not load the strict MARS cohort for v4.2. The frozen three-seed campaign failed the predeclared internal promotion rule on: ap_not_below_v3_mean, recall_at_fpr5_not_below_v3_mean. Preserve this result and revise the research hypothesis before another strict comparison.

Internal validation and the MARS strict/paper benchmarks are different cohorts. These values must not be used to claim MARS-S2L superiority.
