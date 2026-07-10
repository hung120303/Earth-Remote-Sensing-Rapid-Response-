# MARS-S2L group-disjoint scene baselines

Development-tranche result; architecture-screening evidence, not the final paper estimate.

- Training: 768 scenes
- Validation: 384 scenes; thresholds selected here only
- Strict spatial test: 579 scenes / 150 groups
- Features: 98 raw/change + 30 MBMP/physics
- Confidence intervals: 2,000 bootstrap resamples of strict 25 km groups

| Model | Val recall | Val FPR | Test recall | Test specificity | Recall 95% CI | Specificity 95% CI | Selective coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| valid_aware_mbmp_p99 | 0.008 | 0.031 | 0.000 | 0.996 | 0.000-0.000 | 0.990-1.000 | 0.047 |
| raw_scene_logistic | 0.141 | 0.047 | 0.075 | 0.963 | 0.000-0.278 | 0.942-0.981 | 0.368 |
| physics_scene_logistic | 0.133 | 0.043 | 0.030 | 0.979 | 0.000-0.109 | 0.961-0.992 | 0.480 |
| physics_hist_gradient_boosting | 0.141 | 0.047 | 0.104 | 0.947 | 0.034-0.244 | 0.920-0.969 | 0.188 |

## Decision

Validation-selected development baseline: `raw_scene_logistic`. It does not clear the research promotion gate; proceed to a joint presence/segmentation model with hard-negative mining, not a larger backbone alone.

Specificity and FPR are estimable from 512 group-diverse strict-test negatives, while recall uses all 67 strict-test positives. The tranche is class-enriched, so representative-weighted calibration metrics are reported separately and final claims still require the full frozen cohort and five learned-model seeds.
