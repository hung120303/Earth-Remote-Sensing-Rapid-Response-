# MARS-S2L group-disjoint pixel baselines

Predeclared spatial baseline ladder on the verified development tranche; not a final paper estimate.

- Pixel logistic training: 768 scenes / 118,744 plume pixels / 393,216 background pixels
- Operating threshold and minimum component area selected on 384 internal-validation scenes only
- Benchmark: 579 strict-spatial scenes / 150 groups; 67 plume / 512 no plume

| Model | Val recall | Val FPR | Test recall | Test specificity | Pixel AP | Pixel IoU | Recall 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| valid_aware_mbmp | 0.008 | 0.043 | 0.015 | 0.990 | 0.0165 | 0.0000 | 0.000-0.077 |
| pixel_logistic_13_features | 0.039 | 0.031 | 0.000 | 0.988 | 0.0091 | 0.0000 | 0.000-0.000 |

## Decision

Validation-selected spatial baseline: `pixel_logistic_13_features`. It does not clear the promotion gate. Local per-pixel spectra are insufficient; implement the predeclared multi-scale target/reference encoder with joint scene presence, segmentation, and observability heads.

These baselines test whether local target/reference spectra and MBMP alone provide an adequate operating rule. Candidate neural architecture and calibration remain validation-only until frozen; this benchmark is not used for hyperparameter search.
