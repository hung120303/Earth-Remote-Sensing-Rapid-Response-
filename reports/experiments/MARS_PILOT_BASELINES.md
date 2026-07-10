# MARS-S2L contract-pilot baselines

These results are an end-to-end adapter and evaluation smoke test on 18 deliberately balanced samples. They are not an accuracy estimate and are not suitable for a paper claim.

- Source revision: `c26b1d7e31a0c5241fa37c9140802622c215eb32`
- Pilot identity: `4e23043682f4f60a8a8b811a50fd7a080580c743899de0be15cef791d7104942`
- Split: 6 train / 6 validation / 6 test; each has 3 plume and 3 no-plume scenes.
- Test scenes: deterministic samples from locations absent from train and validation.
- Thresholds/component sizes: validation only; test never used for selection.

| Model | Val recall | Val FPR | Test recall | Test specificity | Test pixel AP | Test IoU |
|---|---:|---:|---:|---:|---:|---:|
| MBMP release-compatible | 0.667 | 0.000 | 0.333 | 0.333 | 0.016 | 0.000 |
| MBMP validity-aware | 0.667 | 0.000 | 0.333 | 0.333 | 0.016 | 0.000 |
| Pixel logistic (13 features) | 1.000 | 0.000 | 0.667 | 0.333 | 0.013 | 0.000 |

## Interpretation

- Passing this test means the native target/reference adapter, zero-mask negatives, explicit observability mask, validation-only operating rule, and test path execute coherently.
- With only three test positives and three test negatives, one mistake changes recall or FPR by 0.333. Model rankings are therefore unstable and must not drive an architecture promotion.
- The logistic model is a low-capacity pipeline check. The full cohort and five-seed, site-blocked protocol remain required before comparing architectures.
- Enhancement units are unresolved: conflict: GeoTIFF tags may say DeltaCH4(ppm), while the pinned MARS-S2L README describes enhancement values as ppb; do not use quantitative units until reconciled with the data producer.

The next experiment should reproduce the released MARS-S2L/CH4Net baselines on the frozen cohort, then compare the dual-temporal selective architecture under the same test and calibration contract.
