# EMIT V002 external post-hoc diagnostic

Frozen exploratory diagnosis only. This analysis does not alter the cohort, model, threshold, or once-only primary result, and it must not be used to retune v3.

- External cohort: 55 positive scenes / independent groups
- MARS comparator: 67 frozen strict positive scenes
- EMIT/Sentinel-2 absolute offset: median 1.266 h; 23 scenes at most one hour
- At most one hour: released MARS recall 0.043; ERSRR seed-mean recall 0.000
- Direction-free MBMP mask AUC median: external 0.550 vs strict MARS positives 0.799
- Absolute robust MBMP mask contrast median: external 0.167 vs strict MARS positives 1.363
- External scenes missed by both released MARS and every ERSRR seed: 52
- Within 25 km of an ERSRR v3 fit location: 24 / 55
- Within 25 km of a released MARS-S2L training location: 25 / 55
- Beyond 25 km of ERSRR v3 fit locations: n=31; released MARS recall 0.032; ERSRR seed-mean recall 0.019

## Recall by fixed EMIT/Sentinel-2 offset

| Stratum | Range | n | Released MARS recall | ERSRR seed-mean recall | Seed SD |
|---|---:|---:|---:|---:|---:|
| at_most_1_hour | 0.052-0.941 | 23 | 0.043 | 0.000 | 0.000 |
| over_1_to_2_hours | 1.040-1.907 | 16 | 0.000 | 0.000 | 0.000 |
| over_2_hours | 2.003-4.604 | 16 | 0.000 | 0.037 | 0.031 |

## Recall by EMIT footprint area

| Stratum | Range | n | Released MARS recall | ERSRR seed-mean recall | Seed SD |
|---|---:|---:|---:|---:|---:|
| small | 3636.000-7310.000 | 19 | 0.053 | 0.032 | 0.026 |
| medium | 7909.000-12146.000 | 18 | 0.000 | 0.000 | 0.000 |
| large | 12679.000-23848.000 | 18 | 0.000 | 0.000 | 0.000 |

## Recall by mask-aligned MBMP separability

| Stratum | Range | n | Released MARS recall | ERSRR seed-mean recall | Seed SD |
|---|---:|---:|---:|---:|---:|
| low | 0.501-0.529 | 19 | 0.000 | 0.000 | 0.000 |
| middle | 0.532-0.565 | 18 | 0.056 | 0.022 | 0.027 |
| high | 0.569-0.684 | 18 | 0.000 | 0.011 | 0.022 |

## Interpretation boundary

The EMIT mask is a cross-sensor, time-offset footprint, not simultaneous Sentinel-2 methane truth. Mask-aligned MBMP statistics test whether a colocated Sentinel-2 spectral signal is present; they do not prove methane causality. The MARS strict cohort remains the primary same-distribution plume/no-plume benchmark.

The candidate-builder's 25 km grouping prevents duplicates within this EMIT cohort. The separate proximity audit above determines novelty relative to model-development locations; those are different claims.

The JSON report contains all 55 external rows, all 67 strict-positive diagnostic rows, reflectance distribution comparisons, rank strata, score correlations, source hashes, and the exact frozen hit identities.
