# ERSRR v3 strict post-hoc diagnostic

Exploratory analysis of frozen predictions. It must not be used to retune v3; any resulting v4 hypothesis requires a new untouched test cohort.

- Cohort: 4401 scenes / 150 frozen 25 km groups
- Positives / negatives: 67 / 4334
- Consensus ERSRR plume misses (0/5 seeds): 36
- Plumes detected by all ERSRR seeds: 13
- No-plume scenes falsely flagged by at least one ERSRR seed: 451

## Positive recall by plume area

| Stratum | Range | n | MARS-S2L rate | ERSRR seed-mean rate | Seed SD |
|---|---:|---:|---:|---:|---:|
| small | 111.000-621.000 | 23 | 0.522 | 0.304 | 0.082 |
| medium | 643.000-1389.000 | 22 | 0.545 | 0.145 | 0.067 |
| large | 1425.000-8123.000 | 22 | 0.864 | 0.509 | 0.034 |

## Positive recall by wind speed

| Stratum | Range | n | MARS-S2L rate | ERSRR seed-mean rate | Seed SD |
|---|---:|---:|---:|---:|---:|
| low | 0.328-2.443 | 23 | 0.565 | 0.243 | 0.059 |
| middle | 2.622-3.704 | 22 | 0.727 | 0.464 | 0.034 |
| high | 3.705-10.050 | 22 | 0.636 | 0.255 | 0.079 |

## Positive recall by target/reference interval

| Stratum | Range | n | MARS-S2L rate | ERSRR seed-mean rate | Seed SD |
|---|---:|---:|---:|---:|---:|
| low | 1.993-5.000 | 23 | 0.652 | 0.330 | 0.076 |
| middle | 5.000-10.000 | 22 | 0.636 | 0.282 | 0.053 |
| high | 10.000-30.000 | 22 | 0.636 | 0.345 | 0.022 |

## Negative FPR by wind speed

| Stratum | Range | n | MARS-S2L rate | ERSRR seed-mean rate | Seed SD |
|---|---:|---:|---:|---:|---:|
| low | 0.015-2.324 | 1445 | 0.087 | 0.030 | 0.008 |
| middle | 2.324-3.888 | 1445 | 0.081 | 0.028 | 0.009 |
| high | 3.892-11.648 | 1444 | 0.116 | 0.052 | 0.015 |

## Negative FPR by target/reference interval

| Stratum | Range | n | MARS-S2L rate | ERSRR seed-mean rate | Seed SD |
|---|---:|---:|---:|---:|---:|
| low | 0.000-5.000 | 1445 | 0.076 | 0.029 | 0.007 |
| middle | 5.000-10.001 | 1445 | 0.082 | 0.035 | 0.010 |
| high | 10.001-115.005 | 1444 | 0.126 | 0.046 | 0.014 |

## Interpretation boundary

Exploratory description after the once-only strict campaign. Do not retune v3 from these strata or examples. Use them only to preregister v4 hypotheses and acquisition strata, then evaluate v4 on a newly untouched cohort.

The JSON report contains country and cloud strata plus deterministic error-atlas sample ids.
