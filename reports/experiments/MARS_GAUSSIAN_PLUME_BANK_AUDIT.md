# Gaussian-plume bank distribution audit

This audit compares a deterministic analytical bank with real positive-mask geometry from MARS development folds 3 and 4. It uses no model outcome.

- Synthetic audit members: **5,000**
- Bank capacity: **20,000** disjoint indexed templates
- Distribution gate: **PASS**

| Metric | max discrepancy | gate |
|---|---:|---:|
| area_pixels | 0.0942 | 0.2000 |
| major_4sigma_m | 0.0614 | 0.2000 |
| minor_4sigma_m | 0.0687 | 0.2000 |
| moment_aspect_ratio | 0.0837 | 0.2000 |
| major_axis_wind_alignment | 0.0253 | 0.0500 |

The bank matches geometry only. Peak delta-CH4 is sampled log-uniformly from 500 to 10,000 in the released LUT input units; no physical flux claim is made.
