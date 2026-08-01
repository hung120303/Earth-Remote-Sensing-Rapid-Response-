# MARS Gaussian-plume morphology audit

This is a fit-fold-only parameterization audit. It does not score a candidate model and did not load folds 0, 1, or 2 or the official paper test.

- Positive scenes: **1,524**
- Non-empty masks audited: **1,518**
- Empty positive masks excluded from geometry: **6**
- Physical sites: **81**
- Folds: **3, 4**
- Sentinel-2 / Landsat: **1,139 / 379**

## Geometry anchors

| Quantity | q05 | q25 | q50 | q75 | q95 |
|---|---:|---:|---:|---:|---:|
| area_pixels | 326.9 | 625 | 1008 | 1759 | 3603 |
| major_4sigma_m | 314.2 | 509.3 | 704.2 | 975.3 | 1303 |
| minor_4sigma_m | 123.9 | 161.9 | 205.5 | 284.9 | 492.5 |
| moment_aspect_ratio | 1.518 | 2.237 | 2.809 | 3.503 | 4.714 |
| major_axis_wind_alignment | 0.3352 | 0.8363 | 0.9518 | 0.9897 | 0.9996 |

## Interpretation

The audited quantiles may bound a synthetic morphology bank, but they may not be optimized against held-fold outcomes. Enhancement values are reported without physical units because the released MARS assets contain a documented ppm/ppb metadata conflict.
