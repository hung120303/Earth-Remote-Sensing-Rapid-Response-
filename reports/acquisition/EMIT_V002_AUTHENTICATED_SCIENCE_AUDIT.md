# Authenticated EMIT V002 science-bundle audit

- Collection: `EMITL2BCH4ENH.002`
- Source scenes: 12 / 12
- Protected science rasters: 36 / 36
- Raw bytes: 400,014,715 (ignored; not committed)
- Three-product grid alignment: 12 / 12
- Exact CH4PLM-to-CH4ENH crop identity: 12 / 12
- Credentials, cookies, and signed URLs recorded: no

## Scientific result

Every source scene contains the official enhancement, uncertainty, and sensitivity COG. All three products share one EPSG:4326 pixel grid per scene, are one-band `float32`, use `-9999` nodata, and preserve a nominal 60 m spacing. Uncertainty and sensitivity are strictly positive on valid support.

The strongest provenance check passes: each vetted CH4PLM raster is an integer-offset crop of its declared source CH4ENH scene, and every valid plume pixel matches the source enhancement value exactly at `float32` precision.

| Source scene | Grid | Common valid | Plume pixels checked | Exact crop |
|---|---:|---:|---:|---:|
| `EMIT_L2B_CH4ENH_002_20241017T083946_2429106_009` | 2018x2036 | 2,017,097 | 3,793 | yes |
| `EMIT_L2B_CH4ENH_002_20241019T101302_2429307_021` | 2016x1918 | 1,935,281 | 1,762 | yes |
| `EMIT_L2B_CH4ENH_002_20241020T075129_2429405_003` | 2745x2649 | 2,002,171 | 8,718 | yes |
| `EMIT_L2B_CH4ENH_002_20241020T092144_2429406_003` | 2005x2122 | 2,116,754 | 9,533 | yes |
| `EMIT_L2B_CH4ENH_002_20241020T170504_2429411_003` | 1999x2173 | 2,140,403 | 72,004 | yes |
| `EMIT_L2B_CH4ENH_002_20241022T074913_2429605_016` | 2016x1979 | 1,730,126 | 1,745 | yes |
| `EMIT_L2B_CH4ENH_002_20241023T070038_2429705_008` | 2009x2031 | 2,039,485 | 1,506 | yes |
| `EMIT_L2B_CH4ENH_002_20241023T083741_2429706_026` | 2015x1849 | 1,866,175 | 347 | yes |
| `EMIT_L2B_CH4ENH_002_20241026T061152_2430004_002` | 2010x1953 | 793,499 | 1,580 | yes |
| `EMIT_L2B_CH4ENH_002_20241026T172133_2430011_019` | 2033x2219 | 2,270,593 | 5,860 | yes |
| `EMIT_L2B_CH4ENH_002_20241130T180310_2433512_007` | 1895x2250 | 2,097,384 | 9,843 | yes |
| `EMIT_L2B_CH4ENH_002_20250922T204933_2526514_006` | 1886x2278 | 1,995,402 | 329 | yes |

## Architecture consequence

The external evaluator should read the three products as a single quality-aware label bundle. Enhancement supplies the physical positive target; uncertainty and sensitivity define observability and stratification. They must not be fed to the Sentinel-2 detector, and unreviewed scene background must not be relabeled as `NO_PLUME`.

The official V2 guide defines enhancement as an adaptive matched-filter total-column estimate in `ppm m` and lists the enhancement, uncertainty, and sensitivity COGs as the three CH4ENH science files. The one available `.cmr.json` is retained as optional catalog evidence; it is not part of the three-raster scientific contract.

## Gate result

- Missing scenes: 0.
- Unexpected scenes: 0.
- Contract violations: 0.
- Pilot gate: `pass`.

Source: [NASA/JPL EMIT L2B GHG V2 Product User Guide](https://lpdaac.usgs.gov/documents/2250/EMIT_L2B_GHG_User_Guide_V2.pdf).
