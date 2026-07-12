# Authenticated EMIT V002 plume-raster audit

- Collection: `EMITL2BCH4PLM.002`
- Earthdata order: [6313106439](https://search.earthdata.nasa.gov/downloads/6313106439)
- Protected rasters: 12 / 12
- Raw bytes: 1,005,045 (ignored; not committed)
- SHA-256-bound files passing contract: 12 / 12
- Credentials, cookies, and signed URLs recorded: no

## Verified scientific contract

All 12 pilot products are one-band `float32` V002 rasters in EPSG:4326 at the common nominal 60 m grid spacing, with `-9999` nodata and embedded `ppm m` units. Their filename timestamp, plume-complex identifier, and source CH4ENH scene tags agree with the public CMR pilot.

Valid support occupies 9.60% of the cropped raster pixels. Scene maxima span 1,724.0-22,890.0 ppm m; negative in-footprint retrieval values are retained and must not be reinterpreted as no-plume labels.

| Plume complex | Shape | Valid pixels | Max ppm m | SHA-256 |
|---|---:|---:|---:|---|
| `EMIT_L2B_CH4PLM_002_20241017T083946_003674` | 272x296 | 3,793 | 11,371.5 | `1d7303469523a6031ae6827872d657c1d9fadd181dd75f1a49bcd120502a4060` |
| `EMIT_L2B_CH4PLM_002_20241019T101302_003676` | 258x254 | 1,762 | 1,775.3 | `3c7e3da63f3689850b126bc6d791f7567f6cf6178df6bef1374d77a0407f488a` |
| `EMIT_L2B_CH4PLM_002_20241020T075129_003680` | 299x345 | 8,718 | 8,710.9 | `760ddb7ae502fbfb0483b56e72bc65ac9e4567dbe2da78929347c70652b48ef2` |
| `EMIT_L2B_CH4PLM_002_20241020T092144_003679` | 283x432 | 9,533 | 2,976.2 | `cf2b86534498001aa863168c0ded21b6b7d8cba19eb89b1eecfb0f43ea02170d` |
| `EMIT_L2B_CH4PLM_002_20241020T170504_003677` | 607x572 | 72,004 | 22,890.0 | `3422bcc8b8774a2ad439a3f6219cc57aedb124414940c39e220aea1bdccafa63` |
| `EMIT_L2B_CH4PLM_002_20241022T074913_003701` | 261x252 | 1,745 | 1,724.0 | `216f0d61999aa4156a0a1b705ded2a26cbf34e66023205070666cc2726ff2fff` |
| `EMIT_L2B_CH4PLM_002_20241023T070038_003703` | 247x250 | 1,506 | 3,913.8 | `3a49212df096d39363acb9d1fcb366617e526a88b0cfa51bf8d466b3a7d859c4` |
| `EMIT_L2B_CH4PLM_002_20241023T083741_003699` | 220x231 | 347 | 2,154.0 | `936565ced0663b0ce4e7706c022fb8f533aa06921384d4d037a534e899c40e93` |
| `EMIT_L2B_CH4PLM_002_20241026T061152_003688` | 237x256 | 1,580 | 1,962.2 | `914a3963aeecacb6840c2f466974ad4b1baa14fd0458a24ab294e08cbc571d64` |
| `EMIT_L2B_CH4PLM_002_20241026T172133_003687` | 252x379 | 5,860 | 3,302.8 | `5962d242df8231f2c9ee8592efb24b3efb74dfc45ee223d6c1678af7f5187011` |
| `EMIT_L2B_CH4PLM_002_20241130T180310_003723` | 375x311 | 9,843 | 4,990.0 | `9cbab08b61a08401503419abae599968ca60912ff76fe03a5ae246fb3476dbe2` |
| `EMIT_L2B_CH4PLM_002_20250922T204933_003374` | 221x222 | 329 | 4,699.2 | `d360a6e246ef018d65c712feaaa4029319f215103e0db387254e60e20ad3c309` |

## Gate result

- Missing granules: 0.
- Unexpected granules: 0.
- Contract violations: 0.
- Pilot gate: `pass`.

This proves protected-raster integrity and scientific metadata consistency, not model generalization. The next external-data step is to acquire and align each source CH4ENH enhancement, uncertainty, and sensitivity bundle before constructing quantitative or observability-aware targets.
