# MARS-S2L raster-contract pilot audit

- Source: `UNEP-IMEO/MARS-S2L`
- Revision: `c26b1d7e31a0c5241fa37c9140802622c215eb32`
- Pilot identity: `4e23043682f4f60a8a8b811a50fd7a080580c743899de0be15cef791d7104942`
- Assets: 54 / 19,835,687 bytes, all SHA-256 verified
- Samples: 18 (9 plume / 9 no plume)

## Verified native product contract

- Image: 200 x 200 pixels at 10 m in sample-local UTM; 12-band uint16 target/reference pair.
- Target bands: `B02, B03, B04, B08, B11, B12`.
- Reference bands: `B02_bg, B03_bg, B04_bg, B08_bg, B11_bg, B12_bg`.
- Cloud mask: one-band uint8; observed values 0/1; explicit classes override ambiguous nodata metadata.
- Positive label assets: binary plume mask plus float64 enhancement raster.
- Enhancement units: unresolved; GeoTIFF descriptions say `DeltaCH4(ppm)` where present, while the pinned dataset README says ppb.
- Negative label assets: image and cloud mask only; the adapter must create the zero target in memory.

| Split | Plume | No plume |
|---|---:|---:|
| train | 3 | 3 |
| val | 3 | 3 |
| test | 3 | 3 |

## Gate result

- Contract violations: 0.
- Non-fatal metadata warnings: 22.
- All samples pass: `true`.
- Paired all-band valid fraction: mean 0.999264, range 0.986750-1.000000.
- Positive plume area fraction: mean 0.040664, range 0.008375-0.164375.

The raster gate passes if the violation count remains zero. This validates the adapter contract, not model accuracy.

## Architecture consequences

1. The paper model needs a native MARS adapter: six target bands plus the corresponding six background bands.
2. B08 is part of the released data and should be retained; the legacy five-band ERSRR model remains a separate baseline.
3. Negative samples intentionally omit plume and enhancement files; the loader must synthesize a zero mask in memory without inventing a raw label asset.
4. Cloud and nodata support must gate both loss and evaluation. Product level remains explicitly Sentinel-2 MSI L1C.
5. Some rasters omit descriptive band tags, and cloud nodata can overlap the clear class; resolve roles from the pinned manifest and interpret mask classes explicitly.
6. Preserve enhancement values as raw until UNEP-IMEO reconciles the ppm TIFF tag with the ppb dataset documentation; do not make regression or flux claims from the current unit metadata.
