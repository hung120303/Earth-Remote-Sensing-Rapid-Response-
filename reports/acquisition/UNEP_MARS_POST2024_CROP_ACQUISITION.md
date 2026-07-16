# UNEP MARS post-2024 exact crop acquisition

Generated: 2026-07-16T03:23:53.555319+00:00.

## Result

- Fully resolved nonsealed samples attempted: **167**.
- Crops acquired and hash-verified: **141**.
- Pre-cloud radiometry/geometry gate pass: **141**.
- Acquisition errors: **26**.
- Ignored raster bytes: **87,674,839**.

## Contract

- Exact UNEP target and background products; no product substitution.
- 200×200 pixels at 10 m in the target product CRS (2×2 km).
- Twelve uint16 bands: six target then six reference bands.
- UNEP MultiPolygon plume truth is rasterized on the identical grid.
- Landsat cloud support is the target/reference union of QA_PIXEL fill, dilated-cloud, cirrus, cloud, shadow, and snow bits.
- Sentinel-2 CloudSEN12+ masks remain a separate required acquisition gate.
- Sealed-external samples were excluded.
