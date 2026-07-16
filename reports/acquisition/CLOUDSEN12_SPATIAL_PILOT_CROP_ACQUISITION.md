# CloudSEN12+ clear-scene spatial-pilot crop acquisition

Generated: 2026-07-16T15:50:35.054631+00:00.

## Result

- Fully resolved nonsealed samples attempted: **492**.
- Crops acquired and hash-verified: **492**.
- Pre-cloud radiometry/geometry gate pass: **492**.
- Acquisition errors: **0**.
- Ignored raster bytes: **287,840,064**.

## Contract

- Exact target and background products; no product substitution.
- 200x200 pixels at 10 m in the target product CRS (2x2 km).
- Twelve uint16 bands: six target then six reference bands.
- Sentinel-2 native 10 m bands reproduce the audited producer statistics; B11/B12 follow the released GEE-nearest/nearest-down/bilinear-up algorithm, but the unpublished producer GEE pixels cannot be recovered bit-for-bit from the public JP2 archive.
- Explicit clear-scene negatives require an identically gridded zero plume mask.
- Landsat cloud support is the target/reference union of QA_PIXEL fill, dilated-cloud, cirrus, cloud, shadow, and snow bits.
- Published CloudSEN12+ all-clear labels remain the negative cloud contract.
- Sealed-external samples were excluded.
