# CloudSEN12+ fresh no-plume test crop acquisition

Generated: 2026-07-16T17:04:46.551603+00:00.

## Result

- Fully resolved exact-product samples attempted: **368**.
- Crops acquired and hash-verified: **368**.
- Pre-cloud radiometry/geometry gate pass: **368**.
- Acquisition errors: **0**.
- Ignored raster bytes: **235,585,490**.

## Contract

- Exact target and background products; no product substitution.
- 200x200 pixels at 10 m in the target product CRS (2x2 km).
- Twelve uint16 bands: six target then six reference bands.
- Sentinel-2 native 10 m bands reproduce the audited producer statistics; B11/B12 follow the released GEE-nearest/nearest-down/bilinear-up algorithm, but the unpublished producer GEE pixels cannot be recovered bit-for-bit from the public JP2 archive.
- Fresh external negatives use an identically gridded zero plume mask; the spatial cloud-mask policy is frozen separately before feature extraction.
- Landsat cloud support is the target/reference union of QA_PIXEL fill, dilated-cloud, cirrus, cloud, shadow, and snow bits.
- Published CloudSEN12+ no-plume scene truth is retained, including the declared 42 scenes with non-clear pixels.
- The fixed candidate was authorized before fresh-test access; the exact MARS paper cache remained unopened.
