# CloudSEN12+ fresh-test exact-product resolution

Generated: 2026-07-16T16:33:56.408856+00:00.

The fixed model was authorized before these fresh external-test product identities were resolved.

## Result

- Exact-product samples: **374**.
- Fully resolved exact target/reference pairs: **368**.
- Unresolved pairs: **6**.
- Query errors: **0**.

## By sensor

- Sentinel-2: 368 resolved / 374 total.

## Integrity

- Sentinel-2 identities must match the exact published product URI and cover the source center.
- Landsat identities must match the exact USGS Collection-2 Level-1 product ID and cover the source center.
- Missing real-time Landsat products are reported unavailable; later tier products are not substituted.
- Resolved spectral assets are the six released MARS-S2L bands; Landsat also retains QA_PIXEL.
- Ignored row-level resolver SHA-256: `120cfac9d364c8c75ab44687307645ca5de353a24a7a4b18192f1c93cd948988`.
