# Stanford controlled-release Sentinel-2 L1C pair freeze

- Status: `frozen_complete`
- Eligible resolved S2 L1C targets: 169
- Complete target/reference pairs: 169
- Pair errors: 0
- References matching any source-cohort target: 0
- Ignored pair manifest: `.research/stanford_controlled_release_2024_2025/l1c_stress/pair_manifest.json`
- Pair manifest SHA-256: `10abbcd2605688525d0045526e516b9bce844b829e5f8c7e07a05acd69d290de`

References are prior-only Sentinel-2 Level-1C scenes on the same MGRS tile, from 1 hour through 31 days before the target, with catalog `eo:cloud_cover <= 20%`. Selection is deterministic by smallest time gap, then cloud cover, then item ID, after excluding every resolved source-cohort target scene ID and every listed 2024 Casa Grande controlled-release campaign UTC date. If that window is empty, a frozen 334-410 day seasonal fallback is selected by proximity to 365 days, then cloud cover, then item ID.

Only official public L1C assets accepted by exact `tileInfo.json` product-name and MGRS checks are frozen. Landsat remains pending exact USGS EROS Collection 2 Level-1 authentication; no L2 substitute is allowed.

Release labels/rates were neither selected nor emitted. No detector outcomes were accessed.
