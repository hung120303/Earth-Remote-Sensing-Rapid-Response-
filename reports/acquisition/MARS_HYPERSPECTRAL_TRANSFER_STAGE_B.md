# MARS-Hyperspectral transfer Stage B

**Decision:** FAIL

Authoritative labels came only from published train-split `plumemask.tif` pixels. Validation, test, full-tile, retrieval, and official MARS-S2L outcomes remained unopened.

## Leakage-safe target candidates

- Positive scene pairs within 6 hours: 500
- Positive pairs within 1 hour: 188
- Reviewed negative pairs within 1 hour: 272
- All high-confidence pairs within 1 hour: 460
- Dense reprojection candidates within 15 minutes: 137
- Novel 25 km groups beyond every MARS-S2L location: 204
- Countries: 49
- Unique Sentinel-2 products: 442

## Frozen gates

- PASS `minimum_positive_scene_pairs`
- FAIL `minimum_negative_scene_pairs`
- PASS `minimum_novel_25km_groups`
- PASS `minimum_countries`
- PASS `minimum_high_confidence_pairs_within_1_hour`

## Claim boundary

FAIL does not authorize target-band download or modeling under this protocol. Retain compact receipts and seek an independent source that satisfies the failed gate without changing it post hoc.
