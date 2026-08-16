# MARS-Hyperspectral transfer Stage B

**Decision:** FAIL

Authoritative labels came only from published train-split `plumemask.tif` pixels. Validation, test, full-tile, retrieval, and official MARS-S2L outcomes remained unopened.

## Leakage-safe target candidates

- Positive scene pairs within 6 hours: 286
- Positive pairs within 1 hour: 109
- Reviewed negative pairs within 1 hour: 161
- All high-confidence pairs within 1 hour: 270
- Dense reprojection candidates within 15 minutes: 81
- Novel 25 km groups beyond every MARS-S2L location: 155
- Countries: 41
- Unique Sentinel-2 products: 248

## Frozen gates

- PASS `minimum_positive_scene_pairs`
- FAIL `minimum_negative_scene_pairs`
- PASS `minimum_novel_25km_groups`
- PASS `minimum_countries`
- PASS `minimum_high_confidence_pairs_within_1_hour`

## Claim boundary

PASS establishes enough leakage-safe catalog candidates to preregister target-band acquisition and modeling. Target crop observability, dense reprojection validity, complementarity, and model improvement remain unproven.
