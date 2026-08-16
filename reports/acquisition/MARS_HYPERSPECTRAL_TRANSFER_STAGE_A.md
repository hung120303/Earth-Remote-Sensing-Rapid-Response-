# MARS-Hyperspectral transfer Stage A

**Decision:** PASS

This was a metadata-only audit. No hyperspectral raster, mask truth, MARS model score, or protected MARS outcome was read.

## Coverage

- Samples: 11,531
- Unique location names: 2,463
- Countries: 72
- Coordinate-resolved samples: 5,950
- Coordinate-unresolved samples: 5,581
- Resolved non-protected locations: 282
- Eligible train samples after clear/protected-site filters: 411
- Exact MARS official-test identity overlaps: 4,949
- Resolved overlaps after 25 km exclusion: 5,355

## Existing local target acquisitions

- Within 15 minutes: 0 HSI samples / 0 pairs
- Within 1 hour: 1 HSI samples / 1 pairs
- Within 6 hours: 3 HSI samples / 3 pairs

## Public Copernicus catalog

- Eligible HSI training observations queried: 411
- Sentinel-2 L1C candidates within 15 minutes: 30
- Sentinel-2 L1C candidates within 1 hour: 111
- Sentinel-2 L1C candidates within 6 hours: 122
- Unique Sentinel-2 L1C products: 81

## Frozen gates

- PASS `minimum_total_samples`
- PASS `minimum_countries`
- PASS `minimum_coordinate_resolved_non_mars_test_locations`
- PASS `minimum_existing_or_catalog_query_candidates_within_6_hours`

## Claim boundary

PASS authorizes only Stage B label/georeference and target-catalog audit. It does not authorize model training or establish transferability.
