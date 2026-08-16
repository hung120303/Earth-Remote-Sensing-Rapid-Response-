# STARCOP negative supplement — Stage B

**Decision:** FAIL

This audit used sparse HTTP byte ranges for deterministic train-negative `labelbinary.tif` members only. It did not download a full archive, inspect the test split, or query a Sentinel-2/Landsat catalog.

## Counts

- Resolved zero-mask rows: 1009
- Resolved flightlines: 256
- Rows passing the spatial filter: 25
- Eligible 25 km components: 8

The byte-range totals in the JSON cover the final successful execution. Earlier fail-closed parser/transport attempts are documented in the research ledger; exact label-range bytes from those pre-resume attempts were not persisted.

## Frozen gates

- `minimum_resolved_selected_negative_rows`: PASS
- `minimum_resolved_negative_flightlines`: PASS
- `minimum_eligible_25km_connected_components`: FAIL
- `all_retained_labels_and_coordinates_valid`: PASS

Even on PASS, every detailed row keeps `eligible_for_target_catalog=false`; a separately committed protocol is required before any target-satellite query.
