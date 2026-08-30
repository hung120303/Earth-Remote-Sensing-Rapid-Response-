# GHGSat reference-catalog audit

**Decision: PASS**

## Query receipts

- primary_expected: 79
- primary_logical_queries: 79
- seasonal_logical_queries: 9
- total_logical_queries: 88
- resumed_queries: 0
- network_queries: 88
- http_attempts: 89

## HTTP and byte receipts

- network_bytes_observed_this_execution: 661830
- historical_response_bytes_before_execution: 0
- cumulative_response_bytes: 661830
- status_counts: {'200': 88, '429': 1}
- maximum_response_bytes_each: 2097152
- maximum_response_bytes_total: 134217728
- requests_append_only: .research/ghgsat_landfill_null/reference_catalog/requests.jsonl
- responses_append_only: .research/ghgsat_landfill_null/reference_catalog/responses.jsonl

## Candidate counts

- primary: 252
- seasonal: 18

## Selection counts

- primary: 70
- seasonal: 6

## Distinct counts

- selected_target_reference_pairs: 76
- distinct_source_observations: 64
- distinct_sites: 43
- distinct_components: 43
- distinct_reference_item_ids: 75

## Candidate counts by sensor

- sentinel_2_l1c: 214
- landsat_8_9_level_1: 56

## Selection counts by sensor

- sentinel_2_l1c: 47
- landsat_8_9_level_1: 29

## Gates

- minimum_selected_target_reference_pairs: PASS (observed 76; required 28)
- minimum_distinct_source_observations_with_reference: PASS (observed 64; required 28)
- minimum_distinct_sites_with_reference: PASS (observed 43; required 20)
- minimum_novel_25km_components_with_reference: PASS (observed 43; required 20)
- all_queries_and_retained_items_valid: PASS (observed True; required True)

## Access-boundary proof

- metadata_search_endpoints_only: True
- item_detail_queried: False
- target_assets_accessed: False
- reference_assets_accessed: False
- asset_urls_retained: False
- item_links_retained: False
- preview_or_thumbnail_accessed: False
- raster_bytes_accessed: False
- ghgsat_assets_accessed: False
- model_artifacts_accessed: False
- protected_outcomes_accessed: False
- Frozen inputs mutated: False

Claim boundary: A PASS establishes deterministic prior-reference catalog feasibility only. It does not establish asset accessibility, local cloud/radiometric observability, a dense no-plume mask, training value, calibration, generalization, or superiority over MARS-S2L.
