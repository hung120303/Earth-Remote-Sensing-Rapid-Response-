# GHGSat target-catalog audit

**Decision: PASS**

## Query receipts

- expected: 352
- logical_queries: 352
- resumed_queries: 149
- network_queries: 203
- http_attempts: 203

## HTTP and byte receipts

- network_bytes_observed_this_execution: 100951
- historical_response_bytes_before_execution: 71894
- cumulative_response_bytes: 172845
- status_counts: {'200': 203}
- maximum_response_bytes_each: 2097152
- maximum_response_bytes_total: 268435456
- requests_append_only: .research/ghgsat_landfill_null/target_catalog/requests.jsonl
- responses_append_only: .research/ghgsat_landfill_null/target_catalog/responses.jsonl

## Candidate counts

- sentinel_2_l1c: 70
- landsat_8_9_level_1: 39

## Selection counts

- sentinel_2_l1c: 47
- landsat_8_9_level_1: 32

## Distinct counts

- selected_source_sensor_pairs: 79
- distinct_source_observations: 66
- distinct_sites: 44
- distinct_components: 44
- distinct_target_item_ids: 78

## Offset summary

- count: 79
- minimum: 8.976
- maximum: 3424.024
- mean: 1055.0625705569619
- median: 518.024

## Cloud summary

- count: 79
- minimum: 0.0
- maximum: 97.251
- mean: 9.494623124313256
- median: 1.49

## Gates

- minimum_selected_source_sensor_pairs: PASS (observed 79; required 28)
- minimum_distinct_source_observations_with_pair: PASS (observed 66; required 28)
- minimum_distinct_sites_with_pair: PASS (observed 44; required 20)
- minimum_novel_25km_components_with_pair: PASS (observed 44; required 20)
- minimum_distinct_target_item_ids: PASS (observed 78; required 20)
- all_queries_and_retained_items_valid: PASS (observed True; required True)

## Forbidden-resource proof

- target_assets_accessed: False
- asset_urls_retained: False
- preview_or_thumbnail_accessed: False
- reference_catalog_queried: False
- reference_assets_accessed: False
- ghgsat_assets_accessed: False
- protected_outcomes_accessed: False
- score_caches_accessed: False
- model_artifacts_accessed: False
- model_imports_used: False
- Source JSONL mutated: False

Claim boundary: A PASS establishes public target-catalog timing and footprint feasibility only. It does not establish local clear-sky observability, valid six-band current/reference crops, dense labels, training value, calibration, generalization, or superiority over MARS-S2L.
