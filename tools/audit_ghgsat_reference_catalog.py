"""Audit the frozen GHGSat prior-reference catalogs, with an offline default.

Without ``--execute-network`` this command verifies the exact hash-bound
protocol and frozen local inputs only.  Explicit network execution performs all
79 primary metadata-only searches and performs a seasonal search only when a
row has no valid primary candidate.  No item-detail or asset request exists.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

try:
    from tools import audit_ghgsat_target_catalog as target
except ModuleNotFoundError as exc:  # Direct ``python tools/...py`` execution.
    if exc.name != "tools":
        raise
    import audit_ghgsat_target_catalog as target  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_RELATIVE_PATH = "configs/mars_ghgsat_reference_catalog_protocol.json"
EXPECTED_PROTOCOL = ROOT / PROTOCOL_RELATIVE_PATH
EXPECTED_PROTOCOL_SHA256 = "04d88d0fb6a11c5c54c4b1f38f6872de79db436240e175dfced75b82f2050f24"
SOURCE_IDENTITY_FIELDS = ("site_ID", "obs_ID", "date", "sat_ID", "target_sensor")
OBSERVATION_IDENTITY_FIELDS = ("site_ID", "obs_ID", "date", "sat_ID")
CATALOG_ORDER = ("sentinel_2_l1c", "landsat_8_9_level_1")
SELECTED_TARGET_FIELDS = {
    "site_ID", "obs_ID", "date", "sat_ID", "component_id", "target_sensor",
    "target_item_id", "target_collection", "target_datetime",
    "source_target_offset_seconds", "absolute_source_target_offset_seconds",
    "eo_cloud_cover", "geometry", "bbox",
}
REQUIRED_OUTPUTS = {
    "ignored_root": ".research/ghgsat_landfill_null/reference_catalog",
    "ignored_requests": ".research/ghgsat_landfill_null/reference_catalog/requests.jsonl",
    "ignored_responses": ".research/ghgsat_landfill_null/reference_catalog/responses.jsonl",
    "ignored_candidates": ".research/ghgsat_landfill_null/reference_catalog/candidates.jsonl",
    "ignored_pairs": ".research/ghgsat_landfill_null/reference_catalog/target_reference_pairs.jsonl",
    "compact_json": "reports/acquisition/ghgsat_landfill_reference_catalog.json",
    "compact_markdown": "reports/acquisition/GHGSAT_LANDFILL_REFERENCE_CATALOG.md",
}
ACCESS_PROOF = {
    "metadata_search_endpoints_only": True,
    "item_detail_queried": False,
    "target_assets_accessed": False,
    "reference_assets_accessed": False,
    "asset_urls_retained": False,
    "item_links_retained": False,
    "preview_or_thumbnail_accessed": False,
    "raster_bytes_accessed": False,
    "ghgsat_assets_accessed": False,
    "model_artifacts_accessed": False,
    "protected_outcomes_accessed": False,
}

ReferenceCatalogAuditError = target.TargetCatalogAuditError
ResponseBodyLimitError = target.ResponseBodyLimitError
sha256_bytes = target.sha256_bytes
sha256_file = target.sha256_file
canonical_json_bytes = target.canonical_json_bytes
_finite_number = target._finite_number
_strict_string = target._strict_string
parse_item_utc = target.parse_item_utc
_rfc3339 = target._rfc3339
_rfc3339_precise = target._rfc3339_precise
geometry_covers_point = target.geometry_covers_point
_validate_bbox = target._validate_bbox
_atomic_bytes = target._atomic_bytes
_write_json = target._write_json
_append_jsonl = target._append_jsonl
_read_jsonl = target._read_jsonl


def _write_jsonl(
    path: Path, rows: Iterable[dict[str, object]], *, root: Path = ROOT,
) -> dict[str, object]:
    """Write deterministic derived JSONL with a receipt relative to this run root."""
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    _atomic_bytes(path, payload)
    return {
        "path": path.relative_to(root).as_posix(), "bytes": len(payload),
        "sha256": sha256_bytes(payload), "rows": payload.count(b"\n"),
    }


def historical_response_bytes(path: Path, *, maximum: int) -> int:
    """Validate every append-only byte/hash receipt and enforce the total cap."""
    total = 0
    for receipt in _read_jsonl(path):
        observed = receipt.get("response_bytes")
        digest = receipt.get("response_sha256")
        if (
            receipt.get("schema_version") != 1
            or type(observed) is not int or observed < 0
            or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(receipt.get("body_read_complete")) is not bool
            or type(receipt.get("accepted")) is not bool
        ):
            raise ReferenceCatalogAuditError("Response byte/hash receipt is malformed")
        total += observed
        if total > maximum:
            raise ReferenceCatalogAuditError("Historical STAC responses exceed total byte cap")
    return total


def _assert_protocol_contract(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ReferenceCatalogAuditError("Frozen protocol schema mismatch")
    if protocol.get("status") != "frozen_before_any_ghgsat_reference_catalog_query_or_asset_access":
        raise ReferenceCatalogAuditError("Frozen protocol status mismatch")
    if protocol.get("outputs") != REQUIRED_OUTPUTS:
        raise ReferenceCatalogAuditError("Frozen output paths mismatch")
    inputs = protocol.get("frozen_inputs")
    if not isinstance(inputs, dict) or tuple(inputs) != (
        "target_catalog_protocol", "target_catalog_report", "selected_target_pairs"
    ):
        raise ReferenceCatalogAuditError("Frozen input set/order mismatch")
    expected_inputs = {
        "target_catalog_protocol": (
            "configs/mars_ghgsat_target_catalog_protocol.json", 9357,
            "486e78bdb41b4aa9dfcb6bc6943eb80caf0cf8c7c5899bce1cdb6cc7a06967ff",
        ),
        "target_catalog_report": (
            "reports/acquisition/ghgsat_landfill_target_catalog.json", 3352,
            "ef8d99761c045a817f10a06568de11366aff3bfb7b55403eac4adc8cff1b5e6b",
        ),
        "selected_target_pairs": (
            ".research/ghgsat_landfill_null/target_catalog/selected_pairs.jsonl", 59230,
            "8c7942f0ac7bc07e250603e25ff23bb345e7bafbd86394e4ce0e42d41a33f6a8",
        ),
    }
    for name, (path, size, digest) in expected_inputs.items():
        spec = inputs.get(name, {})
        if spec.get("path") != path or spec.get("bytes") != size or spec.get("sha256") != digest:
            raise ReferenceCatalogAuditError(f"Frozen input contract mismatch: {name}")
    if inputs["target_catalog_report"].get("required_decision") != "PASS":
        raise ReferenceCatalogAuditError("Frozen target report decision contract mismatch")
    if inputs["selected_target_pairs"].get("rows") != 79:
        raise ReferenceCatalogAuditError("Frozen target-pair row contract mismatch")

    source = protocol.get("source_pair_contract", {})
    if source.get("exact_rows") != 79 or source.get("required_target_sensors") != list(CATALOG_ORDER):
        raise ReferenceCatalogAuditError("Frozen source-pair contract mismatch")
    if source.get("unique_identity") != list(SOURCE_IDENTITY_FIELDS):
        raise ReferenceCatalogAuditError("Frozen source identity contract mismatch")
    if source.get("all_78_target_item_ids_excluded_from_reference_selection") is not True:
        raise ReferenceCatalogAuditError("Frozen target exclusion contract mismatch")

    catalogs = protocol.get("catalogs")
    if not isinstance(catalogs, dict) or tuple(catalogs) != CATALOG_ORDER:
        raise ReferenceCatalogAuditError("Frozen catalog order mismatch")
    expected_catalogs = {
        "sentinel_2_l1c": (
            "https://stac.dataspace.copernicus.eu/v1/search", "sentinel-2-l1c",
            ["S2A_", "S2B_"],
        ),
        "landsat_8_9_level_1": (
            "https://landsatlook.usgs.gov/stac-server/search", "landsat-c2l1",
            ["LC08_", "LC09_"],
        ),
    }
    for sensor, (endpoint, collection, prefixes) in expected_catalogs.items():
        spec = catalogs.get(sensor, {})
        if (
            spec.get("endpoint") != endpoint or spec.get("collection") != collection
            or spec.get("allowed_item_prefixes") != prefixes
        ):
            raise ReferenceCatalogAuditError(f"Frozen catalog identity mismatch: {sensor}")
    landsat = catalogs["landsat_8_9_level_1"]
    if landsat.get("required_processing_level") != "L1TP" or landsat.get("required_collection_category") != "T1":
        raise ReferenceCatalogAuditError("Frozen Landsat level/tier mismatch")

    reference = protocol.get("reference_contract", {})
    expected_reference = {
        "direction": "prior only",
        "primary_window": {"minimum_gap_hours": 1, "maximum_lookback_days": 31},
        "seasonal_fallback": {
            "enabled_only_when_no_valid_primary_candidate_exists": True,
            "minimum_lookback_days": 334, "maximum_lookback_days": 410,
            "target_gap_days": 365,
        },
        "maximum_catalog_eo_cloud_cover_pct": 20,
        "same_collection": True, "same_granule": True,
        "excluded_items": "Every frozen target item ID for the same target sensor, globally across the 79 rows.",
        "selection_order_primary": [
            "smallest target-reference time gap", "smallest catalog eo:cloud_cover",
            "lexicographically smallest item ID",
        ],
        "selection_order_seasonal": [
            "smallest absolute difference from 365 days", "smallest catalog eo:cloud_cover",
            "lexicographically smallest item ID",
        ],
        "maximum_selected_per_source_sensor_pair": 1,
    }
    for name, expected in expected_reference.items():
        if reference.get(name) != expected:
            raise ReferenceCatalogAuditError(f"Frozen reference contract mismatch: {name}")

    query = protocol.get("query_contract", {})
    expected_query = {
        "one_primary_query_per_frozen_source_sensor_pair": True,
        "seasonal_query_only_after_valid_primary_set_is_empty": True,
        "stac_limit": 100, "fail_if_limit_reached": True,
        "maximum_response_bytes_each": 2 * 1024 * 1024,
        "maximum_response_bytes_total": 128 * 1024 * 1024,
        "minimum_request_interval_seconds": 0.25,
        "maximum_attempts_per_request": 5,
        "retryable_http_statuses": [429, 500, 502, 503, 504],
    }
    for name, expected in expected_query.items():
        if query.get(name) != expected:
            raise ReferenceCatalogAuditError(f"Frozen query contract mismatch: {name}")
    if protocol.get("gates") != {
        "minimum_selected_target_reference_pairs": 28,
        "minimum_distinct_source_observations_with_reference": 28,
        "minimum_distinct_sites_with_reference": 20,
        "minimum_novel_25km_components_with_reference": 20,
        "all_queries_and_retained_items_valid": True,
    }:
        raise ReferenceCatalogAuditError("Frozen gate contract mismatch")


def load_protocol(path: Path = EXPECTED_PROTOCOL) -> dict[str, Any]:
    if path.resolve() != EXPECTED_PROTOCOL.resolve():
        raise ReferenceCatalogAuditError("Only the exact committed reference-catalog protocol is permitted")
    if sha256_file(path) != EXPECTED_PROTOCOL_SHA256:
        raise ReferenceCatalogAuditError("Frozen reference-catalog protocol SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceCatalogAuditError("Frozen reference-catalog protocol is unreadable") from exc
    if not isinstance(value, dict):
        raise ReferenceCatalogAuditError("Frozen reference-catalog protocol must be an object")
    _assert_protocol_contract(value)
    return value


def validate_frozen_inputs(protocol: dict[str, Any], *, root: Path = ROOT) -> dict[str, object]:
    receipts: dict[str, object] = {}
    for name, spec in protocol["frozen_inputs"].items():
        path = root / spec["path"]
        if not path.is_file():
            raise ReferenceCatalogAuditError(f"Frozen input is missing: {name}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != spec["bytes"]:
            raise ReferenceCatalogAuditError(f"Frozen input byte count mismatch: {name}")
        if digest != spec["sha256"]:
            raise ReferenceCatalogAuditError(f"Frozen input hash mismatch: {name}")
        receipts[name] = {"path": spec["path"], "bytes": size, "sha256": digest}
    report_spec = protocol["frozen_inputs"]["target_catalog_report"]
    try:
        report = json.loads((root / report_spec["path"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceCatalogAuditError("Frozen target report is unreadable") from exc
    if report.get("decision") != report_spec["required_decision"]:
        raise ReferenceCatalogAuditError("Frozen target report decision mismatch")
    return receipts


def source_identity(row: dict[str, object]) -> dict[str, object]:
    return {name: row[name] for name in SOURCE_IDENTITY_FIELDS}


def target_identity(row: dict[str, object]) -> dict[str, object]:
    return {
        "target_sensor": row["target_sensor"], "target_item_id": row["target_item_id"],
        "target_datetime": row["target_datetime"],
    }


def _load_target_sources(*, root: Path) -> tuple[dict[str, Any], dict[tuple[object, ...], dict[str, object]]]:
    target_protocol = target.load_protocol(root / "configs/mars_ghgsat_target_catalog_protocol.json")
    target.validate_frozen_inputs(target_protocol, root=root)
    rows = target.load_source_rows(target_protocol, root=root)
    return target_protocol, {
        tuple(row[name] for name in target.SOURCE_IDENTITY_FIELDS): row for row in rows
    }


def load_source_pairs(protocol: dict[str, Any], *, root: Path = ROOT) -> list[dict[str, object]]:
    target_protocol, sources = _load_target_sources(root=root)
    path = root / protocol["frozen_inputs"]["selected_target_pairs"]["path"]
    rows: list[dict[str, object]] = []
    identities: set[tuple[object, ...]] = set()
    target_ids_by_sensor: dict[str, set[str]] = {sensor: set() for sensor in CATALOG_ORDER}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReferenceCatalogAuditError(f"Target pair {line_number} is invalid JSON") from exc
            if not isinstance(row, dict) or set(row) != SELECTED_TARGET_FIELDS:
                raise ReferenceCatalogAuditError(f"Target pair {line_number} field set mismatch")
            sensor = row.get("target_sensor")
            if sensor not in CATALOG_ORDER:
                raise ReferenceCatalogAuditError(f"Target pair {line_number} sensor mismatch")
            if _granule(str(row.get("target_item_id")), str(sensor)) is None:
                raise ReferenceCatalogAuditError(
                    f"Target pair {line_number} item ID cannot supply the frozen same-granule key"
                )
            identity = tuple(row.get(name) for name in SOURCE_IDENTITY_FIELDS)
            if identity in identities:
                raise ReferenceCatalogAuditError(f"Duplicate source-pair identity at row {line_number}")
            identities.add(identity)
            observation_key = tuple(row.get(name) for name in target.SOURCE_IDENTITY_FIELDS)
            source = sources.get(observation_key)
            if source is None:
                raise ReferenceCatalogAuditError(f"Target pair {line_number} has no frozen source observation")
            if row.get("component_id") != source["component_id"]:
                raise ReferenceCatalogAuditError(f"Target pair {line_number} component mismatch")
            properties: dict[str, object] = {
                "datetime": row.get("target_datetime"), "eo:cloud_cover": row.get("eo_cloud_cover"),
            }
            if sensor == "landsat_8_9_level_1":
                spec = target_protocol["catalogs"][sensor]
                properties.update({
                    "landsat:correction": spec["required_processing_level"],
                    "landsat:collection_category": spec["required_collection_category"],
                })
            rebuilt = target.validate_candidate({
                "type": "Feature", "id": row.get("target_item_id"),
                "collection": row.get("target_collection"), "geometry": row.get("geometry"),
                "bbox": row.get("bbox"), "properties": properties,
            }, source, str(sensor), target_protocol)
            if canonical_json_bytes(rebuilt) != canonical_json_bytes(row):
                raise ReferenceCatalogAuditError(f"Target pair {line_number} content mismatch")
            enriched = dict(row)
            enriched["representative_longitude"] = source["representative_longitude"]
            enriched["representative_latitude"] = source["representative_latitude"]
            rows.append(enriched)
            target_ids_by_sensor[str(sensor)].add(str(row["target_item_id"]))
    if len(rows) != protocol["source_pair_contract"]["exact_rows"]:
        raise ReferenceCatalogAuditError("Frozen source-pair row count mismatch")
    if len(set().union(*target_ids_by_sensor.values())) != 78:
        raise ReferenceCatalogAuditError("Frozen global target-item exclusion set is not exactly 78 IDs")
    return rows


def validation_plan(*, root: Path = ROOT) -> dict[str, object]:
    protocol = load_protocol()
    receipts = validate_frozen_inputs(protocol, root=root)
    rows = load_source_pairs(protocol, root=root)
    return {
        "mode": "validation_only",
        "protocol": {"path": PROTOCOL_RELATIVE_PATH, "sha256": EXPECTED_PROTOCOL_SHA256},
        "frozen_inputs": receipts,
        "source_pairs_validated": len(rows),
        "distinct_target_item_ids_validated": len({str(row["target_item_id"]) for row in rows}),
        "network_client_created": False,
        "network_executed": False,
        "catalog_response_opened": False,
        "reference_catalog_accessed": False,
        **{name: value for name, value in ACCESS_PROOF.items() if name != "metadata_search_endpoints_only"},
    }


def _granule(item_id: str, sensor: str) -> str | None:
    if sensor == "sentinel_2_l1c":
        match = re.fullmatch(
            r"S2[AB]_MSIL1C_\d{8}T\d{6}_N\d{4}_R\d{3}_T(\d{2}[A-Z]{3})_\d{8}T\d{6}",
            item_id,
        )
    elif sensor == "landsat_8_9_level_1":
        match = re.fullmatch(
            r"LC0[89]_L1TP_(\d{6})_\d{8}_\d{8}_02_T1", item_id
        )
    else:
        raise ReferenceCatalogAuditError("Unknown target sensor")
    return None if match is None else match.group(1)


def _query_window(row: dict[str, object], kind: str, protocol: dict[str, Any]) -> tuple[Any, Any]:
    target_time = parse_item_utc(row["target_datetime"])
    reference = protocol["reference_contract"]
    if kind == "primary":
        window = reference["primary_window"]
        start = target_time - timedelta(days=window["maximum_lookback_days"])
        end = target_time - timedelta(hours=window["minimum_gap_hours"])
    elif kind == "seasonal":
        window = reference["seasonal_fallback"]
        start = target_time - timedelta(days=window["maximum_lookback_days"])
        end = target_time - timedelta(days=window["minimum_lookback_days"])
    else:
        raise ReferenceCatalogAuditError("Unknown reference query kind")
    return start, end


def build_request(
    row: dict[str, object], kind: str, protocol: dict[str, Any]
) -> dict[str, object]:
    sensor = str(row["target_sensor"])
    if sensor not in CATALOG_ORDER:
        raise ReferenceCatalogAuditError("Unknown target sensor")
    start, end = _query_window(row, kind, protocol)
    catalog = protocol["catalogs"][sensor]
    required = list(dict.fromkeys(catalog["required_item_fields"]))
    body: dict[str, object] = {
        "collections": [catalog["collection"]],
        "intersects": {
            "type": "Point",
            "coordinates": [
                _finite_number(row["representative_longitude"], name="longitude"),
                _finite_number(row["representative_latitude"], name="latitude"),
            ],
        },
        "datetime": f"{_rfc3339_precise(start)}/{_rfc3339_precise(end)}",
        "limit": protocol["query_contract"]["stac_limit"],
        "fields": {"include": required, "exclude": ["assets", "links"]},
    }
    canonical = canonical_json_bytes(body)
    return {
        "query_kind": kind, "sensor": sensor, "endpoint": catalog["endpoint"],
        "source_identity": source_identity(row), "target_identity": target_identity(row),
        "body": body, "canonical_request": canonical.decode("utf-8"),
        "canonical_request_sha256": sha256_bytes(canonical),
    }


def _candidate_identity_fields(row: dict[str, object]) -> dict[str, object]:
    return {
        **source_identity(row), "component_id": row["component_id"],
        "target_item_id": row["target_item_id"],
        "target_datetime": row["target_datetime"],
    }


def _sanitized_geometry(value: object) -> dict[str, object]:
    """Retain only GeoJSON type/coordinates with finite numeric leaves."""
    if not isinstance(value, dict):
        raise ReferenceCatalogAuditError("Retainable item geometry is malformed")

    def sanitize(node: object) -> object:
        if isinstance(node, list):
            return [sanitize(child) for child in node]
        return _finite_number(node, name="geometry coordinate")

    return {
        "type": _strict_string(value.get("type"), name="geometry type"),
        "coordinates": sanitize(value.get("coordinates")),
    }


def validate_candidate(
    item: dict[str, object], row: dict[str, object], kind: str,
    protocol: dict[str, Any], *, excluded_target_ids: set[str],
) -> dict[str, object] | None:
    sensor = str(row["target_sensor"])
    catalog = protocol["catalogs"][sensor]
    item_id_value = item.get("id")
    if not isinstance(item_id_value, str) or not item_id_value or item_id_value.strip() != item_id_value:
        raise ReferenceCatalogAuditError("STAC item ID is malformed")
    item_id = item_id_value
    prefixes = catalog["allowed_item_prefixes"]
    if not any(item_id.startswith(prefix) for prefix in prefixes):
        return None
    reference_granule = _granule(item_id, sensor)
    target_granule = _granule(str(row["target_item_id"]), sensor)
    if reference_granule is None or target_granule is None:
        return None
    if item.get("collection") != catalog["collection"] or reference_granule != target_granule:
        return None
    if item_id in excluded_target_ids:
        return None
    if item.get("type") != "Feature":
        raise ReferenceCatalogAuditError("Retainable STAC item is not a Feature")
    properties = item.get("properties")
    if not isinstance(properties, dict):
        raise ReferenceCatalogAuditError("Retainable item properties are malformed")
    item_time = parse_item_utc(properties.get("datetime"))
    start, end = _query_window(row, kind, protocol)
    if item_time < start or item_time > end:
        return None
    cloud = _finite_number(properties.get("eo:cloud_cover"), name="eo:cloud_cover")
    if not 0 <= cloud <= 100:
        raise ReferenceCatalogAuditError("eo:cloud_cover is outside [0,100]")
    if cloud > protocol["reference_contract"]["maximum_catalog_eo_cloud_cover_pct"]:
        return None
    if sensor == "landsat_8_9_level_1":
        if properties.get("landsat:correction") != catalog["required_processing_level"]:
            return None
        if properties.get("landsat:collection_category") != catalog["required_collection_category"]:
            return None
    bbox = _validate_bbox(item.get("bbox"))
    longitude = _finite_number(row["representative_longitude"], name="source longitude")
    latitude = _finite_number(row["representative_latitude"], name="source latitude")
    geometry = _sanitized_geometry(item.get("geometry"))
    if not geometry_covers_point(geometry, longitude, latitude):
        return None
    target_time = parse_item_utc(row["target_datetime"])
    gap_seconds = (target_time - item_time).total_seconds()
    if not math.isfinite(gap_seconds) or gap_seconds <= 0:
        raise ReferenceCatalogAuditError("Reference is not strictly prior")
    return {
        **_candidate_identity_fields(row), "selection_window": kind,
        "reference_item_id": item_id, "reference_collection": item["collection"],
        "reference_datetime": _rfc3339_precise(item_time),
        "target_reference_gap_seconds": gap_seconds,
        "target_reference_gap_days": gap_seconds / 86400.0,
        "seasonal_distance_from_365_days": abs(gap_seconds / 86400.0 - 365.0),
        "eo_cloud_cover": cloud, "granule_id": reference_granule,
        "geometry": geometry, "bbox": bbox,
    }


def parse_feature_collection(
    payload: object, row: dict[str, object], kind: str, protocol: dict[str, Any],
    *, excluded_target_ids: set[str],
) -> list[dict[str, object]]:
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ReferenceCatalogAuditError("Response is not a STAC FeatureCollection")
    features = payload.get("features")
    if not isinstance(features, list):
        raise ReferenceCatalogAuditError("FeatureCollection features must be a list")
    if len(features) >= protocol["query_contract"]["stac_limit"]:
        raise ReferenceCatalogAuditError("STAC response reached the frozen 100-item ceiling")
    candidates: list[dict[str, object]] = []
    for item in features:
        if not isinstance(item, dict):
            raise ReferenceCatalogAuditError("STAC response contains a non-item")
        candidate = validate_candidate(
            item, row, kind, protocol, excluded_target_ids=excluded_target_ids
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def select_candidate(candidates: list[dict[str, object]], kind: str) -> dict[str, object] | None:
    if not candidates:
        return None
    if kind == "primary":
        rank = lambda item: (
            float(item["target_reference_gap_seconds"]),
            float(item["eo_cloud_cover"]), str(item["reference_item_id"]),
        )
    elif kind == "seasonal":
        rank = lambda item: (
            float(item["seasonal_distance_from_365_days"]),
            float(item["eo_cloud_cover"]), str(item["reference_item_id"]),
        )
    else:
        raise ReferenceCatalogAuditError("Unknown reference selection kind")
    return min(candidates, key=rank)


def response_checkpoint_sha256(receipt: dict[str, object]) -> str:
    fields = (
        "schema_version", "status", "accepted", "body_read_complete",
        "canonical_request_sha256", "endpoint", "sensor", "query_kind",
        "source_identity", "target_identity", "response_bytes", "response_sha256",
        "candidate_count", "parsed_candidates_sha256",
    )
    return sha256_bytes(canonical_json_bytes({name: receipt.get(name) for name in fields}))


def resume_key(request_record: dict[str, object]) -> tuple[str, str, str, str, bytes, bytes]:
    return (
        str(request_record["canonical_request_sha256"]), str(request_record["endpoint"]),
        str(request_record["sensor"]), str(request_record["query_kind"]),
        canonical_json_bytes(request_record["source_identity"]),
        canonical_json_bytes(request_record["target_identity"]),
    )


def _receipt_key(receipt: dict[str, object]) -> tuple[str, str, str, str, bytes, bytes]:
    required = ("canonical_request_sha256", "endpoint", "sensor", "query_kind")
    if not all(isinstance(receipt.get(name), str) for name in required):
        raise ReferenceCatalogAuditError("Resume key is malformed")
    return (
        str(receipt["canonical_request_sha256"]), str(receipt["endpoint"]),
        str(receipt["sensor"]), str(receipt["query_kind"]),
        canonical_json_bytes(receipt.get("source_identity")),
        canonical_json_bytes(receipt.get("target_identity")),
    )


def load_resume_receipts(
    response_log: Path, *, request_log: Path | None = None,
) -> dict[tuple[str, str, str, str, bytes, bytes], dict[str, object]]:
    request_keys: set[tuple[str, str, str, str, bytes, bytes]] | None = None
    if request_log is not None:
        request_keys = set()
        for receipt in _read_jsonl(request_log):
            canonical = receipt.get("canonical_request")
            if (
                receipt.get("schema_version") != 1 or not isinstance(canonical, str)
                or sha256_bytes(canonical.encode()) != receipt.get("canonical_request_sha256")
            ):
                raise ReferenceCatalogAuditError("Canonical request receipt is malformed")
            request_keys.add(_receipt_key(receipt))
    accepted: dict[tuple[str, str, str, str, bytes, bytes], dict[str, object]] = {}
    for receipt in _read_jsonl(response_log):
        if receipt.get("accepted") is not True:
            continue
        observed = receipt.get("response_bytes")
        digest = receipt.get("response_sha256")
        if (
            receipt.get("schema_version") != 1 or receipt.get("status") != 200
            or receipt.get("body_read_complete") is not True
            or type(observed) is not int or observed < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
        ):
            raise ReferenceCatalogAuditError("Accepted response receipt is malformed")
        key = _receipt_key(receipt)
        if key in accepted:
            raise ReferenceCatalogAuditError("Accepted response receipts are duplicate")
        candidates = receipt.get("parsed_candidates")
        if not isinstance(candidates, list):
            raise ReferenceCatalogAuditError("Accepted receipt has no parsed candidates")
        if receipt.get("candidate_count") != len(candidates):
            raise ReferenceCatalogAuditError("Accepted candidate-count receipt mismatch")
        if receipt.get("parsed_candidates_sha256") != sha256_bytes(canonical_json_bytes(candidates)):
            raise ReferenceCatalogAuditError("Accepted parsed-candidate receipt mismatch")
        if receipt.get("checkpoint_sha256") != response_checkpoint_sha256(receipt):
            raise ReferenceCatalogAuditError("Accepted response checkpoint mismatch")
        if request_keys is not None and key not in request_keys:
            raise ReferenceCatalogAuditError("Accepted response has no matching request receipt")
        accepted[key] = receipt
    return accepted


@dataclass
class STACReferenceAuditClient(target.STACAuditClient):
    def execute(
        self, request_record: dict[str, object], row: dict[str, object],
        *, excluded_target_ids: set[str],
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        endpoint = str(request_record["endpoint"])
        sensor = str(request_record["sensor"])
        kind = str(request_record["query_kind"])
        if endpoint != self.protocol["catalogs"][sensor]["endpoint"]:
            raise ReferenceCatalogAuditError("Request endpoint does not match frozen catalog")
        maximum_attempts = self.protocol["query_contract"]["maximum_attempts_per_request"]
        retryable = set(self.protocol["query_contract"]["retryable_http_statuses"])
        for attempt in range(1, maximum_attempts + 1):
            self._pace()
            self.http_attempts += 1
            request_receipt = {
                "schema_version": 1, "attempt": attempt, "endpoint": endpoint,
                "sensor": sensor, "query_kind": kind,
                "source_identity": request_record["source_identity"],
                "target_identity": request_record["target_identity"],
                "canonical_request": request_record["canonical_request"],
                "canonical_request_sha256": request_record["canonical_request_sha256"],
            }
            _append_jsonl(self.request_log, request_receipt)
            response = self.session.post(
                endpoint, data=str(request_record["canonical_request"]).encode(),
                allow_redirects=False, stream=True, timeout=(15, 90),
                headers={
                    "Accept": "application/geo+json, application/json",
                    "Accept-Encoding": "identity", "Content-Type": "application/json",
                    "User-Agent": "ERSRR-GHGSat-reference-catalog-audit/1.0",
                },
            )
            try:
                status = int(response.status_code)
                self.status_counts[status] += 1
                if getattr(response, "history", []):
                    raise ReferenceCatalogAuditError("Redirected STAC response rejected")
                if str(getattr(response, "url", endpoint)) != endpoint:
                    raise ReferenceCatalogAuditError("STAC response endpoint identity mismatch")
                try:
                    body = self._read_body(response)
                except ResponseBodyLimitError as exc:
                    _append_jsonl(self.response_log, {
                        **{key: value for key, value in request_receipt.items() if key != "canonical_request"},
                        "status": status, "response_bytes": len(exc.partial_body),
                        "response_sha256": sha256_bytes(exc.partial_body),
                        "body_read_complete": False, "accepted": False,
                    })
                    raise
                receipt: dict[str, object] = {
                    **{key: value for key, value in request_receipt.items() if key != "canonical_request"},
                    "status": status, "response_bytes": len(body),
                    "response_sha256": sha256_bytes(body),
                    "body_read_complete": True, "accepted": False,
                }
                if status != 200:
                    _append_jsonl(self.response_log, receipt)
                    if status not in retryable or attempt == maximum_attempts:
                        raise ReferenceCatalogAuditError(f"STAC HTTP status rejected: {status}")
                    self.sleep(min(2 ** (attempt - 1), 8))
                    continue
                content_type = str(response.headers.get("Content-Type", "")).lower()
                if "json" not in content_type or "html" in content_type:
                    _append_jsonl(self.response_log, receipt)
                    raise ReferenceCatalogAuditError("STAC response is not JSON metadata")
                try:
                    payload = json.loads(body)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    _append_jsonl(self.response_log, receipt)
                    raise ReferenceCatalogAuditError("STAC response is invalid JSON") from exc
                try:
                    candidates = parse_feature_collection(
                        payload, row, kind, self.protocol,
                        excluded_target_ids=excluded_target_ids,
                    )
                except ReferenceCatalogAuditError:
                    _append_jsonl(self.response_log, receipt)
                    raise
                receipt.update({
                    "accepted": True, "candidate_count": len(candidates),
                    "parsed_candidates": candidates,
                    "parsed_candidates_sha256": sha256_bytes(canonical_json_bytes(candidates)),
                })
                receipt["checkpoint_sha256"] = response_checkpoint_sha256(receipt)
                _append_jsonl(self.response_log, receipt)
                return candidates, receipt
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        raise AssertionError("unreachable retry loop")


def _resume_candidates(
    receipt: dict[str, object], request_record: dict[str, object], row: dict[str, object],
    protocol: dict[str, Any], *, excluded_target_ids: set[str],
) -> list[dict[str, object]]:
    if (
        receipt.get("schema_version") != 1 or receipt.get("accepted") is not True
        or receipt.get("status") != 200 or receipt.get("body_read_complete") is not True
    ):
        raise ReferenceCatalogAuditError("Resume receipt is not accepted HTTP 200")
    observed = receipt.get("response_bytes")
    digest = receipt.get("response_sha256")
    if (
        type(observed) is not int or observed < 0
        or observed > protocol["query_contract"]["maximum_response_bytes_each"]
        or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ReferenceCatalogAuditError("Resume response byte/hash receipt mismatch")
    if _receipt_key(receipt) != resume_key(request_record):
        raise ReferenceCatalogAuditError("Resume request/source/target identity mismatch")
    if receipt.get("checkpoint_sha256") != response_checkpoint_sha256(receipt):
        raise ReferenceCatalogAuditError("Resume checkpoint mismatch")
    raw = receipt.get("parsed_candidates")
    if not isinstance(raw, list) or receipt.get("parsed_candidates_sha256") != sha256_bytes(canonical_json_bytes(raw)):
        raise ReferenceCatalogAuditError("Resume parsed-candidate hash mismatch")
    kind = str(request_record["query_kind"])
    rebuilt: list[dict[str, object]] = []
    for candidate in raw:
        if not isinstance(candidate, dict):
            raise ReferenceCatalogAuditError("Resume candidate is malformed")
        properties: dict[str, object] = {
            "datetime": candidate.get("reference_datetime"),
            "eo:cloud_cover": candidate.get("eo_cloud_cover"),
        }
        sensor = str(row["target_sensor"])
        if sensor == "landsat_8_9_level_1":
            catalog = protocol["catalogs"][sensor]
            properties.update({
                "landsat:correction": catalog["required_processing_level"],
                "landsat:collection_category": catalog["required_collection_category"],
            })
        value = validate_candidate({
            "type": "Feature", "id": candidate.get("reference_item_id"),
            "collection": candidate.get("reference_collection"),
            "geometry": candidate.get("geometry"), "bbox": candidate.get("bbox"),
            "properties": properties,
        }, row, kind, protocol, excluded_target_ids=excluded_target_ids)
        if value is None or canonical_json_bytes(value) != canonical_json_bytes(candidate):
            raise ReferenceCatalogAuditError("Resume candidate content mismatch")
        rebuilt.append(value)
    if len(rebuilt) != receipt.get("candidate_count"):
        raise ReferenceCatalogAuditError("Resume candidate-count mismatch")
    return rebuilt


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None, "median": None}
    return {
        "count": len(values), "minimum": min(values), "maximum": max(values),
        "mean": statistics.fmean(values), "median": statistics.median(values),
    }


def evaluate_gates(
    selected: list[dict[str, object]], protocol: dict[str, Any], *, all_valid: bool,
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    observations = {tuple(row[name] for name in OBSERVATION_IDENTITY_FIELDS) for row in selected}
    sites = {str(row["site_ID"]) for row in selected}
    components = {str(row["component_id"]) for row in selected}
    observed: dict[str, object] = {
        "minimum_selected_target_reference_pairs": len(selected),
        "minimum_distinct_source_observations_with_reference": len(observations),
        "minimum_distinct_sites_with_reference": len(sites),
        "minimum_novel_25km_components_with_reference": len(components),
        "all_queries_and_retained_items_valid": all_valid,
    }
    gates = {
        name: {
            "observed": observed[name], "required": required,
            "pass": observed[name] is True if required is True else int(observed[name]) >= required,
        }
        for name, required in protocol["gates"].items()
    }
    return gates, {
        "selected_target_reference_pairs": len(selected),
        "distinct_source_observations": len(observations), "distinct_sites": len(sites),
        "distinct_components": len(components),
        "distinct_reference_item_ids": len({str(row["reference_item_id"]) for row in selected}),
    }


def build_report(
    protocol: dict[str, Any], *, candidates: list[dict[str, object]],
    selected: list[dict[str, object]], primary_queries: int, seasonal_queries: int,
    resumed_queries: int, client: STACReferenceAuditClient,
    output_receipts: dict[str, object], all_valid: bool = True,
) -> dict[str, object]:
    gates, counts = evaluate_gates(selected, protocol, all_valid=all_valid)
    candidate_counts = Counter(str(row["selection_window"]) for row in candidates)
    selection_counts = Counter(str(row["selection_window"]) for row in selected)
    candidate_sensor_counts = Counter(str(row["target_sensor"]) for row in candidates)
    selection_sensor_counts = Counter(str(row["target_sensor"]) for row in selected)
    return {
        "schema_version": 1,
        "decision": "PASS" if all(bool(gate["pass"]) for gate in gates.values()) else "FAIL",
        "claim_boundary": protocol["claim_boundary"],
        "queries": {
            "primary_expected": 79, "primary_logical_queries": primary_queries,
            "seasonal_logical_queries": seasonal_queries,
            "total_logical_queries": primary_queries + seasonal_queries,
            "resumed_queries": resumed_queries,
            "network_queries": primary_queries + seasonal_queries - resumed_queries,
            "http_attempts": client.http_attempts,
        },
        "http_and_byte_receipts": {
            "network_bytes_observed_this_execution": client.total_network_bytes,
            "historical_response_bytes_before_execution": client.historical_network_bytes,
            "cumulative_response_bytes": client.historical_network_bytes + client.total_network_bytes,
            "status_counts": {str(key): value for key, value in sorted(client.status_counts.items())},
            "maximum_response_bytes_each": protocol["query_contract"]["maximum_response_bytes_each"],
            "maximum_response_bytes_total": protocol["query_contract"]["maximum_response_bytes_total"],
            "requests_append_only": protocol["outputs"]["ignored_requests"],
            "responses_append_only": protocol["outputs"]["ignored_responses"],
        },
        "candidate_counts_by_window": {kind: candidate_counts[kind] for kind in ("primary", "seasonal")},
        "selection_counts_by_window": {kind: selection_counts[kind] for kind in ("primary", "seasonal")},
        "candidate_counts_by_sensor": {
            sensor: candidate_sensor_counts[sensor] for sensor in CATALOG_ORDER
        },
        "selection_counts_by_sensor": {
            sensor: selection_sensor_counts[sensor] for sensor in CATALOG_ORDER
        },
        "target_reference_gap_days_summary": _summary([
            float(row["target_reference_gap_days"]) for row in selected
        ]),
        "cloud_cover_summary": _summary([float(row["eo_cloud_cover"]) for row in selected]),
        "distinct_counts": counts, "gates": gates, "output_receipts": output_receipts,
        "access_boundary_proof": dict(ACCESS_PROOF), "frozen_inputs_mutated": False,
    }


def build_failure_report(
    protocol: dict[str, Any], error: Exception, *, client: STACReferenceAuditClient | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1, "decision": "FAIL", "error": str(error),
        "all_queries_and_retained_items_valid": False,
        "queries": {
            "primary_expected": 79, "primary_logical_queries": None,
            "seasonal_logical_queries": None, "total_logical_queries": None,
            "resumed_queries": None, "network_queries": None,
            "http_attempts": 0 if client is None else client.http_attempts,
        },
        "candidate_counts_by_window": {"primary": None, "seasonal": None},
        "selection_counts_by_window": {"primary": None, "seasonal": None},
        "candidate_counts_by_sensor": {sensor: None for sensor in CATALOG_ORDER},
        "selection_counts_by_sensor": {sensor: None for sensor in CATALOG_ORDER},
        "distinct_counts": {
            "selected_target_reference_pairs": None, "distinct_source_observations": None,
            "distinct_sites": None, "distinct_components": None,
            "distinct_reference_item_ids": None,
        },
        "gates": {
            name: {"observed": False if required is True else None, "required": required, "pass": False}
            for name, required in protocol["gates"].items()
        },
        "http_and_byte_receipts": {
            "network_bytes_observed_this_execution": 0 if client is None else client.total_network_bytes,
            "historical_response_bytes_before_execution": 0 if client is None else client.historical_network_bytes,
            "cumulative_response_bytes": 0 if client is None else client.historical_network_bytes + client.total_network_bytes,
            "status_counts": {} if client is None else {
                str(key): value for key, value in sorted(client.status_counts.items())
            },
        },
        "access_boundary_proof": dict(ACCESS_PROOF), "frozen_inputs_mutated": False,
        "stale_candidate_and_pair_outputs_removed": True,
        "claim_boundary": protocol["claim_boundary"],
    }


def _markdown(report: dict[str, object]) -> str:
    lines = ["# GHGSat reference-catalog audit", "", f"**Decision: {report['decision']}**", ""]
    if "error" in report:
        lines += [f"Failure: `{report['error']}`", ""]
    for heading, key in (
        ("Query receipts", "queries"), ("HTTP and byte receipts", "http_and_byte_receipts"),
        ("Candidate counts", "candidate_counts_by_window"),
        ("Selection counts", "selection_counts_by_window"), ("Distinct counts", "distinct_counts"),
        ("Candidate counts by sensor", "candidate_counts_by_sensor"),
        ("Selection counts by sensor", "selection_counts_by_sensor"),
    ):
        values = report.get(key)
        if isinstance(values, dict):
            lines += [f"## {heading}", ""] + [f"- {name}: {value}" for name, value in values.items()] + [""]
    lines += ["## Gates", ""]
    for name, gate in dict(report.get("gates", {})).items():
        lines.append(
            f"- {name}: {'PASS' if gate['pass'] else 'FAIL'} "
            f"(observed {gate['observed']!r}; required {gate['required']!r})"
        )
    lines += ["", "## Access-boundary proof", ""]
    lines += [f"- {name}: {value}" for name, value in dict(report["access_boundary_proof"]).items()]
    lines += ["- Frozen inputs mutated: False", "", f"Claim boundary: {report['claim_boundary']}", ""]
    return "\n".join(lines)


def _write_reports(protocol: dict[str, Any], report: dict[str, object], *, root: Path = ROOT) -> None:
    _write_json(root / protocol["outputs"]["compact_json"], report)
    _atomic_bytes(
        root / protocol["outputs"]["compact_markdown"], _markdown(report).encode("utf-8")
    )


def _remove_stale_outputs(protocol: dict[str, Any], *, root: Path = ROOT) -> None:
    for name in ("ignored_candidates", "ignored_pairs"):
        (root / protocol["outputs"][name]).unlink(missing_ok=True)


def execute_network_audit(
    *, root: Path = ROOT, session: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    protocol = load_protocol()
    client: STACReferenceAuditClient | None = None
    try:
        validate_frozen_inputs(protocol, root=root)
        rows = load_source_pairs(protocol, root=root)
        excluded_by_sensor = {
            sensor: {str(row["target_item_id"]) for row in rows if row["target_sensor"] == sensor}
            for sensor in CATALOG_ORDER
        }
        outputs = protocol["outputs"]
        request_log = root / outputs["ignored_requests"]
        response_log = root / outputs["ignored_responses"]
        historical = historical_response_bytes(
            response_log, maximum=protocol["query_contract"]["maximum_response_bytes_total"]
        )
        accepted = load_resume_receipts(response_log, request_log=request_log)
        client = STACReferenceAuditClient(
            session=session if session is not None else target.requests.Session(),
            protocol=protocol, request_log=request_log, response_log=response_log,
            sleep=sleep, monotonic=monotonic, historical_network_bytes=historical,
        )
        candidates: list[dict[str, object]] = []
        selected: list[dict[str, object]] = []
        primary_queries = 0
        seasonal_queries = 0
        resumed_queries = 0

        def run(row: dict[str, object], kind: str) -> list[dict[str, object]]:
            nonlocal resumed_queries
            request = build_request(row, kind, protocol)
            receipt = accepted.get(resume_key(request))
            exclusions = excluded_by_sensor[str(row["target_sensor"])]
            if receipt is not None:
                resumed_queries += 1
                return _resume_candidates(
                    receipt, request, row, protocol, excluded_target_ids=exclusions
                )
            fresh, _ = client.execute(request, row, excluded_target_ids=exclusions)
            return fresh

        for row in rows:
            primary_queries += 1
            primary = run(row, "primary")
            candidates.extend(primary)
            choice = select_candidate(primary, "primary")
            if choice is None:
                seasonal_queries += 1
                seasonal = run(row, "seasonal")
                candidates.extend(seasonal)
                choice = select_candidate(seasonal, "seasonal")
            if choice is not None:
                selected.append(choice)
        if primary_queries != 79:
            raise ReferenceCatalogAuditError("Did not execute exactly 79 primary logical searches")
        if seasonal_queries != sum(
            1 for row in rows
            if not any(
                tuple(candidate[name] for name in SOURCE_IDENTITY_FIELDS)
                == tuple(row[name] for name in SOURCE_IDENTITY_FIELDS)
                and candidate["selection_window"] == "primary"
                for candidate in candidates
            )
        ):
            raise ReferenceCatalogAuditError("Seasonal query gating mismatch")
        candidate_receipt = _write_jsonl(
            root / outputs["ignored_candidates"], candidates, root=root
        )
        pair_receipt = _write_jsonl(root / outputs["ignored_pairs"], selected, root=root)
        report = build_report(
            protocol, candidates=candidates, selected=selected,
            primary_queries=primary_queries, seasonal_queries=seasonal_queries,
            resumed_queries=resumed_queries, client=client,
            output_receipts={"candidates": candidate_receipt, "target_reference_pairs": pair_receipt},
        )
        _write_reports(protocol, report, root=root)
        return report
    except Exception as exc:
        _remove_stale_outputs(protocol, root=root)
        failure = build_failure_report(protocol, exc, client=client)
        _write_reports(protocol, failure, root=root)
        if isinstance(exc, ReferenceCatalogAuditError):
            raise
        raise ReferenceCatalogAuditError(str(exc)) from exc
    finally:
        if session is None and client is not None:
            close = getattr(client.session, "close", None)
            if callable(close):
                close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-network", action="store_true",
        help="execute all 79 primary searches plus only required seasonal fallbacks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute_network_audit() if args.execute_network else validation_plan()
    except ReferenceCatalogAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("decision", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
