"""Audit the exact frozen GHGSat target catalogs, with a safe offline default.

Without ``--execute-network`` this command validates only the hash-bound protocol
and its three frozen local inputs.  The explicit network mode issues every one
of the 176 x two frozen metadata-only STAC searches; it has no subset, widened
window, alternate endpoint, asset fetch, or fallback path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_RELATIVE_PATH = "configs/mars_ghgsat_target_catalog_protocol.json"
EXPECTED_PROTOCOL = ROOT / PROTOCOL_RELATIVE_PATH
EXPECTED_PROTOCOL_SHA256 = "486e78bdb41b4aa9dfcb6bc6943eb80caf0cf8c7c5899bce1cdb6cc7a06967ff"
SOURCE_IDENTITY_FIELDS = ("site_ID", "obs_ID", "date", "sat_ID")
EXPECTED_SOURCE_FIELDS = {
    "site_ID", "obs_ID", "date", "sat_ID", "year", "observation_state",
    "plume_row_count", "source_data_rows", "representative_latitude",
    "representative_longitude", "representative_positive_data_row",
    "positive_coordinate_span_km", "nearest_official_mars_test_km",
    "nearest_official_mars_test_location", "nearest_prior_negative_km",
    "nearest_prior_negative_id", "excluded_by_official_mars_test_radius",
    "excluded_by_prior_negative_radius", "passes_protected_distance_filter",
    "eligible_for_target_catalog", "component_id",
}
CATALOG_ORDER = ("sentinel_2_l1c", "landsat_8_9_level_1")
REQUIRED_OUTPUTS = {
    "ignored_root": ".research/ghgsat_landfill_null/target_catalog",
    "ignored_requests": ".research/ghgsat_landfill_null/target_catalog/requests.jsonl",
    "ignored_responses": ".research/ghgsat_landfill_null/target_catalog/responses.jsonl",
    "ignored_candidates": ".research/ghgsat_landfill_null/target_catalog/candidates.jsonl",
    "ignored_selected_pairs": ".research/ghgsat_landfill_null/target_catalog/selected_pairs.jsonl",
    "compact_json": "reports/acquisition/ghgsat_landfill_target_catalog.json",
    "compact_markdown": "reports/acquisition/GHGSAT_LANDFILL_TARGET_CATALOG.md",
}
ACCESS_PROOF = {
    "target_assets_accessed": False,
    "asset_urls_retained": False,
    "preview_or_thumbnail_accessed": False,
    "reference_catalog_queried": False,
    "reference_assets_accessed": False,
    "ghgsat_assets_accessed": False,
    "protected_outcomes_accessed": False,
    "score_caches_accessed": False,
    "model_artifacts_accessed": False,
    "model_imports_used": False,
}


class TargetCatalogAuditError(RuntimeError):
    """The frozen target-catalog contract was violated."""


class ResponseBodyLimitError(TargetCatalogAuditError):
    """A streamed response crossed a cap; ``partial_body`` is non-refundable."""

    def __init__(self, message: str, partial_body: bytes) -> None:
        super().__init__(message)
        self.partial_body = partial_body


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TargetCatalogAuditError("Value cannot be canonically encoded as JSON") from exc


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(payload)
    partial.replace(path)


def _write_json(path: Path, value: object) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> dict[str, object]:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    _atomic_bytes(path, payload)
    return {
        "path": path.relative_to(ROOT).as_posix(), "bytes": len(payload),
        "sha256": sha256_bytes(payload), "rows": payload.count(b"\n"),
    }


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(row) + b"\n")
        handle.flush()


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TargetCatalogAuditError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TargetCatalogAuditError(f"{name} must be a finite number")
    return result


def _strict_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TargetCatalogAuditError(f"{name} must be a non-empty trimmed string")
    return value


def parse_source_utc(value: object) -> datetime:
    text = _strict_string(value, name="source date")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TargetCatalogAuditError("source date must be an exact UTC timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed.utcoffset() != timedelta(0) or parsed.microsecond:
        raise TargetCatalogAuditError("source date must be an exact UTC timestamp")
    return parsed.astimezone(timezone.utc)


def parse_item_utc(value: object) -> datetime:
    text = _strict_string(value, name="item datetime")
    if not (text.endswith("Z") or text.endswith("+00:00")):
        raise TargetCatalogAuditError("item datetime must explicitly be UTC")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TargetCatalogAuditError("item datetime must be finite UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise TargetCatalogAuditError("item datetime must be finite UTC")
    return parsed.astimezone(timezone.utc)


def _rfc3339(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _rfc3339_precise(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="auto").replace("+00:00", "Z")


def _assert_protocol_contract(protocol: dict[str, Any]) -> None:
    if protocol.get("status") != "frozen_before_any_sentinel2_or_landsat_target_catalog_query":
        raise TargetCatalogAuditError("Frozen protocol status mismatch")
    if protocol.get("outputs") != REQUIRED_OUTPUTS:
        raise TargetCatalogAuditError("Frozen output paths mismatch")
    query = protocol.get("query_contract")
    expected_query = {
        "maximum_absolute_time_offset_seconds": 3600,
        "stac_limit": 100,
        "fail_if_limit_reached": True,
        "maximum_response_bytes_each": 2 * 1024 * 1024,
        "maximum_response_bytes_total": 256 * 1024 * 1024,
        "minimum_request_interval_seconds": 0.25,
        "maximum_attempts_per_request": 5,
        "retryable_http_statuses": [429, 500, 502, 503, 504],
    }
    if not isinstance(query, dict):
        raise TargetCatalogAuditError("Frozen query contract is missing")
    for key, expected in expected_query.items():
        if query.get(key) != expected:
            raise TargetCatalogAuditError(f"Frozen query contract mismatch: {key}")
    catalogs = protocol.get("catalogs")
    if not isinstance(catalogs, dict) or tuple(catalogs) != CATALOG_ORDER:
        raise TargetCatalogAuditError("Frozen catalog order mismatch")
    exact_catalogs = {
        "sentinel_2_l1c": (
            "https://stac.dataspace.copernicus.eu/v1/search", "sentinel-2-l1c"
        ),
        "landsat_8_9_level_1": (
            "https://landsatlook.usgs.gov/stac-server/search", "landsat-c2l1"
        ),
    }
    for sensor, (endpoint, collection) in exact_catalogs.items():
        spec = catalogs.get(sensor, {})
        if spec.get("endpoint") != endpoint or spec.get("collection") != collection:
            raise TargetCatalogAuditError(f"Frozen catalog identity mismatch: {sensor}")
        if spec.get("assets_requested") is not False:
            raise TargetCatalogAuditError(f"Assets are not forbidden for {sensor}")
    source = protocol.get("source_observation_contract", {})
    if set(source.get("required_fields", [])) != {
        "site_ID", "obs_ID", "date", "sat_ID", "observation_state",
        "representative_latitude", "representative_longitude", "component_id",
        "passes_protected_distance_filter", "eligible_for_target_catalog",
    }:
        raise TargetCatalogAuditError("Frozen source field contract mismatch")
    required = source.get("required_values", {})
    if required != {
        "sat_ID": [1, 2], "observation_state": "null",
        "passes_protected_distance_filter": True,
        "eligible_for_target_catalog": False,
    }:
        raise TargetCatalogAuditError("Frozen source value contract mismatch")
    if protocol.get("gates") != {
        "minimum_selected_source_sensor_pairs": 28,
        "minimum_distinct_source_observations_with_pair": 28,
        "minimum_distinct_sites_with_pair": 20,
        "minimum_novel_25km_components_with_pair": 20,
        "minimum_distinct_target_item_ids": 20,
        "all_queries_and_retained_items_valid": True,
    }:
        raise TargetCatalogAuditError("Frozen gate contract mismatch")


def load_protocol(path: Path = EXPECTED_PROTOCOL) -> dict[str, Any]:
    if path.resolve() != EXPECTED_PROTOCOL.resolve():
        raise TargetCatalogAuditError("Only the exact committed target-catalog protocol is permitted")
    if sha256_file(path) != EXPECTED_PROTOCOL_SHA256:
        raise TargetCatalogAuditError("Frozen target-catalog protocol SHA-256 mismatch")
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetCatalogAuditError("Frozen target-catalog protocol is unreadable") from exc
    if not isinstance(protocol, dict):
        raise TargetCatalogAuditError("Frozen target-catalog protocol must be an object")
    _assert_protocol_contract(protocol)
    return protocol


def validate_frozen_inputs(protocol: dict[str, Any], *, root: Path = ROOT) -> dict[str, object]:
    receipts: dict[str, object] = {}
    inputs = protocol.get("frozen_source_inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "metadata_protocol", "metadata_report", "eligible_observations"
    }:
        raise TargetCatalogAuditError("Frozen source-input set mismatch")
    for role, spec in inputs.items():
        path = root / str(spec["path"])
        if not path.is_file():
            raise TargetCatalogAuditError(f"Frozen input is missing: {role}")
        observed_bytes = path.stat().st_size
        observed_hash = sha256_file(path)
        if observed_bytes != int(spec["bytes"]):
            raise TargetCatalogAuditError(f"Frozen input byte count mismatch: {role}")
        if observed_hash != spec["sha256"]:
            raise TargetCatalogAuditError(f"Frozen input hash mismatch: {role}")
        receipts[role] = {
            "path": spec["path"], "bytes": observed_bytes, "sha256": observed_hash,
        }
    report_path = root / str(inputs["metadata_report"]["path"])
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetCatalogAuditError("Frozen metadata report is unreadable") from exc
    if report.get("decision") != inputs["metadata_report"]["required_decision"]:
        raise TargetCatalogAuditError("Frozen metadata report decision mismatch")
    return receipts


def load_source_rows(protocol: dict[str, Any], *, root: Path = ROOT) -> list[dict[str, object]]:
    spec = protocol["frozen_source_inputs"]["eligible_observations"]
    path = root / spec["path"]
    rows: list[dict[str, object]] = []
    identities: set[tuple[object, ...]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TargetCatalogAuditError(f"Source row {line_number} is invalid JSON") from exc
            if not isinstance(row, dict) or set(row) != EXPECTED_SOURCE_FIELDS:
                raise TargetCatalogAuditError(f"Source row {line_number} field set mismatch")
            try:
                site = _strict_string(row["site_ID"], name="site_ID")
                obs = _strict_string(row["obs_ID"], name="obs_ID")
                source_time = parse_source_utc(row["date"])
                component = _strict_string(row["component_id"], name="component_id")
                latitude = _finite_number(row["representative_latitude"], name="latitude")
                longitude = _finite_number(row["representative_longitude"], name="longitude")
                if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                    raise TargetCatalogAuditError("source coordinate is outside WGS84")
                if type(row["sat_ID"]) is not int or row["sat_ID"] not in (1, 2):
                    raise TargetCatalogAuditError("sat_ID is not exact C1/C2")
                if row["observation_state"] != "null":
                    raise TargetCatalogAuditError("observation_state is not null")
                if row["passes_protected_distance_filter"] is not True:
                    raise TargetCatalogAuditError("protected-distance flag is not true")
                if row["eligible_for_target_catalog"] is not False:
                    raise TargetCatalogAuditError("eligible_for_target_catalog must remain false")
                if type(row["year"]) is not int or row["year"] != source_time.year:
                    raise TargetCatalogAuditError("source year mismatch")
                for name in ("excluded_by_official_mars_test_radius", "excluded_by_prior_negative_radius"):
                    if row[name] is not False:
                        raise TargetCatalogAuditError(f"{name} must be false")
                if type(row["plume_row_count"]) is not int or row["plume_row_count"] != 0:
                    raise TargetCatalogAuditError("plume_row_count must be zero")
                if type(row["representative_positive_data_row"]) is not int or row["representative_positive_data_row"] <= 0:
                    raise TargetCatalogAuditError("representative row must be positive")
                source_data_rows = row["source_data_rows"]
                if (not isinstance(source_data_rows, list) or len(source_data_rows) != 1
                        or type(source_data_rows[0]) is not int or source_data_rows[0] <= 0):
                    raise TargetCatalogAuditError("source_data_rows must contain one positive integer")
                for name in (
                    "positive_coordinate_span_km", "nearest_official_mars_test_km",
                    "nearest_prior_negative_km",
                ):
                    if _finite_number(row[name], name=name) < 0:
                        raise TargetCatalogAuditError(f"{name} must be nonnegative")
                _strict_string(row["nearest_official_mars_test_location"], name="nearest official location")
                _strict_string(row["nearest_prior_negative_id"], name="nearest prior ID")
            except TargetCatalogAuditError as exc:
                raise TargetCatalogAuditError(f"Source row {line_number}: {exc}") from exc
            identity = (site, obs, str(row["date"]), row["sat_ID"])
            if identity in identities:
                raise TargetCatalogAuditError(f"Duplicate source identity at row {line_number}")
            identities.add(identity)
            rows.append(row)
    if len(rows) != int(spec["rows"]):
        raise TargetCatalogAuditError("Frozen source row count mismatch")
    return rows


def validation_plan(*, root: Path = ROOT) -> dict[str, object]:
    protocol = load_protocol()
    receipts = validate_frozen_inputs(protocol, root=root)
    rows = load_source_rows(protocol, root=root)
    return {
        "mode": "validation_only",
        "protocol": {"path": PROTOCOL_RELATIVE_PATH, "sha256": EXPECTED_PROTOCOL_SHA256},
        "frozen_inputs": receipts,
        "source_rows_validated": len(rows),
        "eligible_for_target_catalog_expected_false_rows": len(rows),
        "network_client_created": False,
        "network_executed": False,
        "target_response_opened": False,
        "target_catalog_accessed": False,
        **ACCESS_PROOF,
    }


def source_identity(row: dict[str, object]) -> dict[str, object]:
    return {name: row[name] for name in SOURCE_IDENTITY_FIELDS}


def build_request(row: dict[str, object], sensor: str, protocol: dict[str, Any]) -> dict[str, object]:
    if sensor not in CATALOG_ORDER:
        raise TargetCatalogAuditError("Unknown target sensor")
    catalog = protocol["catalogs"][sensor]
    source_time = parse_source_utc(row["date"])
    seconds = int(protocol["query_contract"]["maximum_absolute_time_offset_seconds"])
    # ``type`` is mandatory STAC Feature structure. It is not a scientific
    # item field in the frozen catalog list, but literal Fields projection by
    # CDSE omits it unless requested, which would make the projected response
    # impossible to validate as a Feature.
    include = ["type", *catalog["required_item_fields"]]
    body: dict[str, object] = {
        "collections": [catalog["collection"]],
        "intersects": {
            "type": "Point",
            "coordinates": [
                _finite_number(row["representative_longitude"], name="longitude"),
                _finite_number(row["representative_latitude"], name="latitude"),
            ],
        },
        "datetime": f"{_rfc3339(source_time - timedelta(seconds=seconds))}/{_rfc3339(source_time + timedelta(seconds=seconds))}",
        "limit": int(protocol["query_contract"]["stac_limit"]),
        "fields": {"include": include, "exclude": ["assets", "links"]},
    }
    canonical = canonical_json_bytes(body)
    return {
        "sensor": sensor,
        "endpoint": catalog["endpoint"],
        "source_identity": source_identity(row),
        "body": body,
        "canonical_request": canonical.decode("utf-8"),
        "canonical_request_sha256": sha256_bytes(canonical),
    }


def _normalized_longitude_delta(longitude: float, reference: float) -> float:
    delta = (longitude - reference + 180.0) % 360.0 - 180.0
    if delta == -180.0 and longitude - reference > 0:
        return 180.0
    return delta


def _position(value: object, *, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) < 2:
        raise TargetCatalogAuditError(f"{name} is not a GeoJSON position")
    longitude = _finite_number(value[0], name=f"{name}.longitude")
    latitude = _finite_number(value[1], name=f"{name}.latitude")
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise TargetCatalogAuditError(f"{name} is outside WGS84")
    return longitude, latitude


def _point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> bool:
    scale = max(1.0, abs(px), abs(py), abs(ax), abs(ay), abs(bx), abs(by))
    tolerance = 1e-12 * scale
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > tolerance:
        return False
    return min(ax, bx) - tolerance <= px <= max(ax, bx) + tolerance and min(ay, by) - tolerance <= py <= max(ay, by) + tolerance


def _segments_intersect(
    left_a: tuple[float, float], left_b: tuple[float, float],
    right_a: tuple[float, float], right_b: tuple[float, float],
) -> bool:
    def cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    values = (
        cross(left_a, left_b, right_a), cross(left_a, left_b, right_b),
        cross(right_a, right_b, left_a), cross(right_a, right_b, left_b),
    )
    scale = max(1.0, *(abs(value) for point in (left_a, left_b, right_a, right_b) for value in point))
    tolerance = 1e-12 * scale * scale
    if ((values[0] > tolerance and values[1] < -tolerance) or (values[0] < -tolerance and values[1] > tolerance)) and (
        (values[2] > tolerance and values[3] < -tolerance) or (values[2] < -tolerance and values[3] > tolerance)
    ):
        return True
    return any((
        abs(values[0]) <= tolerance and _point_on_segment(*right_a, *left_a, *left_b),
        abs(values[1]) <= tolerance and _point_on_segment(*right_b, *left_a, *left_b),
        abs(values[2]) <= tolerance and _point_on_segment(*left_a, *right_a, *right_b),
        abs(values[3]) <= tolerance and _point_on_segment(*left_b, *right_a, *right_b),
    ))


def _ring_relation(point_lon: float, point_lat: float, value: object, *, name: str) -> str:
    if not isinstance(value, list) or len(value) < 4:
        raise TargetCatalogAuditError(f"{name} is not a valid linear ring")
    ring = [_position(position, name=f"{name}[{index}]") for index, position in enumerate(value)]
    first, last = ring[0], ring[-1]
    if _normalized_longitude_delta(first[0], last[0]) != 0.0 or first[1] != last[1]:
        raise TargetCatalogAuditError(f"{name} is not closed")
    unwrapped: list[tuple[float, float]] = [ring[0]]
    for longitude, latitude in ring[1:]:
        previous_longitude = unwrapped[-1][0]
        unwrapped.append(
            (
                previous_longitude
                + _normalized_longitude_delta(longitude, previous_longitude),
                latitude,
            )
        )
    center_longitude = statistics.fmean(longitude for longitude, _ in unwrapped[:-1])
    query_longitude = point_lon + 360.0 * round((center_longitude - point_lon) / 360.0)
    xy = [(longitude - query_longitude, latitude) for longitude, latitude in unwrapped]
    if len(set(xy[:-1])) < 3 or any(left == right for left, right in zip(xy, xy[1:])):
        raise TargetCatalogAuditError(f"{name} is degenerate")
    area_twice = sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(xy, xy[1:])
    )
    if abs(area_twice) <= 1e-12:
        raise TargetCatalogAuditError(f"{name} has zero area")
    segment_count = len(xy) - 1
    for left_index in range(segment_count):
        for right_index in range(left_index + 1, segment_count):
            if right_index == left_index + 1 or (left_index == 0 and right_index == segment_count - 1):
                continue
            if _segments_intersect(
                xy[left_index], xy[left_index + 1],
                xy[right_index], xy[right_index + 1],
            ):
                raise TargetCatalogAuditError(f"{name} self-intersects")
    inside = False
    px = 0.0
    for (ax, ay), (bx, by) in zip(xy, xy[1:]):
        if _point_on_segment(px, point_lat, ax, ay, bx, by):
            return "boundary"
        if (ay > point_lat) != (by > point_lat):
            intersection_x = ax + (point_lat - ay) * (bx - ax) / (by - ay)
            if intersection_x > px:
                inside = not inside
    return "inside" if inside else "outside"


def _polygon_covers(point_lon: float, point_lat: float, coordinates: object, *, name: str) -> bool:
    if not isinstance(coordinates, list) or not coordinates:
        raise TargetCatalogAuditError(f"{name} has no rings")
    outer = _ring_relation(point_lon, point_lat, coordinates[0], name=f"{name}.outer")
    holes = [
        _ring_relation(point_lon, point_lat, hole, name=f"{name}.hole[{index}]")
        for index, hole in enumerate(coordinates[1:], 1)
    ]
    if outer == "outside":
        return False
    if outer == "boundary":
        return True
    for relation in holes:
        if relation == "boundary":
            return True
        if relation == "inside":
            return False
    return True


def geometry_covers_point(geometry: object, longitude: float, latitude: float) -> bool:
    longitude = _finite_number(longitude, name="query longitude")
    latitude = _finite_number(latitude, name="query latitude")
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise TargetCatalogAuditError("Query point is outside WGS84")
    if not isinstance(geometry, dict) or set(geometry) - {"type", "coordinates", "bbox"}:
        raise TargetCatalogAuditError("Malformed GeoJSON geometry object")
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Point":
        lon, lat = _position(coordinates, name="Point")
        return _normalized_longitude_delta(lon, longitude) == 0.0 and lat == latitude
    if geometry_type == "MultiPoint":
        if not isinstance(coordinates, list) or not coordinates:
            raise TargetCatalogAuditError("MultiPoint must contain positions")
        positions = [
            _position(value, name=f"MultiPoint[{index}]")
            for index, value in enumerate(coordinates)
        ]
        return any(
            _normalized_longitude_delta(lon, longitude) == 0.0 and lat == latitude
            for lon, lat in positions
        )
    if geometry_type == "Polygon":
        return _polygon_covers(longitude, latitude, coordinates, name="Polygon")
    if geometry_type == "MultiPolygon":
        if not isinstance(coordinates, list) or not coordinates:
            raise TargetCatalogAuditError("MultiPolygon must contain polygons")
        coverage = [
            _polygon_covers(longitude, latitude, polygon, name=f"MultiPolygon[{index}]")
            for index, polygon in enumerate(coordinates)
        ]
        return any(coverage)
    raise TargetCatalogAuditError("Unsupported or malformed GeoJSON geometry type")


def _validate_bbox(value: object) -> list[float]:
    if not isinstance(value, list) or len(value) not in (4, 6):
        raise TargetCatalogAuditError("Item bbox must have four or six finite numbers")
    result = [_finite_number(number, name="bbox coordinate") for number in value]
    maximum_longitude_index = len(result) // 2
    maximum_latitude_index = maximum_longitude_index + 1
    if not (-180 <= result[0] <= 180 and -180 <= result[maximum_longitude_index] <= 180):
        raise TargetCatalogAuditError("Item bbox longitude bounds are malformed")
    if (
        not (-90 <= result[1] <= 90 and -90 <= result[maximum_latitude_index] <= 90)
        or result[1] > result[maximum_latitude_index]
    ):
        raise TargetCatalogAuditError("Item bbox latitude bounds are malformed")
    if len(result) == 6 and result[2] > result[5]:
        raise TargetCatalogAuditError("Item bbox vertical bounds are malformed")
    return result


def validate_candidate(
    item: object, row: dict[str, object], sensor: str, protocol: dict[str, Any]
) -> dict[str, object]:
    if not isinstance(item, dict) or item.get("type") != "Feature":
        raise TargetCatalogAuditError("STAC response contains a non-Feature")
    catalog = protocol["catalogs"][sensor]
    item_id = _strict_string(item.get("id"), name="item ID")
    if item.get("collection") != catalog["collection"]:
        raise TargetCatalogAuditError("Item collection does not match exact request")
    prefixes = catalog.get("allowed_platform_item_prefixes", catalog.get("allowed_item_prefixes"))
    if not isinstance(prefixes, list) or not any(item_id.startswith(prefix) for prefix in prefixes):
        raise TargetCatalogAuditError("Item platform is not allowed")
    _validate_bbox(item.get("bbox"))
    longitude = _finite_number(row["representative_longitude"], name="source longitude")
    latitude = _finite_number(row["representative_latitude"], name="source latitude")
    if not geometry_covers_point(item.get("geometry"), longitude, latitude):
        raise TargetCatalogAuditError("Full item geometry does not cover the frozen point")
    properties = item.get("properties")
    if not isinstance(properties, dict):
        raise TargetCatalogAuditError("Item properties must be an object")
    # STAC Fields is optional server behavior.  The frozen request excludes
    # assets/links, but a conforming audit must not turn a provider's extra
    # metadata into a false source failure.  Only the explicitly required
    # scalar fields and footprint are validated and copied into the sanitized
    # candidate; provider links, assets, extensions, and extra properties are
    # never persisted.
    item_time = parse_item_utc(properties.get("datetime"))
    source_time = parse_source_utc(row["date"])
    offset = (item_time - source_time).total_seconds()
    maximum = int(protocol["query_contract"]["maximum_absolute_time_offset_seconds"])
    if not math.isfinite(offset) or abs(offset) > maximum:
        raise TargetCatalogAuditError("Item datetime is outside the closed frozen window")
    cloud = _finite_number(properties.get("eo:cloud_cover"), name="eo:cloud_cover")
    if not 0 <= cloud <= 100:
        raise TargetCatalogAuditError("eo:cloud_cover is outside [0,100]")
    if sensor == "landsat_8_9_level_1":
        if properties.get("landsat:correction") != catalog["required_processing_level"]:
            raise TargetCatalogAuditError("Landsat item is not L1TP")
        if properties.get("landsat:collection_category") != catalog["required_collection_category"]:
            raise TargetCatalogAuditError("Landsat item is not Tier 1")
    return {
        **source_identity(row),
        "component_id": row["component_id"],
        "target_sensor": sensor,
        "target_item_id": item_id,
        "target_collection": item["collection"],
        "target_datetime": _rfc3339_precise(item_time),
        "source_target_offset_seconds": offset,
        "absolute_source_target_offset_seconds": abs(offset),
        "eo_cloud_cover": cloud,
        "geometry": item["geometry"],
        "bbox": item["bbox"],
    }


def parse_feature_collection(
    payload: object, row: dict[str, object], sensor: str, protocol: dict[str, Any]
) -> list[dict[str, object]]:
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise TargetCatalogAuditError("Response is not a STAC FeatureCollection")
    features = payload.get("features")
    if not isinstance(features, list):
        raise TargetCatalogAuditError("FeatureCollection features must be a list")
    limit = int(protocol["query_contract"]["stac_limit"])
    if len(features) >= limit:
        raise TargetCatalogAuditError("STAC response reached the frozen 100-item ceiling")
    candidates: list[dict[str, object]] = []
    for item in features:
        candidates.append(validate_candidate(item, row, sensor, protocol))
    return candidates


def select_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: dict[tuple[object, ...], dict[str, object]] = {}
    for candidate in candidates:
        key = tuple(candidate[name] for name in SOURCE_IDENTITY_FIELDS) + (candidate["target_sensor"],)
        rank = (
            float(candidate["absolute_source_target_offset_seconds"]),
            float(candidate["eo_cloud_cover"]), str(candidate["target_item_id"]),
        )
        previous = selected.get(key)
        if previous is None or rank < (
            float(previous["absolute_source_target_offset_seconds"]),
            float(previous["eo_cloud_cover"]), str(previous["target_item_id"]),
        ):
            selected[key] = candidate
    return sorted(
        selected.values(),
        key=lambda item: tuple(str(item[name]) for name in SOURCE_IDENTITY_FIELDS)
        + (str(item["target_sensor"]), str(item["target_item_id"])),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TargetCatalogAuditError(f"Receipt log is invalid at line {line_number}") from exc
            if not isinstance(value, dict):
                raise TargetCatalogAuditError(f"Receipt log row {line_number} is not an object")
            rows.append(value)
    return rows


def response_checkpoint_sha256(receipt: dict[str, object]) -> str:
    bound = {
        name: receipt.get(name)
        for name in (
            "canonical_request_sha256", "endpoint", "sensor", "source_identity",
            "response_bytes", "response_sha256", "parsed_observation_identity",
            "candidate_count", "parsed_candidates_sha256",
        )
    }
    return sha256_bytes(canonical_json_bytes(bound))


def resume_key(
    request_hash: object, endpoint: object, sensor: object, identity: object,
) -> tuple[str, str, str, bytes]:
    if not all(isinstance(value, str) for value in (request_hash, endpoint, sensor)):
        raise TargetCatalogAuditError("Resume key is malformed")
    return (
        str(request_hash), str(endpoint), str(sensor), canonical_json_bytes(identity),
    )


def load_resume_receipts(
    path: Path, *, request_log: Path | None = None,
) -> dict[tuple[str, str, str, bytes], dict[str, object]]:
    accepted: dict[tuple[str, str, str, bytes], dict[str, object]] = {}
    request_keys = None
    if request_log is not None:
        request_keys = set()
        for row in _read_jsonl(request_log):
            if row.get("schema_version") != 1:
                raise TargetCatalogAuditError("Request receipt schema is malformed")
            canonical = row.get("canonical_request")
            if not isinstance(canonical, str) or sha256_bytes(canonical.encode("utf-8")) != row.get("canonical_request_sha256"):
                raise TargetCatalogAuditError("Canonical request receipt hash mismatch")
            request_keys.add(resume_key(
                row.get("canonical_request_sha256"), row.get("endpoint"),
                row.get("sensor"), row.get("source_identity"),
            ))
    for receipt in _read_jsonl(path):
        if receipt.get("accepted") is not True:
            continue
        request_hash = receipt.get("canonical_request_sha256")
        key = resume_key(
            request_hash, receipt.get("endpoint"), receipt.get("sensor"),
            receipt.get("source_identity"),
        )
        if key in accepted:
            raise TargetCatalogAuditError("Accepted response receipts are duplicate or malformed")
        candidates = receipt.get("parsed_candidates")
        if not isinstance(candidates, list):
            raise TargetCatalogAuditError("Accepted response receipt has no parsed candidates")
        if receipt.get("parsed_candidates_sha256") != sha256_bytes(canonical_json_bytes(candidates)):
            raise TargetCatalogAuditError("Accepted parsed-candidate receipt mismatch")
        if receipt.get("candidate_count") != len(candidates):
            raise TargetCatalogAuditError("Accepted candidate-count receipt mismatch")
        if receipt.get("checkpoint_sha256") != response_checkpoint_sha256(receipt):
            raise TargetCatalogAuditError("Accepted response byte/hash checkpoint mismatch")
        if request_keys is not None:
            if key not in request_keys:
                raise TargetCatalogAuditError("Accepted response has no matching request receipt")
        accepted[key] = receipt
    return accepted


def historical_response_bytes(path: Path, *, maximum: int) -> int:
    total = 0
    for receipt in _read_jsonl(path):
        observed = receipt.get("response_bytes")
        digest = receipt.get("response_sha256")
        if type(observed) is not int or observed < 0 or not isinstance(digest, str) or len(digest) != 64:
            raise TargetCatalogAuditError("Response byte/hash receipt is malformed")
        total += observed
        if total > maximum:
            raise TargetCatalogAuditError("Historical STAC responses exceed total byte cap")
    return total


@dataclass
class STACAuditClient:
    session: Any
    protocol: dict[str, Any]
    request_log: Path
    response_log: Path
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    total_network_bytes: int = 0
    historical_network_bytes: int = 0
    http_attempts: int = 0
    last_request_started: float | None = None
    status_counts: Counter[int] = field(default_factory=Counter)

    def _pace(self) -> None:
        minimum = float(self.protocol["query_contract"]["minimum_request_interval_seconds"])
        now = self.monotonic()
        if self.last_request_started is not None:
            remaining = minimum - (now - self.last_request_started)
            if remaining > 0:
                self.sleep(remaining)
        self.last_request_started = self.monotonic()

    def _read_body(self, response: Any) -> bytes:
        each_cap = int(self.protocol["query_contract"]["maximum_response_bytes_each"])
        total_cap = int(self.protocol["query_contract"]["maximum_response_bytes_total"])
        content_encoding = str(response.headers.get("Content-Encoding", "")).lower()
        if content_encoding not in ("", "identity"):
            raise ResponseBodyLimitError(
                "Compressed STAC response rejected for exact byte accounting", b""
            )
        declared = response.headers.get("Content-Length")
        declared_bytes: int | None = None
        if declared is not None:
            try:
                declared_bytes = int(declared)
            except (TypeError, ValueError) as exc:
                raise ResponseBodyLimitError("Invalid STAC Content-Length", b"") from exc
            if declared_bytes < 0 or declared_bytes > each_cap:
                raise ResponseBodyLimitError("STAC response exceeds per-response byte cap", b"")
            if self.historical_network_bytes + self.total_network_bytes + declared_bytes > total_cap:
                raise ResponseBodyLimitError("STAC responses exceed total byte cap", b"")
        chunks: list[bytes] = []
        observed = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            observed += len(chunk)
            self.total_network_bytes += len(chunk)
            chunks.append(chunk)
            if observed > each_cap or self.historical_network_bytes + self.total_network_bytes > total_cap:
                raise ResponseBodyLimitError(
                    "Streamed STAC response exceeds frozen byte cap", b"".join(chunks)
                )
        body = b"".join(chunks)
        if declared_bytes is not None and declared_bytes != observed:
            raise ResponseBodyLimitError("STAC Content-Length does not match observed bytes", body)
        return body

    def execute(
        self, request_record: dict[str, object], row: dict[str, object]
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        endpoint = str(request_record["endpoint"])
        sensor = str(request_record["sensor"])
        catalog = self.protocol["catalogs"][sensor]
        if endpoint != catalog["endpoint"]:
            raise TargetCatalogAuditError("Request endpoint does not match frozen catalog")
        maximum_attempts = int(self.protocol["query_contract"]["maximum_attempts_per_request"])
        retryable = set(self.protocol["query_contract"]["retryable_http_statuses"])
        request_hash = str(request_record["canonical_request_sha256"])
        for attempt in range(1, maximum_attempts + 1):
            self._pace()
            self.http_attempts += 1
            _append_jsonl(self.request_log, {
                "schema_version": 1, "attempt": attempt, "endpoint": endpoint,
                "sensor": sensor, "source_identity": request_record["source_identity"],
                "canonical_request": request_record["canonical_request"],
                "canonical_request_sha256": request_hash,
            })
            response = self.session.post(
                endpoint, data=str(request_record["canonical_request"]).encode("utf-8"),
                allow_redirects=False, stream=True, timeout=(15, 90),
                headers={
                    "Accept": "application/geo+json, application/json",
                    "Accept-Encoding": "identity",
                    "Content-Type": "application/json",
                    "User-Agent": "ERSRR-GHGSat-target-catalog-audit/1.0",
                },
            )
            try:
                status = int(response.status_code)
                self.status_counts[status] += 1
                if getattr(response, "history", []):
                    raise TargetCatalogAuditError("Redirected STAC response rejected")
                if str(getattr(response, "url", endpoint)) != endpoint:
                    raise TargetCatalogAuditError("STAC response endpoint identity mismatch")
                try:
                    body = self._read_body(response)
                except ResponseBodyLimitError as exc:
                    _append_jsonl(self.response_log, {
                        "schema_version": 1, "attempt": attempt, "status": status,
                        "endpoint": endpoint, "sensor": sensor,
                        "source_identity": request_record["source_identity"],
                        "canonical_request_sha256": request_hash,
                        "response_bytes": len(exc.partial_body),
                        "response_sha256": sha256_bytes(exc.partial_body),
                        "body_read_complete": False, "accepted": False,
                    })
                    raise
                receipt: dict[str, object] = {
                    "schema_version": 1, "attempt": attempt, "status": status,
                    "endpoint": endpoint, "sensor": sensor,
                    "source_identity": request_record["source_identity"],
                    "canonical_request_sha256": request_hash,
                    "response_bytes": len(body), "response_sha256": sha256_bytes(body),
                    "body_read_complete": True, "accepted": False,
                }
                if status != 200:
                    _append_jsonl(self.response_log, receipt)
                    if status not in retryable or attempt == maximum_attempts:
                        raise TargetCatalogAuditError(f"STAC HTTP status rejected: {status}")
                    self.sleep(min(2 ** (attempt - 1), 8))
                    continue
                content_type = str(response.headers.get("Content-Type", "")).lower()
                if "json" not in content_type or "html" in content_type:
                    _append_jsonl(self.response_log, receipt)
                    raise TargetCatalogAuditError("STAC response is not JSON metadata")
                try:
                    payload = json.loads(body)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    _append_jsonl(self.response_log, receipt)
                    raise TargetCatalogAuditError("STAC response is invalid JSON") from exc
                try:
                    candidates = parse_feature_collection(payload, row, sensor, self.protocol)
                except TargetCatalogAuditError:
                    _append_jsonl(self.response_log, receipt)
                    raise
                receipt["accepted"] = True
                receipt["parsed_observation_identity"] = source_identity(row)
                receipt["candidate_count"] = len(candidates)
                receipt["parsed_candidates"] = candidates
                receipt["parsed_candidates_sha256"] = sha256_bytes(canonical_json_bytes(candidates))
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
    protocol: dict[str, Any],
) -> list[dict[str, object]]:
    if receipt.get("schema_version") != 1 or receipt.get("accepted") is not True or receipt.get("status") != 200:
        raise TargetCatalogAuditError("Resume receipt is not an accepted HTTP 200 receipt")
    if receipt.get("checkpoint_sha256") != response_checkpoint_sha256(receipt):
        raise TargetCatalogAuditError("Resume response byte/hash checkpoint mismatch")
    if receipt.get("canonical_request_sha256") != request_record["canonical_request_sha256"]:
        raise TargetCatalogAuditError("Resume request hash mismatch")
    if receipt.get("endpoint") != request_record["endpoint"]:
        raise TargetCatalogAuditError("Resume endpoint mismatch")
    if receipt.get("sensor") != request_record["sensor"]:
        raise TargetCatalogAuditError("Resume sensor mismatch")
    if receipt.get("source_identity") != source_identity(row):
        raise TargetCatalogAuditError("Resume source identity mismatch")
    if receipt.get("parsed_observation_identity") != source_identity(row):
        raise TargetCatalogAuditError("Resume parsed observation identity mismatch")
    if type(receipt.get("response_bytes")) is not int or not isinstance(receipt.get("response_sha256"), str):
        raise TargetCatalogAuditError("Resume response byte/hash receipt mismatch")
    raw_candidates = receipt.get("parsed_candidates")
    if not isinstance(raw_candidates, list):
        raise TargetCatalogAuditError("Resume parsed candidates missing")
    if receipt.get("parsed_candidates_sha256") != sha256_bytes(canonical_json_bytes(raw_candidates)):
        raise TargetCatalogAuditError("Resume parsed-candidate hash mismatch")
    candidates: list[dict[str, object]] = []
    sensor = str(request_record["sensor"])
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            raise TargetCatalogAuditError("Resume candidate is not an object")
        properties: dict[str, object] = {
            "datetime": raw_candidate.get("target_datetime"),
            "eo:cloud_cover": raw_candidate.get("eo_cloud_cover"),
        }
        if sensor == "landsat_8_9_level_1":
            catalog = protocol["catalogs"][sensor]
            properties.update({
                "landsat:correction": catalog["required_processing_level"],
                "landsat:collection_category": catalog["required_collection_category"],
            })
        rebuilt = validate_candidate(
            {
                "type": "Feature", "id": raw_candidate.get("target_item_id"),
                "collection": raw_candidate.get("target_collection"),
                "geometry": raw_candidate.get("geometry"), "bbox": raw_candidate.get("bbox"),
                "properties": properties,
            },
            row, sensor, protocol,
        )
        if canonical_json_bytes(rebuilt) != canonical_json_bytes(raw_candidate):
            raise TargetCatalogAuditError("Resume candidate content mismatch")
        candidates.append(rebuilt)
    if len(candidates) != receipt.get("candidate_count"):
        raise TargetCatalogAuditError("Resume candidate-count identity mismatch")
    return candidates


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None, "median": None}
    return {
        "count": len(values), "minimum": min(values), "maximum": max(values),
        "mean": statistics.fmean(values), "median": statistics.median(values),
    }


def evaluate_gates(
    selected: list[dict[str, object]], protocol: dict[str, Any], *, all_valid: bool
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    source_ids = {tuple(row[name] for name in SOURCE_IDENTITY_FIELDS) for row in selected}
    sites = {str(row["site_ID"]) for row in selected}
    components = {str(row["component_id"]) for row in selected}
    items = {str(row["target_item_id"]) for row in selected}
    observed = {
        "minimum_selected_source_sensor_pairs": len(selected),
        "minimum_distinct_source_observations_with_pair": len(source_ids),
        "minimum_distinct_sites_with_pair": len(sites),
        "minimum_novel_25km_components_with_pair": len(components),
        "minimum_distinct_target_item_ids": len(items),
        "all_queries_and_retained_items_valid": all_valid,
    }
    gates: dict[str, dict[str, object]] = {}
    for name, required in protocol["gates"].items():
        value = observed[name]
        passed = value is True if required is True else int(value) >= int(required)
        gates[name] = {"observed": value, "required": required, "pass": passed}
    counts = {
        "selected_source_sensor_pairs": len(selected),
        "distinct_source_observations": len(source_ids), "distinct_sites": len(sites),
        "distinct_components": len(components), "distinct_target_item_ids": len(items),
    }
    return gates, counts


def build_report(
    protocol: dict[str, Any], *, candidates: list[dict[str, object]],
    selected: list[dict[str, object]], logical_queries: int, resumed_queries: int,
    client: STACAuditClient, output_receipts: dict[str, object], all_valid: bool = True,
) -> dict[str, object]:
    gates, distinct = evaluate_gates(selected, protocol, all_valid=all_valid)
    candidate_counts = Counter(str(row["target_sensor"]) for row in candidates)
    selection_counts = Counter(str(row["target_sensor"]) for row in selected)
    return {
        "schema_version": 1,
        "decision": "PASS" if all(bool(value["pass"]) for value in gates.values()) else "FAIL",
        "claim_boundary": protocol["claim_boundary"],
        "queries": {
            "expected": int(protocol["frozen_source_inputs"]["eligible_observations"]["rows"]) * len(CATALOG_ORDER),
            "logical_queries": logical_queries, "resumed_queries": resumed_queries,
            "network_queries": logical_queries - resumed_queries,
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
        "candidate_counts_by_sensor": {sensor: candidate_counts[sensor] for sensor in CATALOG_ORDER},
        "selection_counts_by_sensor": {sensor: selection_counts[sensor] for sensor in CATALOG_ORDER},
        "absolute_offset_seconds_summary": _summary([float(row["absolute_source_target_offset_seconds"]) for row in selected]),
        "cloud_cover_summary": _summary([float(row["eo_cloud_cover"]) for row in selected]),
        "distinct_counts": distinct,
        "gates": gates,
        "output_receipts": output_receipts,
        "access_boundary_proof": dict(ACCESS_PROOF),
        "source_jsonl_mutated": False,
    }


def build_failure_report(
    protocol: dict[str, Any], error: Exception, *, client: STACAuditClient | None = None
) -> dict[str, object]:
    return {
        "schema_version": 1, "decision": "FAIL", "error": str(error),
        "all_queries_and_retained_items_valid": False,
        "queries": {
            "expected": int(protocol["frozen_source_inputs"]["eligible_observations"]["rows"]) * len(CATALOG_ORDER),
            "logical_queries": None, "resumed_queries": None, "network_queries": None,
            "http_attempts": 0 if client is None else client.http_attempts,
        },
        "candidate_counts_by_sensor": {sensor: None for sensor in CATALOG_ORDER},
        "selection_counts_by_sensor": {sensor: None for sensor in CATALOG_ORDER},
        "absolute_offset_seconds_summary": _summary([]),
        "cloud_cover_summary": _summary([]),
        "distinct_counts": {
            "selected_source_sensor_pairs": None, "distinct_source_observations": None,
            "distinct_sites": None, "distinct_components": None,
            "distinct_target_item_ids": None,
        },
        "gates": {
            name: {"observed": False if required is True else None, "required": required, "pass": False}
            for name, required in protocol["gates"].items()
        },
        "http_and_byte_receipts": {
            "network_bytes_observed_this_execution": 0 if client is None else client.total_network_bytes,
            "historical_response_bytes_before_execution": 0 if client is None else client.historical_network_bytes,
            "cumulative_response_bytes": 0 if client is None else client.historical_network_bytes + client.total_network_bytes,
            "http_attempts": 0 if client is None else client.http_attempts,
            "status_counts": {} if client is None else {str(key): value for key, value in sorted(client.status_counts.items())},
        },
        "access_boundary_proof": dict(ACCESS_PROOF),
        "source_jsonl_mutated": False,
        "stale_candidate_and_selected_outputs_removed": True,
        "claim_boundary": protocol["claim_boundary"],
    }


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# GHGSat target-catalog audit", "", f"**Decision: {report['decision']}**", "",
    ]
    if "error" in report:
        lines += [f"Failure: `{report['error']}`", ""]
    queries = report.get("queries")
    if isinstance(queries, dict):
        lines += ["## Query receipts", ""]
        for key, value in queries.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    http = report.get("http_and_byte_receipts")
    if isinstance(http, dict):
        lines += ["## HTTP and byte receipts", ""]
        for key, value in http.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    for heading, key in (("Candidate counts", "candidate_counts_by_sensor"), ("Selection counts", "selection_counts_by_sensor"), ("Distinct counts", "distinct_counts")):
        values = report.get(key)
        if isinstance(values, dict):
            lines += [f"## {heading}", ""]
            for name, value in values.items():
                lines.append(f"- {name}: {value}")
            lines.append("")
    for heading, key in (("Offset summary", "absolute_offset_seconds_summary"), ("Cloud summary", "cloud_cover_summary")):
        values = report.get(key)
        if isinstance(values, dict):
            lines += [f"## {heading}", ""]
            for name, value in values.items():
                lines.append(f"- {name}: {value}")
            lines.append("")
    lines += ["## Gates", ""]
    for name, gate in dict(report.get("gates", {})).items():
        lines.append(f"- {name}: {'PASS' if gate['pass'] else 'FAIL'} (observed {gate['observed']!r}; required {gate['required']!r})")
    lines += ["", "## Forbidden-resource proof", ""]
    for name, accessed in dict(report["access_boundary_proof"]).items():
        lines.append(f"- {name}: {accessed}")
    lines += ["- Source JSONL mutated: False", "", f"Claim boundary: {report['claim_boundary']}", ""]
    return "\n".join(lines)


def _write_reports(protocol: dict[str, Any], report: dict[str, object], *, root: Path = ROOT) -> None:
    _write_json(root / protocol["outputs"]["compact_json"], report)
    _atomic_bytes(root / protocol["outputs"]["compact_markdown"], _markdown(report).encode("utf-8"))


def _remove_stale_outputs(protocol: dict[str, Any], *, root: Path = ROOT) -> None:
    for name in ("ignored_candidates", "ignored_selected_pairs"):
        (root / protocol["outputs"][name]).unlink(missing_ok=True)


def execute_network_audit(
    *, root: Path = ROOT, session: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    protocol = load_protocol()
    client: STACAuditClient | None = None
    try:
        validate_frozen_inputs(protocol, root=root)
        rows = load_source_rows(protocol, root=root)
        outputs = protocol["outputs"]
        request_log = root / outputs["ignored_requests"]
        response_log = root / outputs["ignored_responses"]
        maximum_total = int(protocol["query_contract"]["maximum_response_bytes_total"])
        historical_bytes = historical_response_bytes(response_log, maximum=maximum_total)
        accepted = load_resume_receipts(response_log, request_log=request_log)
        client = STACAuditClient(
            session=session if session is not None else requests.Session(), protocol=protocol,
            request_log=request_log, response_log=response_log, sleep=sleep, monotonic=monotonic,
            historical_network_bytes=historical_bytes,
        )
        candidates: list[dict[str, object]] = []
        logical_queries = 0
        resumed_queries = 0
        for row in rows:
            for sensor in CATALOG_ORDER:
                logical_queries += 1
                request_record = build_request(row, sensor, protocol)
                request_hash = str(request_record["canonical_request_sha256"])
                key = resume_key(
                    request_hash, request_record["endpoint"], sensor,
                    request_record["source_identity"],
                )
                receipt = accepted.get(key)
                if receipt is not None:
                    candidates.extend(_resume_candidates(receipt, request_record, row, protocol))
                    resumed_queries += 1
                else:
                    fresh, _ = client.execute(request_record, row)
                    candidates.extend(fresh)
        expected = len(rows) * len(CATALOG_ORDER)
        if logical_queries != expected:
            raise TargetCatalogAuditError("Did not execute exactly two logical searches per source row")
        selected = select_candidates(candidates)
        candidate_receipt = _write_jsonl(root / outputs["ignored_candidates"], candidates)
        selected_receipt = _write_jsonl(root / outputs["ignored_selected_pairs"], selected)
        report = build_report(
            protocol, candidates=candidates, selected=selected,
            logical_queries=logical_queries, resumed_queries=resumed_queries, client=client,
            output_receipts={"candidates": candidate_receipt, "selected_pairs": selected_receipt},
        )
        _write_reports(protocol, report, root=root)
        return report
    except Exception as exc:
        _remove_stale_outputs(protocol, root=root)
        failure = build_failure_report(protocol, exc, client=client)
        _write_reports(protocol, failure, root=root)
        if isinstance(exc, TargetCatalogAuditError):
            raise
        raise TargetCatalogAuditError(str(exc)) from exc
    finally:
        if session is None and client is not None:
            close = getattr(client.session, "close", None)
            if callable(close):
                close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-network", action="store_true",
        help="execute all 352 exact frozen metadata-only STAC searches (or verified resumes)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute_network_audit() if args.execute_network else validation_plan()
    except TargetCatalogAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("decision", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
