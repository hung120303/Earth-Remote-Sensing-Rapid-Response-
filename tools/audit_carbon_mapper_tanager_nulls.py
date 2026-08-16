"""Audit public Carbon Mapper Tanager source-level null-detect metadata.

The safe default validates the committed protocol, frozen local inputs, and
the snapshotted official API schema without making a network request.  The
explicit metadata mode may call only the three preregistered public catalog
routes.  It cannot access image assets or a Sentinel-2/Landsat target catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_mars_hyperspectral_train_masks import geographic_group_ids  # noqa: E402
from tools.audit_mars_hyperspectral_transfer import (  # noqa: E402
    FORBIDDEN_MARS_COLUMNS,
    SAFE_MARS_COLUMNS,
    haversine_km,
    read_mars_observations,
)
from tools.filter_jpl_cach4_metadata_eligibility import (  # noqa: E402
    load_prior_negative_coordinates,
    nearest_named_distance,
    numeric_summary,
    official_test_locations,
    within_exclusion_radius,
)

EXPECTED_PROTOCOL = ROOT / "configs/mars_carbon_mapper_tanager_null_protocol.json"
EXPECTED_PROTOCOL_SHA256 = (
    "06b2058a05ecf748dac25cb7389bb62e9fe88e36859dded389fc03b13d5b0ad0"
)
SOURCES_URL = (
    "https://api.carbonmapper.org/api/v1/catalog/sources.geojson?"
    "plume_gas=CH4&instrument=tan&cloud_cover_pct_max=25&status=published&"
    "minpoints=1&eps=100"
)
SOURCE_DETAIL_TEMPLATE = (
    "https://api.carbonmapper.org/api/v1/catalog/source/{source_name}?"
    "instrument=tan&plume_gas=CH4&cloud_cover_pct_max=25&status=published&"
    "minpoints=1&explain=true"
)
ANNOTATED_SCENES_TEMPLATE = (
    "https://api.carbonmapper.org/api/v1/catalog/scenes/annotated?"
    "plume_gas=CH4&instruments=tan&not_cloudy=true&sort=asc&limit=100&offset={offset}"
)
IGNORED_ROOT = ROOT / ".research/carbon_mapper_tanager"
SOURCES_CACHE = IGNORED_ROOT / "sources.geojson"
SOURCE_DETAILS_JSONL = IGNORED_ROOT / "source_details.jsonl"
ANNOTATED_SCENES_JSONL = IGNORED_ROOT / "annotated_scenes.jsonl"
NULL_ROWS_JSONL = IGNORED_ROOT / "eligible_null_source_scenes.jsonl"
SOURCE_DETAIL_CACHE = IGNORED_ROOT / "source_detail_cache"
SCENE_PAGE_CACHE = IGNORED_ROOT / "annotated_scene_page_cache"
COMPACT_JSON = ROOT / "reports/acquisition/carbon_mapper_tanager_null_metadata.json"
COMPACT_MARKDOWN = (
    ROOT / "reports/acquisition/CARBON_MAPPER_TANAGER_NULL_METADATA.md"
)

MAX_SOURCE_REQUESTS = 1_000
MAX_SCENE_PAGES = 100
MAX_RESPONSE_BYTES = 8_388_608
MAX_TOTAL_RESPONSE_BYTES = 268_435_456
MINIMUM_REQUEST_INTERVAL_SECONDS = 0.25
MAX_ATTEMPTS = 5
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_SCENES_PER_SOURCE = 4
COORDINATE_TOLERANCE_KM = 0.1
EXCLUSION_RADIUS_KM = 25.0
MINIMUM_PAIRS = 50
MINIMUM_SCENES = 28
MINIMUM_SOURCES = 30
MINIMUM_COMPONENTS = 20

SOURCE_NAME_RE = re.compile(
    r"^CH4_(?P<sector>[A-Za-z0-9-]+(?:_[A-Za-z0-9-]+)*)_100m_"
    r"(?P<longitude>-?(?:0|[1-9]\d*)(?:\.\d+)?)_"
    r"(?P<latitude>-?(?:0|[1-9]\d*)(?:\.\d+)?)$"
)
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


class CarbonMapperAuditError(RuntimeError):
    """Raised when a frozen metadata-audit requirement is violated."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_lf_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256_bytes(payload)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(payload)
    partial.replace(path)


def write_json(path: Path, value: object) -> None:
    write_bytes_atomic(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    payload = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )
    write_bytes_atomic(path, payload)
    return payload.count(b"\n")


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise CarbonMapperAuditError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CarbonMapperAuditError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise CarbonMapperAuditError(f"{name} must be a finite number")
    return result


def _strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CarbonMapperAuditError(f"{name} must be an integer >= {minimum}")
    return value


def _strict_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise CarbonMapperAuditError(f"{name} must be boolean")
    return value


def _strict_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CarbonMapperAuditError(f"{name} must be a non-empty trimmed string")
    return value


def parse_utc(value: object, *, name: str) -> datetime:
    text = _strict_string(value, name=name)
    if not (text.endswith("Z") or text.endswith("+00:00")):
        raise CarbonMapperAuditError(f"{name} must be RFC3339 UTC")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CarbonMapperAuditError(f"{name} must be RFC3339 UTC") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise CarbonMapperAuditError(f"{name} must be RFC3339 UTC")
    return parsed


def parse_uuid(value: object, *, name: str) -> str:
    text = _strict_string(value, name=name).lower()
    if UUID_RE.fullmatch(text) is None:
        raise CarbonMapperAuditError(f"{name} must be a canonical UUID")
    try:
        if str(uuid.UUID(text)) != text:
            raise ValueError
    except ValueError as exc:
        raise CarbonMapperAuditError(f"{name} must be a canonical UUID") from exc
    return text


def parse_point(value: object, *, name: str) -> tuple[float, float]:
    if not isinstance(value, dict) or value.get("type") != "Point":
        raise CarbonMapperAuditError(f"{name} must be a GeoJSON Point")
    coordinates = value.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) != 2:
        raise CarbonMapperAuditError(f"{name} must have two coordinates")
    longitude = _finite_number(coordinates[0], name=f"{name}.longitude")
    latitude = _finite_number(coordinates[1], name=f"{name}.latitude")
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise CarbonMapperAuditError(f"{name} is outside WGS84")
    return latitude, longitude


def validate_protocol_path(path: Path) -> None:
    if path.resolve() != EXPECTED_PROTOCOL.resolve():
        raise CarbonMapperAuditError("Only the exact committed Tanager protocol is permitted")


def _assert_protocol_contract(protocol: dict[str, Any]) -> None:
    api = protocol["permitted_api"]
    expected_api = {
        "hosts": ["api.carbonmapper.org"],
        "sources_query": SOURCES_URL,
        "source_detail_template": SOURCE_DETAIL_TEMPLATE,
        "annotated_scenes_template": ANNOTATED_SCENES_TEMPLATE,
        "maximum_source_detail_requests": MAX_SOURCE_REQUESTS,
        "maximum_scene_pages": MAX_SCENE_PAGES,
        "maximum_response_bytes_each": MAX_RESPONSE_BYTES,
        "maximum_response_bytes_total": MAX_TOTAL_RESPONSE_BYTES,
        "minimum_request_interval_seconds": MINIMUM_REQUEST_INTERVAL_SECONDS,
        "retryable_http_statuses": sorted(RETRYABLE_STATUSES),
        "maximum_attempts_per_request": MAX_ATTEMPTS,
    }
    if api != expected_api:
        raise CarbonMapperAuditError("Frozen API contract mismatch")
    if protocol["deterministic_row_selection"]["maximum_null_scenes_per_source"] != MAX_SCENES_PER_SOURCE:
        raise CarbonMapperAuditError("Frozen row-selection contract mismatch")
    if protocol["eligibility"]["radius_km"] != EXCLUSION_RADIUS_KM:
        raise CarbonMapperAuditError("Frozen spatial contract mismatch")
    gates = protocol["gates"]
    expected_gates = {
        "minimum_authoritative_null_source_scene_pairs": MINIMUM_PAIRS,
        "minimum_distinct_tanager_scenes": MINIMUM_SCENES,
        "minimum_distinct_null_source_points": MINIMUM_SOURCES,
        "minimum_novel_25km_connected_components": MINIMUM_COMPONENTS,
        "all_retained_source_scene_records_valid": True,
        "license_permits_noncommercial_research_derivatives_with_attribution_and_share_alike": True,
    }
    for key, expected in expected_gates.items():
        if gates.get(key) != expected:
            raise CarbonMapperAuditError(f"Frozen gate mismatch: {key}")
    outputs = protocol["outputs"]
    expected_outputs = {
        "ignored_root": ".research/carbon_mapper_tanager",
        "ignored_sources": ".research/carbon_mapper_tanager/sources.geojson",
        "ignored_source_details": ".research/carbon_mapper_tanager/source_details.jsonl",
        "ignored_annotated_scenes": ".research/carbon_mapper_tanager/annotated_scenes.jsonl",
        "ignored_null_rows": ".research/carbon_mapper_tanager/eligible_null_source_scenes.jsonl",
        "compact_json": "reports/acquisition/carbon_mapper_tanager_null_metadata.json",
        "compact_markdown": "reports/acquisition/CARBON_MAPPER_TANAGER_NULL_METADATA.md",
    }
    if outputs != expected_outputs:
        raise CarbonMapperAuditError("Frozen output contract mismatch")
    if not protocol["target_catalog_boundary"]["target_assets_forbidden_in_this_protocol"]:
        raise CarbonMapperAuditError("Target-asset prohibition is not frozen")


def load_protocol(path: Path = EXPECTED_PROTOCOL) -> dict[str, Any]:
    validate_protocol_path(path)
    if sha256_file(path) != EXPECTED_PROTOCOL_SHA256:
        raise CarbonMapperAuditError("Frozen Tanager protocol SHA-256 mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CarbonMapperAuditError("Frozen Tanager protocol must be an object")
    _assert_protocol_contract(value)
    return value


def validate_frozen_local_inputs(protocol: dict[str, Any]) -> dict[str, object]:
    receipts: dict[str, object] = {}
    for role, spec in protocol["frozen_local_inputs"].items():
        path = ROOT / spec["path"]
        if not path.is_file():
            raise CarbonMapperAuditError(f"Frozen local input is missing: {role}")
        if path.stat().st_size != spec["bytes"]:
            raise CarbonMapperAuditError(f"Frozen local-input bytes mismatch: {role}")
        hash_name = "normalized_lf_sha256" if "normalized_lf_sha256" in spec else "sha256"
        observed = normalized_lf_sha256(path) if hash_name == "normalized_lf_sha256" else sha256_file(path)
        if observed != spec[hash_name]:
            raise CarbonMapperAuditError(f"Frozen local-input hash mismatch: {role}")
        receipts[role] = {
            "path": spec["path"],
            "bytes": path.stat().st_size,
            "hash_kind": hash_name,
            "sha256": observed,
        }
    return receipts


def _schema_properties(openapi: dict[str, Any], name: str) -> set[str]:
    try:
        return set(openapi["components"]["schemas"][name]["properties"])
    except (KeyError, TypeError) as exc:
        raise CarbonMapperAuditError(f"OpenAPI schema is missing: {name}") from exc


def validate_openapi(protocol: dict[str, Any]) -> dict[str, object]:
    spec = protocol["official_sources"]["openapi"]
    path = ROOT / spec["ignored_snapshot"]
    if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
        raise CarbonMapperAuditError("Frozen Carbon Mapper OpenAPI snapshot mismatch")
    openapi = json.loads(path.read_text(encoding="utf-8"))
    info = openapi.get("info", {})
    if info.get("title") != spec["api_title"] or info.get("version") != spec["api_version"]:
        raise CarbonMapperAuditError("Carbon Mapper API identity changed")
    path_contracts = {
        "/api/v1/catalog/sources.geojson": ("SourceFeatureCollection", "TokenUserOptionalScopedAuth"),
        "/api/v1/catalog/source/{gas}_{sector}_{eps}m_{lon}_{lat}": ("PlumeSource", "TokenUserOptionalScopedAuth"),
        "/api/v1/catalog/scenes/annotated": ("PagedSceneAnnotatedOut", "TokenUserOptionalScopedAuth"),
    }
    for route, (response_schema, auth_name) in path_contracts.items():
        try:
            operation = openapi["paths"][route]["get"]
            reference = operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        except (KeyError, TypeError) as exc:
            raise CarbonMapperAuditError(f"Required public OpenAPI route changed: {route}") from exc
        if reference != f"#/components/schemas/{response_schema}":
            raise CarbonMapperAuditError(f"OpenAPI response schema changed: {route}")
        security = operation.get("security")
        if security != [{auth_name: []}]:
            raise CarbonMapperAuditError(f"OpenAPI optional-auth contract changed: {route}")
    required = {
        "SourceProperties": {"gas", "source_name", "observation_scenes_names", "observation_date_count", "detection_date_count"},
        "PlumeSource": {"scenes", "point", "source_name", "observation_dates", "detection_dates", "explanation"},
        "SourceExplanation": {"summary", "daily_breakdown"},
        "SourceExplanationSummary": {"number_of_null_detect_days"},
        "SourceExplanationDailyBreakdown": {"date", "has_null_detection", "scenes", "scene_names", "observation_scene_count", "daily_emission_scene_count", "null_detection_scene_count", "detection_scene_count"},
        "SourceExplanationSceneBreakdown": {"id", "name", "timestamp", "instrument", "counts_as_null_detection", "counts_toward_daily_emissions", "has_detection", "has_non_null_emission"},
        "SceneAnnotatedOut": {"id", "bounds", "published_plume_count", "cloud_cover_pct_assessed", "not_cloudy", "timestamp", "published_at", "instrument", "mission_phase", "name"},
        "PagedSceneAnnotatedOut": {"items", "count"},
    }
    for schema, fields in required.items():
        missing = fields - _schema_properties(openapi, schema)
        if missing:
            raise CarbonMapperAuditError(f"OpenAPI {schema} fields changed: {sorted(missing)}")
    return {
        "path": spec["ignored_snapshot"],
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "api_title": info["title"],
        "api_version": info["version"],
        "optional_auth_routes_validated": len(path_contracts),
    }


def validation_plan() -> dict[str, object]:
    protocol = load_protocol()
    return {
        "mode": "validation_only",
        "protocol": {"path": EXPECTED_PROTOCOL.relative_to(ROOT).as_posix(), "sha256": EXPECTED_PROTOCOL_SHA256},
        "frozen_local_inputs": validate_frozen_local_inputs(protocol),
        "openapi": validate_openapi(protocol),
        "network_executed": False,
        "carbon_mapper_catalog_enumerated": False,
        "image_assets_accessed": False,
        "target_catalog_accessed": False,
    }


def validate_api_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "api.carbonmapper.org" or parsed.username or parsed.password or parsed.fragment:
        raise CarbonMapperAuditError("Only the frozen Carbon Mapper API host is permitted")
    if url == SOURCES_URL:
        return
    offsets = [ANNOTATED_SCENES_TEMPLATE.format(offset=100 * index) for index in range(MAX_SCENE_PAGES)]
    if url in offsets:
        return
    prefix, suffix = SOURCE_DETAIL_TEMPLATE.split("{source_name}")
    if url.startswith(prefix) and url.endswith(suffix):
        source_name = url[len(prefix) : len(url) - len(suffix)]
        if SOURCE_NAME_RE.fullmatch(source_name):
            return
    raise CarbonMapperAuditError("URL is outside the frozen Carbon Mapper metadata routes")


@dataclass
class HttpAuditClient:
    session: Any
    sleep: Callable[[float], None] = time.sleep
    total_network_bytes: int = 0
    last_request_started: float | None = None

    def _pace(self) -> None:
        now = time.monotonic()
        if self.last_request_started is not None:
            remaining = MINIMUM_REQUEST_INTERVAL_SECONDS - (now - self.last_request_started)
            if remaining > 0:
                self.sleep(remaining)
        self.last_request_started = time.monotonic()

    def _read_response(self, response: Any, *, expected_url: str) -> bytes:
        if getattr(response, "history", []):
            raise CarbonMapperAuditError("Redirected Carbon Mapper response rejected")
        if getattr(response, "url", expected_url) != expected_url:
            raise CarbonMapperAuditError("Carbon Mapper response URL identity mismatch")
        content_type = str(response.headers.get("Content-Type", "")).lower()
        if "html" in content_type or not ("json" in content_type or "geo+json" in content_type):
            raise CarbonMapperAuditError("Carbon Mapper response is not JSON metadata")
        declared = response.headers.get("Content-Length")
        if declared is not None:
            try:
                declared_bytes = int(declared)
            except ValueError as exc:
                raise CarbonMapperAuditError("Invalid Carbon Mapper Content-Length") from exc
            if declared_bytes < 0 or declared_bytes > MAX_RESPONSE_BYTES:
                raise CarbonMapperAuditError("Carbon Mapper response exceeds per-response cap")
            if self.total_network_bytes + declared_bytes > MAX_TOTAL_RESPONSE_BYTES:
                raise CarbonMapperAuditError("Carbon Mapper responses exceed total byte cap")
        chunks: list[bytes] = []
        observed = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            observed += len(chunk)
            if observed > MAX_RESPONSE_BYTES or self.total_network_bytes + observed > MAX_TOTAL_RESPONSE_BYTES:
                raise CarbonMapperAuditError("Streamed Carbon Mapper response exceeds frozen cap")
            chunks.append(chunk)
        self.total_network_bytes += observed
        payload = b"".join(chunks)
        if payload.lstrip().lower().startswith((b"<html", b"<!doctype html")):
            raise CarbonMapperAuditError("HTML/auth payload rejected")
        return payload

    def get_json(self, url: str, *, cache_path: Path) -> tuple[Any, dict[str, object]]:
        validate_api_url(url)
        if cache_path.is_file():
            payload = cache_path.read_bytes()
            if len(payload) > MAX_RESPONSE_BYTES:
                raise CarbonMapperAuditError("Cached Carbon Mapper response exceeds cap")
            try:
                value = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise CarbonMapperAuditError("Cached Carbon Mapper response is invalid JSON") from exc
            return value, {"url": url, "cache_hit": True, "bytes": len(payload), "sha256": sha256_bytes(payload), "attempts": 0}
        response = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._pace()
            response = self.session.get(
                url,
                allow_redirects=False,
                stream=True,
                timeout=(15, 90),
                headers={"Accept": "application/json, application/geo+json"},
            )
            status = int(response.status_code)
            if status == 200:
                payload = self._read_response(response, expected_url=url)
                try:
                    value = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise CarbonMapperAuditError("Carbon Mapper response is invalid JSON") from exc
                write_bytes_atomic(cache_path, payload)
                return value, {"url": url, "cache_hit": False, "bytes": len(payload), "sha256": sha256_bytes(payload), "attempts": attempt}
            if status not in RETRYABLE_STATUSES or attempt == MAX_ATTEMPTS:
                raise CarbonMapperAuditError(f"Carbon Mapper HTTP status rejected: {status}")
            close = getattr(response, "close", None)
            if callable(close):
                close()
            self.sleep(min(2 ** (attempt - 1), 8))
        raise AssertionError(response)


def parse_source_features(payload: object) -> tuple[list[dict[str, object]], dict[str, int]]:
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
        raise CarbonMapperAuditError("Sources response is not a GeoJSON FeatureCollection")
    seen: set[str] = set()
    candidates: list[dict[str, object]] = []
    rejection_counts: dict[str, int] = defaultdict(int)
    for feature in payload["features"]:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise CarbonMapperAuditError("Sources collection contains a non-Feature")
        try:
            latitude, longitude = parse_point(feature.get("geometry"), name="source.geometry")
            properties = feature.get("properties")
            if not isinstance(properties, dict):
                raise CarbonMapperAuditError("Source properties must be an object")
            source_name = _strict_string(properties.get("source_name"), name="source_name")
            if source_name in seen:
                raise CarbonMapperAuditError(f"Duplicate Carbon Mapper source: {source_name}")
            seen.add(source_name)
            match = SOURCE_NAME_RE.fullmatch(source_name)
            if match is None:
                rejection_counts["source_name_not_exact_100m_ch4"] += 1
                continue
            if properties.get("gas") != "CH4":
                rejection_counts["gas_not_ch4"] += 1
                continue
            encoded_lon = float(match.group("longitude"))
            encoded_lat = float(match.group("latitude"))
            if haversine_km(latitude, longitude, encoded_lat, encoded_lon) > COORDINATE_TOLERANCE_KM:
                rejection_counts["encoded_coordinate_disagreement"] += 1
                continue
            observations = _strict_int(properties.get("observation_date_count"), name="observation_date_count")
            detections = _strict_int(properties.get("detection_date_count"), name="detection_date_count")
            if observations <= detections:
                rejection_counts["no_null_day_headroom"] += 1
                continue
            names = properties.get("observation_scenes_names")
            if not isinstance(names, list) or not names:
                rejection_counts["empty_observation_scene_names"] += 1
                continue
            parsed_names = [_strict_string(item, name="observation_scene_name") for item in names]
            if len(set(parsed_names)) != len(parsed_names):
                raise CarbonMapperAuditError(f"Duplicate observation scene name for {source_name}")
            candidates.append({"source_name": source_name, "latitude": latitude, "longitude": longitude, "observation_date_count": observations, "detection_date_count": detections, "observation_scenes_names": parsed_names})
        except CarbonMapperAuditError:
            raise
    ordered = sorted(candidates, key=lambda row: (sha256_bytes(str(row["source_name"]).encode("utf-8")), str(row["source_name"])))
    return ordered[:MAX_SOURCE_REQUESTS], {**rejection_counts, "features_total": len(payload["features"]), "candidates_before_cap": len(ordered), "candidates_requested": min(len(ordered), MAX_SOURCE_REQUESTS), "truncated_by_request_cap": max(0, len(ordered) - MAX_SOURCE_REQUESTS)}


def parse_annotated_scene(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CarbonMapperAuditError("Annotated scene must be an object")
    scene_id = parse_uuid(value.get("id"), name="annotated_scene.id")
    name = _strict_string(value.get("name"), name="annotated_scene.name")
    if value.get("instrument") != "tan":
        raise CarbonMapperAuditError(f"Annotated scene is not Tanager: {name}")
    if value.get("mission_phase") != "production":
        raise CarbonMapperAuditError(f"Annotated scene is not production-phase: {name}")
    if value.get("published_at") is None:
        raise CarbonMapperAuditError(f"Annotated scene is not public: {name}")
    timestamp = parse_utc(value.get("timestamp"), name="annotated_scene.timestamp")
    published_at = parse_utc(value.get("published_at"), name="annotated_scene.published_at")
    if _strict_bool(value.get("not_cloudy"), name="annotated_scene.not_cloudy") is not True:
        raise CarbonMapperAuditError(f"Annotated scene is cloudy: {name}")
    cloud = _strict_int(value.get("cloud_cover_pct_assessed"), name="cloud_cover_pct_assessed")
    if cloud > 25:
        raise CarbonMapperAuditError(f"Annotated scene exceeds 25% assessed cloud: {name}")
    bounds = value.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise CarbonMapperAuditError(f"Annotated scene has invalid bounds: {name}")
    west, south, east, north = [_finite_number(item, name="annotated_scene.bounds") for item in bounds]
    if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
        raise CarbonMapperAuditError(f"Annotated scene bounds are not finite WGS84: {name}")
    plume_count = _strict_int(value.get("published_plume_count"), name="published_plume_count")
    return {"id": scene_id, "name": name, "instrument": "tan", "mission_phase": "production", "timestamp": timestamp.isoformat(), "published_at": published_at.isoformat(), "not_cloudy": True, "cloud_cover_pct_assessed": cloud, "bounds": [west, south, east, north], "published_plume_count": plume_count}


def parse_scene_page(payload: object, *, offset: int) -> tuple[list[dict[str, object]], int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise CarbonMapperAuditError(f"Annotated scene page at {offset} is invalid")
    count = _strict_int(payload.get("count"), name="annotated_scene.count")
    items = [parse_annotated_scene(item) for item in payload["items"]]
    if len(items) > 100 or offset + len(items) > count:
        raise CarbonMapperAuditError("Annotated scene pagination is inconsistent")
    if offset < count and not items:
        raise CarbonMapperAuditError("Annotated scene pagination ended early")
    return items, count


def _source_detail_cache_path(source_name: str) -> Path:
    digest = sha256_bytes(source_name.encode("utf-8"))
    return SOURCE_DETAIL_CACHE / f"{digest}.json"


def _scene_page_cache_path(offset: int) -> Path:
    return SCENE_PAGE_CACHE / f"offset_{offset:06d}.json"


def fetch_annotated_scenes(client: HttpAuditClient) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    expected_count: int | None = None
    for page in range(MAX_SCENE_PAGES):
        offset = page * 100
        payload, receipt = client.get_json(ANNOTATED_SCENES_TEMPLATE.format(offset=offset), cache_path=_scene_page_cache_path(offset))
        items, count = parse_scene_page(payload, offset=offset)
        if expected_count is None:
            expected_count = count
        elif count != expected_count:
            raise CarbonMapperAuditError("Annotated scene count changed during pagination")
        result.extend(items)
        receipts.append(receipt)
        if len(result) == count:
            break
    else:
        raise CarbonMapperAuditError("Annotated scene enumeration exceeded frozen page cap")
    by_id: set[str] = set()
    by_name: set[str] = set()
    for scene in result:
        if scene["id"] in by_id or scene["name"] in by_name:
            raise CarbonMapperAuditError("Annotated scene IDs/names are not unique")
        by_id.add(str(scene["id"]))
        by_name.add(str(scene["name"]))
    return result, receipts


def _scene_identity(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CarbonMapperAuditError(f"{context} scene must be an object")
    return {"id": parse_uuid(value.get("id"), name=f"{context}.id"), "name": _strict_string(value.get("name"), name=f"{context}.name"), "timestamp": parse_utc(value.get("timestamp"), name=f"{context}.timestamp").isoformat(), "instrument": value.get("instrument")}


def authoritative_null_rows(
    *,
    candidate: dict[str, object],
    detail: object,
    annotated_by_id: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    if not isinstance(detail, dict):
        raise CarbonMapperAuditError("Source detail response must be an object")
    source_name = str(candidate["source_name"])
    if detail.get("source_name") != source_name:
        raise CarbonMapperAuditError(f"Source detail identity mismatch: {source_name}")
    latitude, longitude = parse_point(detail.get("point"), name=f"{source_name}.point")
    if haversine_km(latitude, longitude, float(candidate["latitude"]), float(candidate["longitude"])) > COORDINATE_TOLERANCE_KM:
        raise CarbonMapperAuditError(f"Source detail point mismatch: {source_name}")
    top = detail.get("scenes")
    if not isinstance(top, list):
        raise CarbonMapperAuditError(f"Source detail scenes missing: {source_name}")
    top_by_id: dict[str, dict[str, object]] = {}
    top_names: set[str] = set()
    for item in top:
        scene = _scene_identity(item, context="top_level")
        if scene["instrument"] != "tan":
            raise CarbonMapperAuditError(f"Non-Tanager top-level scene: {source_name}")
        if scene["id"] in top_by_id or scene["name"] in top_names:
            raise CarbonMapperAuditError(f"Duplicate top-level scene: {source_name}")
        top_by_id[str(scene["id"])] = scene
        top_names.add(str(scene["name"]))
    explanation = detail.get("explanation")
    if not isinstance(explanation, dict) or not isinstance(explanation.get("summary"), dict) or not isinstance(explanation.get("daily_breakdown"), list):
        raise CarbonMapperAuditError(f"Source explanation missing: {source_name}")
    summary = explanation["summary"]
    declared_null_days = _strict_int(summary.get("number_of_null_detect_days"), name="number_of_null_detect_days")
    if declared_null_days <= 0:
        return []
    rows: list[dict[str, object]] = []
    observed_null_days = 0
    observed_scene_ids: set[str] = set()
    for day in explanation["daily_breakdown"]:
        if not isinstance(day, dict) or not isinstance(day.get("scenes"), list):
            raise CarbonMapperAuditError(f"Invalid daily breakdown: {source_name}")
        day_text = _strict_string(day.get("date"), name="daily_breakdown.date")
        try:
            datetime.strptime(day_text, "%Y-%m-%d")
        except ValueError as exc:
            raise CarbonMapperAuditError("Daily breakdown date is invalid") from exc
        scenes = [_scene_identity(item, context="daily") | {
            "counts_as_null_detection": _strict_bool(item.get("counts_as_null_detection"), name="counts_as_null_detection"),
            "counts_toward_daily_emissions": _strict_bool(item.get("counts_toward_daily_emissions"), name="counts_toward_daily_emissions"),
            "has_detection": _strict_bool(item.get("has_detection"), name="has_detection"),
            "has_non_null_emission": _strict_bool(item.get("has_non_null_emission"), name="has_non_null_emission"),
        } for item in day["scenes"]]
        if len({str(scene["id"]) for scene in scenes}) != len(scenes):
            raise CarbonMapperAuditError(f"Duplicate daily scene: {source_name} {day_text}")
        expected_counts = {
            "observation_scene_count": len(scenes),
            "daily_emission_scene_count": sum(bool(scene["counts_toward_daily_emissions"]) for scene in scenes),
            "null_detection_scene_count": sum(bool(scene["counts_as_null_detection"]) for scene in scenes),
            "detection_scene_count": sum(bool(scene["has_detection"]) for scene in scenes),
        }
        for field, expected in expected_counts.items():
            if _strict_int(day.get(field), name=field) != expected:
                raise CarbonMapperAuditError(f"Daily scene counter mismatch: {source_name} {field}")
        scene_names = day.get("scene_names")
        if not isinstance(scene_names, list) or set(scene_names) != {scene["name"] for scene in scenes} or len(scene_names) != len(scenes):
            raise CarbonMapperAuditError(f"Daily scene-name mismatch: {source_name}")
        has_null = _strict_bool(day.get("has_null_detection"), name="has_null_detection")
        if has_null != any(bool(scene["counts_as_null_detection"]) for scene in scenes):
            raise CarbonMapperAuditError(f"Daily null flag mismatch: {source_name}")
        observed_null_days += int(has_null)
        for scene in scenes:
            if scene["instrument"] != "tan":
                continue
            top_scene = top_by_id.get(str(scene["id"]))
            annotated = annotated_by_id.get(str(scene["id"]))
            if top_scene is None or top_scene["name"] != scene["name"] or top_scene["timestamp"] != scene["timestamp"]:
                raise CarbonMapperAuditError(f"Daily/top-level scene mismatch: {source_name}")
            if annotated is None or annotated["name"] != scene["name"] or annotated["timestamp"] != scene["timestamp"]:
                raise CarbonMapperAuditError(f"Daily/annotated scene mismatch: {source_name}")
            timestamp = datetime.fromisoformat(str(scene["timestamp"]))
            if timestamp.date().isoformat() != day_text:
                raise CarbonMapperAuditError(f"Daily scene date mismatch: {source_name}")
            west, south, east, north = [float(item) for item in annotated["bounds"]]
            if not (west <= longitude <= east and south <= latitude <= north):
                raise CarbonMapperAuditError(f"Source is outside annotated scene bounds: {source_name}")
            qualifies = (
                scene["counts_as_null_detection"] is True
                and scene["counts_toward_daily_emissions"] is True
                and scene["has_detection"] is False
                and scene["has_non_null_emission"] is False
            )
            if not qualifies:
                continue
            if str(scene["id"]) in observed_scene_ids:
                raise CarbonMapperAuditError(f"Null scene repeated across days: {source_name}")
            observed_scene_ids.add(str(scene["id"]))
            row_id = sha256_bytes((source_name + "\0" + str(scene["id"])).encode("utf-8"))
            rows.append({
                "row_id": row_id,
                "source_name": source_name,
                "scene_id": scene["id"],
                "scene_name": scene["name"],
                "timestamp": scene["timestamp"],
                "instrument": "tan",
                "mission_phase": "production",
                "latitude": latitude,
                "longitude": longitude,
                "counts_as_null_detection": True,
                "counts_toward_daily_emissions": True,
                "has_detection": False,
                "has_non_null_emission": False,
                "cloud_cover_pct_assessed": annotated["cloud_cover_pct_assessed"],
                "not_cloudy": True,
                "published_at": annotated["published_at"],
                "published_plume_count_scene_wide_not_used_for_label": annotated["published_plume_count"],
                "label_claim": "No Carbon Mapper detection above applicable Tanager sensitivity at this reviewed source in this qualifying scene; not physical zero methane.",
                "passes_authoritative_null_contract": True,
                "eligible_for_target_catalog": False,
            })
    if observed_null_days != declared_null_days:
        raise CarbonMapperAuditError(f"Source null-day summary mismatch: {source_name}")
    return rows


def select_rows_per_source(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_name"])].append(row)
    selected: list[dict[str, object]] = []
    for source_name in sorted(grouped):
        ordered = sorted(grouped[source_name], key=lambda row: (sha256_bytes((source_name + "\0" + str(row["scene_id"])).encode("utf-8")), str(row["scene_id"])))
        selected.extend(ordered[:MAX_SCENES_PER_SOURCE])
    return selected


def spatial_filter_rows(
    rows: list[dict[str, object]],
    *,
    all_mars_locations: dict[str, tuple[float, float]],
    protected_mars_locations: dict[str, tuple[float, float]],
    prior_negative_coordinates: dict[str, tuple[float, float]],
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    passing_coordinates: dict[str, tuple[float, float]] = {}
    for source in rows:
        row = dict(source)
        latitude, longitude = float(row["latitude"]), float(row["longitude"])
        nearest_test, test_name = nearest_named_distance(latitude, longitude, protected_mars_locations)
        nearest_all, all_name = nearest_named_distance(latitude, longitude, all_mars_locations)
        nearest_prior, prior_id = nearest_named_distance(latitude, longitude, prior_negative_coordinates)
        test_overlap = within_exclusion_radius(nearest_test, EXCLUSION_RADIUS_KM)
        prior_overlap = within_exclusion_radius(nearest_prior, EXCLUSION_RADIUS_KM)
        passes = not test_overlap and not prior_overlap
        if passes:
            passing_coordinates[str(row["row_id"])] = (latitude, longitude)
        reasons: list[str] = []
        if test_overlap:
            reasons.append("within_25km_of_official_mars_test_location")
        if prior_overlap:
            reasons.append("within_25km_of_counted_prior_negative_source_crop")
        row.update({
            "nearest_mars_test_km": nearest_test,
            "nearest_mars_test_location": test_name,
            "nearest_any_mars_km": nearest_all,
            "nearest_any_mars_location": all_name,
            "nearest_prior_negative_pair_km": nearest_prior,
            "nearest_prior_negative_sample_id": prior_id,
            "novel_beyond_all_mars_25km": nearest_all > EXCLUSION_RADIUS_KM,
            "passes_frozen_spatial_filter": passes,
            "eligibility_status": "passes_frozen_metadata_filter_target_catalog_not_authorized" if passes else ";".join(reasons),
            "eligible_for_target_catalog": False,
            "group_id": None,
            "component_novel_beyond_all_mars_25km": None,
        })
        filtered.append(row)
    groups = geographic_group_ids(passing_coordinates, EXCLUSION_RADIUS_KM)
    members: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in filtered:
        group_id = groups.get(str(row["row_id"]))
        row["group_id"] = group_id
        if group_id:
            members[group_id].append(row)
    novelty = {group: all(bool(row["novel_beyond_all_mars_25km"]) for row in values) for group, values in members.items()}
    for row in filtered:
        if row["group_id"]:
            row["component_novel_beyond_all_mars_25km"] = novelty[str(row["group_id"])]
    return filtered


def _report_file(path: Path) -> dict[str, object]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_report(
    *,
    protocol: dict[str, Any],
    source_counts: dict[str, int],
    source_details: list[dict[str, object]],
    annotated_scenes: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    filtered_rows: list[dict[str, object]],
    receipts: list[dict[str, object]],
    client: HttpAuditClient,
    frozen_receipts: dict[str, object],
    prior_counts: dict[str, int],
) -> dict[str, object]:
    passing = [row for row in filtered_rows if row["passes_frozen_spatial_filter"]]
    scenes = {str(row["scene_id"]) for row in passing}
    sources = {str(row["source_name"]) for row in passing}
    groups = {str(row["group_id"]) for row in passing if row["component_novel_beyond_all_mars_25km"]}
    gates = {
        "license_permits_noncommercial_research_derivatives_with_attribution_and_share_alike": {"required": True, "observed": bool(protocol["gates"]["license_permits_noncommercial_research_derivatives_with_attribution_and_share_alike"]), "passed": True},
        "minimum_authoritative_null_source_scene_pairs": {"required": MINIMUM_PAIRS, "observed": len(passing), "passed": len(passing) >= MINIMUM_PAIRS},
        "minimum_distinct_tanager_scenes": {"required": MINIMUM_SCENES, "observed": len(scenes), "passed": len(scenes) >= MINIMUM_SCENES},
        "minimum_distinct_null_source_points": {"required": MINIMUM_SOURCES, "observed": len(sources), "passed": len(sources) >= MINIMUM_SOURCES},
        "minimum_novel_25km_connected_components": {"required": MINIMUM_COMPONENTS, "observed": len(groups), "passed": len(groups) >= MINIMUM_COMPONENTS},
        "all_retained_source_scene_records_valid": {"required": True, "observed": True, "passed": True},
    }
    decision = "PASS" if all(bool(gate["passed"]) for gate in gates.values()) else "FAIL"
    return {
        "schema_version": 1,
        "decision": decision,
        "protocol": {"path": EXPECTED_PROTOCOL.relative_to(ROOT).as_posix(), "sha256": EXPECTED_PROTOCOL_SHA256, "committed_before_catalog_enumeration": True},
        "counts": {
            **source_counts,
            "source_details_requested": len(source_details),
            "annotated_production_tanager_scenes": len(annotated_scenes),
            "authoritative_null_rows_before_per_source_cap": sum(int(detail["authoritative_null_row_count"]) for detail in source_details),
            "deterministically_selected_rows": len(selected_rows),
            "rows_passing_official_test_and_prior_negative_exclusions": len(passing),
            "distinct_passing_tanager_scenes": len(scenes),
            "distinct_passing_null_sources": len(sources),
            "passing_components": len({str(row["group_id"]) for row in passing}),
            "novel_beyond_all_mars_components": len(groups),
        },
        "gates": gates,
        "distance_km": {
            "nearest_official_mars_test": numeric_summary(float(row["nearest_mars_test_km"]) for row in filtered_rows),
            "nearest_prior_counted_negative": numeric_summary(float(row["nearest_prior_negative_pair_km"]) for row in filtered_rows),
            "nearest_any_mars": numeric_summary(float(row["nearest_any_mars_km"]) for row in filtered_rows),
        },
        "network": {"network_bytes_this_execution": client.total_network_bytes, "requests_this_execution": sum(0 if receipt["cache_hit"] else 1 for receipt in receipts), "cache_hits": sum(bool(receipt["cache_hit"]) for receipt in receipts), "receipts": receipts},
        "inputs": frozen_receipts,
        "prior_negative_counts": prior_counts,
        "outputs": {"sources": _report_file(SOURCES_CACHE), "source_details": _report_file(SOURCE_DETAILS_JSONL), "annotated_scenes": _report_file(ANNOTATED_SCENES_JSONL), "null_rows": _report_file(NULL_ROWS_JSONL)},
        "access_boundary": {"carbon_mapper_metadata_catalog_enumerated": True, "carbon_mapper_image_assets_accessed": False, "target_sentinel_landsat_catalog_accessed": False, "protected_mars_outcome_columns_read": False, "every_detailed_row_eligible_for_target_catalog": False},
        "label_claim": protocol["authoritative_null_contract"]["claim"],
        "license_boundary": protocol["license_contract"],
        "next_action": protocol["pass_action"] if decision == "PASS" else protocol["failure_action"],
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_markdown(report: dict[str, object]) -> None:
    counts = report["counts"]
    lines = [
        "# Carbon Mapper Tanager null metadata audit",
        "",
        f"**Decision:** {report['decision']}",
        "",
        "This audit used public, source-local Tanager null-detect metadata only. It did not access Carbon Mapper imagery or query a Sentinel-2/Landsat catalog.",
        "",
        "## Observed population",
        "",
        f"- Candidate source details audited: {counts['source_details_requested']}",
        f"- Deterministically selected null pairs: {counts['deterministically_selected_rows']}",
        f"- Pairs after leakage exclusions: {counts['rows_passing_official_test_and_prior_negative_exclusions']}",
        f"- Distinct passing Tanager scenes: {counts['distinct_passing_tanager_scenes']}",
        f"- Distinct passing source points: {counts['distinct_passing_null_sources']}",
        f"- All-MARS-novel 25 km components: {counts['novel_beyond_all_mars_components']}",
        "",
        "## Frozen gates",
        "",
    ]
    for name, gate in report["gates"].items():
        lines.append(f"- `{name}`: {'PASS' if gate['passed'] else 'FAIL'} ({gate['observed']} observed; {gate['required']} required)")
    lines.extend([
        "",
        "The label means no Carbon Mapper detection above the applicable Tanager sensitivity at that reviewed source and scene. It does not mean physically zero methane. Scene-wide plume count was retained only for auditability and was never a label criterion.",
        "",
        "Every detailed row remains `eligible_for_target_catalog=false`. A new committed protocol is required before any target-satellite query.",
        "",
    ])
    write_bytes_atomic(COMPACT_MARKDOWN, "\n".join(lines).encode("utf-8"))


def execute_metadata_audit(session: Any | None = None, *, sleep: Callable[[float], None] = time.sleep) -> dict[str, object]:
    protocol = load_protocol()
    frozen_receipts = validate_frozen_local_inputs(protocol)
    validate_openapi(protocol)
    if SAFE_MARS_COLUMNS & FORBIDDEN_MARS_COLUMNS:
        raise AssertionError("Safe and protected MARS columns overlap")
    client = HttpAuditClient(session=requests.Session() if session is None else session, sleep=sleep)
    sources_payload, sources_receipt = client.get_json(SOURCES_URL, cache_path=SOURCES_CACHE)
    candidates, source_counts = parse_source_features(sources_payload)
    annotated_scenes, scene_receipts = fetch_annotated_scenes(client)
    annotated_by_id = {str(row["id"]): row for row in annotated_scenes}
    receipts = [sources_receipt, *scene_receipts]
    all_null_rows: list[dict[str, object]] = []
    source_details: list[dict[str, object]] = []
    for candidate in candidates:
        source_name = str(candidate["source_name"])
        url = SOURCE_DETAIL_TEMPLATE.format(source_name=source_name)
        detail, receipt = client.get_json(url, cache_path=_source_detail_cache_path(source_name))
        receipts.append(receipt)
        null_rows = authoritative_null_rows(candidate=candidate, detail=detail, annotated_by_id=annotated_by_id)
        all_null_rows.extend(null_rows)
        source_details.append({"source_name": source_name, "latitude": candidate["latitude"], "longitude": candidate["longitude"], "response_sha256": receipt["sha256"], "response_bytes": receipt["bytes"], "authoritative_null_row_count": len(null_rows), "eligible_for_target_catalog": False})
    selected = select_rows_per_source(all_null_rows)
    frozen = protocol["frozen_local_inputs"]
    safe_manifest = ROOT / frozen["safe_mars_manifest"]["path"]
    mars = read_mars_observations(safe_manifest)
    all_mars, protected_mars = official_test_locations(mars)
    prior_coordinates, prior_counts = load_prior_negative_coordinates(
        stage_b_report_path=ROOT / frozen["mars_hyperspectral_stage_b_report"]["path"],
        pair_catalog_path=ROOT / frozen["mars_hyperspectral_pairs"]["path"],
        mask_catalog_path=ROOT / frozen["mars_hyperspectral_mask_catalog"]["path"],
    )
    filtered = spatial_filter_rows(selected, all_mars_locations=all_mars, protected_mars_locations=protected_mars, prior_negative_coordinates=prior_coordinates)
    write_jsonl(SOURCE_DETAILS_JSONL, source_details)
    write_jsonl(ANNOTATED_SCENES_JSONL, annotated_scenes)
    write_jsonl(NULL_ROWS_JSONL, filtered)
    report = build_report(protocol=protocol, source_counts=source_counts, source_details=source_details, annotated_scenes=annotated_scenes, selected_rows=selected, filtered_rows=filtered, receipts=receipts, client=client, frozen_receipts=frozen_receipts, prior_counts=prior_counts)
    write_json(COMPACT_JSON, report)
    _write_markdown(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-metadata-audit", action="store_true", help="Enumerate only the preregistered public Carbon Mapper metadata routes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = execute_metadata_audit() if args.execute_metadata_audit else validation_plan()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
