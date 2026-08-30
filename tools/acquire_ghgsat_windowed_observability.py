#!/usr/bin/env python3
"""Acquire the frozen GHGSat scene-head-negative observability cohort.

The default command is deliberately offline: it validates the exact protocol and
all hash-bound local inputs, and does not construct an HTTP client, retain an
asset URL, open a raster, or load CloudSEN12 weights.  Only the explicit
``--execute-network`` mode crosses that boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote

import numpy as np

try:
    from tools import audit_ghgsat_reference_catalog as reference
except ModuleNotFoundError as exc:  # direct ``python tools/...py`` execution
    if exc.name != "tools":
        raise
    import audit_ghgsat_reference_catalog as reference  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_RELATIVE_PATH = "configs/mars_ghgsat_windowed_observability_protocol.json"
EXPECTED_PROTOCOL = ROOT / PROTOCOL_RELATIVE_PATH
EXPECTED_PROTOCOL_SHA256 = "75be0a55f954c2c20ba548496d6ed39d6a177f0b56da0775d19659efcf94f8a4"
PAIR_JOIN_KEY = ("site_ID", "obs_ID", "date", "sat_ID", "target_sensor", "target_item_id")
OBSERVATION_KEY = ("site_ID", "obs_ID", "date", "sat_ID")
S2_SENSOR = "sentinel_2_l1c"
LANDSAT_SENSOR = "landsat_8_9_level_1"
SENSORS = (S2_SENSOR, LANDSAT_SENSOR)
SIZE = 200
RESOLUTION = 10.0
MIN_FRACTION = 0.8
S2_TRAINING_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
LANDSAT_TRAINING_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07")
CLOUDSEN_BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
)
S2_EXTRA_BANDS = tuple(b for b in CLOUDSEN_BANDS if b not in S2_TRAINING_BANDS)
S2_PREFIX = "https://sentinel-s2-l1c.s3.amazonaws.com/"
LANDSAT_PREFIX = "https://landsatlook.usgs.gov/data/collection02/level-1/standard/oli-tirs/"
EXPECTED_OUTPUTS = {
    "ignored_root": ".research/ghgsat_landfill_null/windowed_observability",
    "ignored_request_receipts": ".research/ghgsat_landfill_null/windowed_observability/requests.jsonl",
    "ignored_response_receipts": ".research/ghgsat_landfill_null/windowed_observability/responses.jsonl",
    "ignored_asset_manifest": ".research/ghgsat_landfill_null/windowed_observability/resolved_assets.jsonl",
    "ignored_crop_root": ".research/ghgsat_landfill_null/windowed_observability/crops",
    "ignored_observable_manifest": ".research/ghgsat_landfill_null/windowed_observability/observable_scene_head_negatives.jsonl",
    "compact_json": "reports/acquisition/ghgsat_landfill_windowed_observability.json",
    "compact_markdown": "reports/acquisition/GHGSAT_LANDFILL_WINDOWED_OBSERVABILITY.md",
}


class ObservabilityAuditError(RuntimeError):
    """A fail-closed frozen-contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_value(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _assert_protocol_contract(protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ObservabilityAuditError("Frozen protocol schema mismatch")
    if protocol.get("status") != "frozen_before_any_ghgsat_target_or_reference_asset_url_or_raster_access":
        raise ObservabilityAuditError("Frozen protocol status mismatch")
    if protocol.get("outputs") != EXPECTED_OUTPUTS:
        raise ObservabilityAuditError("Frozen output paths mismatch")
    expected_inputs = {
        "reference_catalog_protocol": ("configs/mars_ghgsat_reference_catalog_protocol.json", 7998, "04d88d0fb6a11c5c54c4b1f38f6872de79db436240e175dfced75b82f2050f24"),
        "reference_catalog_report": ("reports/acquisition/ghgsat_landfill_reference_catalog.json", 3601, "416913dc048093cb2e04a54004b15b6c8f7a4b76f9e495d1a354781d7dbf6516"),
        "target_reference_pairs": (".research/ghgsat_landfill_null/reference_catalog/target_reference_pairs.jsonl", 73828, "ebd2cd3e4f4a201905f441c86eeabd1a50798e57c3818a3bd46d4fb11709f6a2"),
        "selected_targets": (".research/ghgsat_landfill_null/target_catalog/selected_pairs.jsonl", 59230, "8c7942f0ac7bc07e250603e25ff23bb345e7bafbd86394e4ce0e42d41a33f6a8"),
        "cloudsen12_weights": ("EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/emit-v002-external-l1c-2026-07/_cloudsen12_weights/UNetMobV2_V2.pt", 26796072, "218fa69aa3c7212d4e690b48af88ac6f3c976fc50d07f275b8fd623909183d7a"),
    }
    inputs = protocol.get("frozen_inputs")
    if not isinstance(inputs, dict) or tuple(inputs) != tuple(expected_inputs):
        raise ObservabilityAuditError("Frozen input set/order mismatch")
    for name, expected in expected_inputs.items():
        spec = inputs[name]
        if (spec.get("path"), spec.get("bytes"), spec.get("sha256")) != expected:
            raise ObservabilityAuditError(f"Frozen input contract mismatch: {name}")
    if inputs["reference_catalog_report"].get("required_decision") != "PASS":
        raise ObservabilityAuditError("Reference report decision contract mismatch")
    if inputs["target_reference_pairs"].get("rows") != 76 or inputs["selected_targets"].get("rows") != 79:
        raise ObservabilityAuditError("Frozen input row contract mismatch")
    pair = protocol.get("pair_contract", {})
    if pair.get("exact_rows_attempted") != 76 or pair.get("join_key") != list(PAIR_JOIN_KEY):
        raise ObservabilityAuditError("Frozen pair/join contract mismatch")
    s2 = protocol.get("sentinel_2_l1c_resolution", {})
    if (
        s2.get("search_endpoint") != "https://earth-search.aws.element84.com/v1/search"
        or s2.get("collection") != "sentinel-2-l1c"
        or s2.get("crosswalk_window_seconds") != 1
        or s2.get("allowed_asset_prefix") != S2_PREFIX
        or s2.get("l2_products_forbidden") is not True
    ):
        raise ObservabilityAuditError("Frozen Sentinel-2 resolution contract mismatch")
    landsat = protocol.get("landsat_8_9_level_1_resolution", {})
    if (
        landsat.get("item_endpoint_template") != "https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l1/items/{item_id}"
        or landsat.get("required_processing_level") != "L1TP"
        or landsat.get("required_collection_category") != "T1"
        or landsat.get("allowed_asset_prefix") != LANDSAT_PREFIX
    ):
        raise ObservabilityAuditError("Frozen Landsat resolution contract mismatch")
    network = protocol.get("asset_resolution_network_contract", {})
    expected_network = {
        "metadata_response_maximum_bytes_each": 4 * 1024 * 1024,
        "metadata_response_maximum_bytes_total": 128 * 1024 * 1024,
        "minimum_request_interval_seconds": 0.25,
        "maximum_attempts_per_request": 5,
        "retryable_http_statuses": [429, 500, 502, 503, 504],
        "redirects_forbidden": True,
        "append_only_request_response_receipts": True,
        "asset_urls_retained_only_in_ignored_manifests": True,
        "full_product_download_forbidden": True,
        "previews_and_thumbnails_forbidden": True,
    }
    if network != expected_network:
        raise ObservabilityAuditError("Frozen network contract mismatch")
    crop = protocol.get("crop_contract", {})
    if (
        crop.get("shape_pixels") != [SIZE, SIZE]
        or crop.get("resolution_m") != 10
        or crop.get("training_band_order") != ["B02", "B03", "B04", "B08_or_B05", "B11_or_B06", "B12_or_B07"]
        or crop.get("frame_order") != ["target", "reference"]
        or crop.get("storage") != EXPECTED_OUTPUTS["ignored_crop_root"]
        or crop.get("storage_must_be_git_ignored") is not True
    ):
        raise ObservabilityAuditError("Frozen crop contract mismatch")
    local = protocol.get("local_observability", {})
    if (
        local.get("minimum_radiometric_valid_fraction") != MIN_FRACTION
        or local.get("minimum_union_clear_fraction") != MIN_FRACTION
        or local.get("landsat_nonclear_qa_pixel_bits") != [0, 1, 2, 3, 4, 5]
        or local.get("sentinel_cloud_model", {}).get("band_order") != list(CLOUDSEN_BANDS)
        or local.get("sentinel_cloud_model", {}).get("weights_sha256") != expected_inputs["cloudsen12_weights"][2]
    ):
        raise ObservabilityAuditError("Frozen local-observability contract mismatch")
    supervision = protocol.get("supervision_contract", {})
    if (
        supervision.get("scene_presence_label") != 0
        or supervision.get("dense_mask_or_dense_pixel_loss_authorized") is not False
        or supervision.get("zero_mask_creation_forbidden") is not True
    ):
        raise ObservabilityAuditError("Frozen scene-head-only contract mismatch")
    if protocol.get("gates") != {
        "minimum_locally_observable_source_sensor_pairs": 28,
        "minimum_distinct_locally_observable_source_observations": 28,
        "minimum_distinct_sites": 20,
        "minimum_novel_25km_components": 20,
        "all_retained_assets_products_and_crops_valid": True,
        "no_dense_supervision_created": True,
    }:
        raise ObservabilityAuditError("Frozen final gates mismatch")


def load_protocol(path: Path = EXPECTED_PROTOCOL) -> dict[str, Any]:
    if path.resolve() != EXPECTED_PROTOCOL.resolve():
        raise ObservabilityAuditError("Only the exact committed observability protocol is permitted")
    if sha256_file(path) != EXPECTED_PROTOCOL_SHA256:
        raise ObservabilityAuditError("Frozen observability protocol SHA-256 mismatch")
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObservabilityAuditError("Frozen observability protocol is unreadable") from exc
    if not isinstance(protocol, dict):
        raise ObservabilityAuditError("Frozen observability protocol must be an object")
    _assert_protocol_contract(protocol)
    return protocol


def validate_frozen_inputs(protocol: dict[str, Any], *, root: Path = ROOT) -> dict[str, object]:
    receipts: dict[str, object] = {}
    for name, spec in protocol["frozen_inputs"].items():
        path = root / spec["path"]
        if not path.is_file() or path.stat().st_size != spec["bytes"]:
            raise ObservabilityAuditError(f"Frozen input missing or byte count changed: {name}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise ObservabilityAuditError(f"Frozen input SHA-256 mismatch: {name}")
        receipts[name] = {"path": spec["path"], "bytes": spec["bytes"], "sha256": digest}
    report_spec = protocol["frozen_inputs"]["reference_catalog_report"]
    report = json.loads((root / report_spec["path"]).read_text(encoding="utf-8"))
    if report.get("decision") != report_spec["required_decision"]:
        raise ObservabilityAuditError("Frozen reference-catalog report is not PASS")
    return receipts


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ObservabilityAuditError(f"Invalid JSONL at {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ObservabilityAuditError(f"Non-object JSONL row at {path}:{number}")
            rows.append(row)
    return rows


def join_frozen_pairs(protocol: dict[str, Any], *, root: Path = ROOT) -> list[dict[str, Any]]:
    pairs = _read_jsonl(root / protocol["frozen_inputs"]["target_reference_pairs"]["path"])
    targets = _read_jsonl(root / protocol["frozen_inputs"]["selected_targets"]["path"])
    if len(pairs) != 76 or len(targets) != 79:
        raise ObservabilityAuditError("Frozen pair/target row count changed")
    target_index: dict[tuple[object, ...], dict[str, Any]] = {}
    for target in targets:
        key = tuple(target.get(name) for name in PAIR_JOIN_KEY)
        if key in target_index:
            raise ObservabilityAuditError("Selected-target join key is not unique")
        target_index[key] = target
    joined: list[dict[str, Any]] = []
    seen: set[tuple[object, ...]] = set()
    for pair in pairs:
        key = tuple(pair.get(name) for name in PAIR_JOIN_KEY)
        if key in seen or key not in target_index:
            raise ObservabilityAuditError("Reference pair join is duplicate or unmatched")
        seen.add(key)
        target = target_index[key]
        joined.append({
            **pair,
            "representative_longitude": target["representative_longitude"] if "representative_longitude" in target else None,
            "representative_latitude": target["representative_latitude"] if "representative_latitude" in target else None,
        })
    # Coordinates are intentionally joined from the hash-bound source through the
    # reviewed reference primitive, rather than trusted from either pair file.
    ref_protocol = reference.load_protocol(root / "configs/mars_ghgsat_reference_catalog_protocol.json")
    source_rows = reference.load_source_pairs(ref_protocol, root=root)
    source_index = {
        tuple(row[name] for name in PAIR_JOIN_KEY): row for row in source_rows
    }
    for row in joined:
        source = source_index.get(tuple(row[name] for name in PAIR_JOIN_KEY))
        if source is None:
            raise ObservabilityAuditError("Joined pair has no hash-bound source coordinate")
        row["representative_longitude"] = source["representative_longitude"]
        row["representative_latitude"] = source["representative_latitude"]
    if Counter(row["target_sensor"] for row in joined) != Counter({S2_SENSOR: 47, LANDSAT_SENSOR: 29}):
        raise ObservabilityAuditError("Frozen sensor pair counts changed")
    return joined


def assert_ignored(root: Path, path: Path) -> None:
    import subprocess
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    result = subprocess.run(["git", "check-ignore", "--quiet", "--", relative], cwd=root)
    if result.returncode != 0:
        raise ObservabilityAuditError(f"Bulk output is not git-ignored: {relative}")


def validation_plan(*, root: Path = ROOT) -> dict[str, object]:
    protocol = load_protocol()
    receipts = validate_frozen_inputs(protocol, root=root)
    rows = join_frozen_pairs(protocol, root=root)
    return {
        "mode": "validation_only",
        "protocol": {"path": PROTOCOL_RELATIVE_PATH, "sha256": EXPECTED_PROTOCOL_SHA256},
        "frozen_inputs": receipts,
        "pairs_joined": len(rows),
        "pairs_by_sensor": dict(sorted(Counter(str(r["target_sensor"]) for r in rows).items())),
        "network_client_created": False,
        "network_executed": False,
        "asset_url_observed": False,
        "remote_raster_opened": False,
        "raster_bytes_accessed": False,
        "cloudsen_model_loaded": False,
        "model_weights_opened_as_model": False,
        "dense_mask_created": False,
        "outputs_written": False,
    }


def parse_s2_physical(item_id: str) -> tuple[str, str, str]:
    match = re.fullmatch(
        r"(S2[AB])_MSIL1C_(\d{8}T\d{6})_N\d{4}_R\d{3}_T(\d{2}[A-Z]{3})_\d{8}T\d{6}(?:\.SAFE)?",
        item_id,
    )
    if match is None:
        raise ObservabilityAuditError(f"Invalid Sentinel-2 L1C product identity: {item_id}")
    return match.group(1), match.group(2), match.group(3)


def _s2_time(value: str) -> Any:
    return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def validate_s2_crosswalk_item(
    item: dict[str, Any], cdse_item_id: str, *, longitude: float, latitude: float
) -> dict[str, Any] | None:
    spacecraft, sensing, mgrs = parse_s2_physical(cdse_item_id)
    if item.get("type") != "Feature" or item.get("collection") != "sentinel-2-l1c":
        return None
    props = item.get("properties")
    if not isinstance(props, dict):
        raise ObservabilityAuditError("Earth Search item properties are malformed")
    product_uri = props.get("s2:product_uri")
    if not isinstance(product_uri, str):
        raise ObservabilityAuditError("Earth Search item lacks s2:product_uri")
    try:
        mirror_spacecraft, mirror_sensing, mirror_mgrs = parse_s2_physical(product_uri)
    except ObservabilityAuditError:
        return None
    if (mirror_spacecraft, mirror_mgrs) != (spacecraft, mgrs):
        return None
    if abs((_s2_time(mirror_sensing) - _s2_time(sensing)).total_seconds()) > 1:
        return None
    geometry = item.get("geometry")
    if not isinstance(geometry, dict) or not reference.geometry_covers_point(geometry, longitude, latitude):
        return None
    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise ObservabilityAuditError("Earth Search item ID is malformed")
    assets = item.get("assets")
    if not isinstance(assets, dict):
        raise ObservabilityAuditError("Earth Search item assets are malformed")
    return {"item_id": item_id, "cdse_product_uri": cdse_item_id,
            "mirror_product_uri": product_uri, "mgrs_tile": mgrs,
            "geometry": geometry, "assets": assets}


def select_s2_crosswalk(
    items: Iterable[dict[str, Any]], cdse_item_id: str, *, longitude: float, latitude: float
) -> dict[str, Any]:
    candidates = [
        value for item in items
        if (value := validate_s2_crosswalk_item(item, cdse_item_id, longitude=longitude, latitude=latitude)) is not None
    ]
    if not candidates:
        raise ObservabilityAuditError(f"No exact Earth Search mirror for {cdse_item_id}")
    return min(candidates, key=lambda value: value["item_id"])


def _asset_href(asset: object, *, name: str) -> str:
    if not isinstance(asset, dict) or not isinstance(asset.get("href"), str):
        raise ObservabilityAuditError(f"Required asset is missing: {name}")
    href = asset["href"]
    if "?" in href or "#" in href:
        raise ObservabilityAuditError(f"Signed or fragment asset URL forbidden: {name}")
    return href


def validate_s2_assets(record: dict[str, Any]) -> dict[str, str]:
    assets = record["assets"]
    aliases = {"B02": ("blue", "B02"), "B03": ("green", "B03"), "B04": ("red", "B04"),
               "B08": ("nir", "B08"), "B11": ("swir16", "B11"), "B12": ("swir22", "B12"),
               "tileinfo": ("tileinfo_metadata", "tileinfo")}
    resolved: dict[str, str] = {}
    for output, choices in aliases.items():
        key = next((choice for choice in choices if choice in assets), None)
        if key is None:
            raise ObservabilityAuditError(f"Required Sentinel asset role missing: {output}")
        href = _asset_href(assets[key], name=output)
        if not href.startswith(S2_PREFIX):
            raise ObservabilityAuditError("Sentinel asset is outside official public L1C root")
        resolved[output] = href
    suffixes = {"B02": "/B02.jp2", "B03": "/B03.jp2", "B04": "/B04.jp2", "B08": "/B08.jp2",
                "B11": "/B11.jp2", "B12": "/B12.jp2", "tileinfo": "/tileInfo.json"}
    roots = set()
    for name, suffix in suffixes.items():
        if not resolved[name].endswith(suffix):
            raise ObservabilityAuditError(f"Unexpected official Sentinel asset suffix: {name}")
        roots.add(resolved[name][:-len(suffix)])
    if len(roots) != 1:
        raise ObservabilityAuditError("Sentinel assets do not share one tile root")
    root = roots.pop()
    resolved.update({band: f"{root}/{band}.jp2" for band in S2_EXTRA_BANDS})
    return resolved


def validate_tileinfo(tileinfo: object, record: dict[str, Any]) -> None:
    if not isinstance(tileinfo, dict):
        raise ObservabilityAuditError("tileInfo metadata is malformed")
    expected_product = str(record["mirror_product_uri"]).removesuffix(".SAFE")
    if tileinfo.get("productName") != expected_product:
        raise ObservabilityAuditError("tileInfo productName mismatch")
    tile = f"{tileinfo.get('utmZone')}{tileinfo.get('latitudeBand')}{tileinfo.get('gridSquare')}"
    if tile != record["mgrs_tile"]:
        raise ObservabilityAuditError("tileInfo MGRS tile mismatch")
    expected_sensing = parse_s2_physical(record["mirror_product_uri"])[1]
    observed = tileinfo.get("timestamp") or tileinfo.get("sensingTime")
    if not isinstance(observed, str):
        raise ObservabilityAuditError("tileInfo sensing time missing")
    parsed = reference.parse_item_utc(observed)
    if abs((parsed - _s2_time(expected_sensing)).total_seconds()) > 1:
        raise ObservabilityAuditError("tileInfo sensing time is a different acquisition")


def validate_landsat_item(item: dict[str, Any], item_id: str) -> dict[str, Any]:
    if item.get("type") != "Feature" or item.get("id") != item_id or item.get("collection") != "landsat-c2l1":
        raise ObservabilityAuditError("USGS item identity/collection mismatch")
    if re.fullmatch(r"LC0[89]_L1TP_\d{6}_\d{8}_\d{8}_02_T1", item_id) is None:
        raise ObservabilityAuditError("Landsat item is not exact C2 L1TP T1")
    props = item.get("properties")
    if not isinstance(props, dict) or props.get("landsat:correction") != "L1TP" or props.get("landsat:collection_category") != "T1":
        raise ObservabilityAuditError("USGS Landsat processing level/tier mismatch")
    assets = item.get("assets")
    if not isinstance(assets, dict):
        raise ObservabilityAuditError("USGS item assets are malformed")
    # LandsatLook exposes semantic STAC roles while Collection-2 filenames use
    # unpadded band numbers. Accept the frozen semantic roles plus exact band
    # aliases, then bind every href to its physical filename below.
    aliases = {
        "B02": ("blue", "B02", "B2"),
        "B03": ("green", "B03", "B3"),
        "B04": ("red", "B04", "B4"),
        "B05": ("nir08", "B05", "B5"),
        "B06": ("swir16", "B06", "B6"),
        "B07": ("swir22", "B07", "B7"),
        "QA_PIXEL": ("qa_pixel", "QA_PIXEL"),
    }
    resolved: dict[str, str] = {}
    for output, choices in aliases.items():
        key = next((choice for choice in choices if choice in assets), None)
        if key is None:
            raise ObservabilityAuditError(f"Required Landsat asset missing: {output}")
        href = _asset_href(assets[key], name=output)
        if not href.startswith(LANDSAT_PREFIX):
            raise ObservabilityAuditError("Landsat asset is outside official C2 L1 root")
        resolved[output] = href
    suffixes = {
        "B02": "_B2.TIF", "B03": "_B3.TIF", "B04": "_B4.TIF",
        "B05": "_B5.TIF", "B06": "_B6.TIF", "B07": "_B7.TIF",
        "QA_PIXEL": "_QA_PIXEL.TIF",
    }
    roots: set[str] = set()
    for name, suffix in suffixes.items():
        expected_suffix = f"/{item_id}{suffix}"
        if not resolved[name].endswith(expected_suffix):
            raise ObservabilityAuditError(f"Landsat asset is not bound to exact item: {name}")
        roots.add(resolved[name][:-len(expected_suffix)])
    if len(roots) != 1:
        raise ObservabilityAuditError("Landsat assets do not share one exact product root")
    geometry = item.get("geometry")
    if not isinstance(geometry, dict):
        raise ObservabilityAuditError("USGS item geometry is malformed")
    return {"item_id": item_id, "geometry": geometry, "assets": resolved}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_bytes(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for row in rows:
            stream.write(canonical_bytes(row) + b"\n")
    os.replace(temporary, path)


class MetadataClient:
    """Strict capped metadata client with append-only hash-bound resume receipts."""

    def __init__(self, protocol: dict[str, Any], request_log: Path, response_log: Path,
                 *, session: Any, sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.contract = protocol["asset_resolution_network_contract"]
        self.request_log, self.response_log = request_log, response_log
        self.session, self.sleep, self.monotonic = session, sleep, monotonic
        self.last_request: float | None = None
        self.total_bytes = 0
        self.resume: dict[str, dict[str, Any]] = {}
        self.historical_attempts: Counter[str] = Counter()
        request_ids: set[str] = set()
        for request in _read_jsonl(request_log) if request_log.exists() else []:
            identity = request.get("request_sha256")
            rebuilt = {
                "method": request.get("method"),
                "endpoint": request.get("endpoint"),
                "body": request.get("body"),
            }
            if (
                request.get("schema_version") != 1
                or type(request.get("attempt")) is not int
                or int(request["attempt"]) < 1
                or not isinstance(identity, str)
                or identity != sha256_value(rebuilt)
            ):
                raise ObservabilityAuditError("Metadata request resume receipt is malformed")
            request_ids.add(identity)
            self.historical_attempts[identity] += 1
            if (
                self.historical_attempts[identity] > self.contract["maximum_attempts_per_request"]
                or int(request["attempt"]) > self.contract["maximum_attempts_per_request"]
            ):
                raise ObservabilityAuditError("Historical metadata request attempt cap exceeded")
        for receipt in _read_jsonl(response_log) if response_log.exists() else []:
            observed = receipt.get("response_bytes")
            digest = receipt.get("response_sha256")
            if (
                receipt.get("schema_version") != 1
                or type(observed) is not int or observed < 0
                or observed > self.contract["metadata_response_maximum_bytes_each"]
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or type(receipt.get("body_read_complete")) is not bool
                or type(receipt.get("accepted")) is not bool
            ):
                raise ObservabilityAuditError("Metadata response resume receipt is malformed")
            if receipt.get("accepted") is True:
                identity = receipt.get("request_sha256")
                payload = receipt.get("sanitized_payload")
                if (
                    receipt.get("schema_version") != 1 or receipt.get("status") != 200
                    or receipt.get("body_read_complete") is not True
                    or not isinstance(identity, str) or identity not in request_ids
                    or sha256_value(payload) != receipt.get("payload_sha256")
                    or receipt.get("checkpoint_sha256") != sha256_value({k: receipt[k] for k in receipt if k != "checkpoint_sha256"})
                ):
                    raise ObservabilityAuditError("Accepted metadata resume receipt is malformed")
                if identity in self.resume:
                    raise ObservabilityAuditError("Duplicate accepted metadata receipt")
                self.resume[identity] = receipt
            self.total_bytes += observed
        if self.total_bytes > self.contract["metadata_response_maximum_bytes_total"]:
            raise ObservabilityAuditError("Historical metadata byte cap exceeded")

    def _pace(self) -> None:
        if self.last_request is not None:
            remaining = self.contract["minimum_request_interval_seconds"] - (self.monotonic() - self.last_request)
            if remaining > 0:
                self.sleep(remaining)
        self.last_request = self.monotonic()

    def json(self, method: str, endpoint: str, *, body: object | None = None) -> object:
        request = {"method": method, "endpoint": endpoint, "body": body}
        identity = sha256_value(request)
        if identity in self.resume:
            return self.resume[identity]["sanitized_payload"]
        retryable = set(self.contract["retryable_http_statuses"])
        prior_attempts = self.historical_attempts[identity]
        maximum_attempts = self.contract["maximum_attempts_per_request"]
        if prior_attempts >= maximum_attempts:
            raise ObservabilityAuditError("Metadata request exhausted its frozen attempt cap")
        for attempt in range(prior_attempts + 1, maximum_attempts + 1):
            request_receipt = {"schema_version": 1, "attempt": attempt, **request, "request_sha256": identity}
            _append_jsonl(self.request_log, request_receipt)
            self._pace()
            kwargs = {"allow_redirects": False, "stream": True, "timeout": (15, 90),
                      "headers": {"Accept": "application/geo+json, application/json", "Accept-Encoding": "identity",
                                  "User-Agent": "ERSRR-GHGSat-windowed-observability/1.0"}}
            response = self.session.request(method, endpoint, json=body, **kwargs)
            try:
                status = int(response.status_code)
                if getattr(response, "history", []) or str(getattr(response, "url", endpoint)) != endpoint:
                    raise ObservabilityAuditError("Metadata redirect/endpoint change rejected")
                maximum = self.contract["metadata_response_maximum_bytes_each"]
                chunks: list[bytes] = []
                length = 0
                complete = True
                for chunk in response.iter_content(65536):
                    length += len(chunk)
                    if length > maximum or self.total_bytes + length > self.contract["metadata_response_maximum_bytes_total"]:
                        complete = False
                        break
                    chunks.append(chunk)
                raw = b"".join(chunks)
                self.total_bytes += length
                receipt: dict[str, Any] = {"schema_version": 1, "request_sha256": identity,
                    "status": status, "response_bytes": length, "response_sha256": hashlib.sha256(raw).hexdigest(),
                    "body_read_complete": complete, "accepted": False}
                if not complete:
                    _append_jsonl(self.response_log, receipt)
                    raise ObservabilityAuditError("Metadata response byte cap exceeded")
                if status != 200:
                    _append_jsonl(self.response_log, receipt)
                    if status in retryable and attempt < self.contract["maximum_attempts_per_request"]:
                        self.sleep(min(2 ** (attempt - 1), 8)); continue
                    raise ObservabilityAuditError(f"Metadata HTTP status rejected: {status}")
                if "json" not in str(response.headers.get("Content-Type", "")).lower():
                    _append_jsonl(self.response_log, receipt)
                    raise ObservabilityAuditError("Metadata response is not JSON")
                payload = json.loads(raw)
                receipt.update({"accepted": True, "sanitized_payload": payload, "payload_sha256": sha256_value(payload)})
                receipt["checkpoint_sha256"] = sha256_value(receipt)
                _append_jsonl(self.response_log, receipt)
                return payload
            finally:
                response.close()
        raise AssertionError("unreachable metadata retry loop")


def _s2_search_body(item_id: str, longitude: float, latitude: float) -> dict[str, Any]:
    _, sensing, _ = parse_s2_physical(item_id)
    center = _s2_time(sensing)
    start = (center.timestamp() - 1)
    end = (center.timestamp() + 1)
    return {"collections": ["sentinel-2-l1c"], "datetime": f"{datetime.fromtimestamp(start, timezone.utc).isoformat().replace('+00:00','Z')}/{datetime.fromtimestamp(end, timezone.utc).isoformat().replace('+00:00','Z')}",
            "intersects": {"type": "Point", "coordinates": [longitude, latitude]}, "limit": 100}


def _resolution_failure(row: dict[str, Any], missing: list[tuple[str, Exception]]) -> dict[str, Any]:
    """Return an ignored, URL-free record for a non-resolvable frozen pair."""
    return {
        "schema_version": 1,
        "resolution_status": "unavailable_or_invalid",
        **{name: row[name] for name in (*PAIR_JOIN_KEY, "component_id", "reference_item_id")},
        "failed_sides": [
            {"side": side, "error_type": type(exc).__name__}
            for side, exc in missing
        ],
    }


def resolve_assets(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    client: MetadataClient,
    *,
    continue_on_pair_failure: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    centers: dict[tuple[object, ...], set[tuple[float, float]]] = {}
    aliases: dict[str, tuple[object, ...]] = {}
    for row in rows:
        for item_id in (row["target_item_id"], row["reference_item_id"]):
            key: tuple[object, ...] = (
                ("sentinel-2-l1c", *parse_s2_physical(item_id))
                if item_id.startswith(("S2A_", "S2B_"))
                else ("landsat-c2l1", item_id)
            )
            aliases[item_id] = key
            center = (float(row["representative_longitude"]), float(row["representative_latitude"]))
            centers.setdefault(key, set()).add(center)
    resolved_physical: dict[tuple[object, ...], dict[str, Any]] = {}
    failed_physical: dict[tuple[object, ...], Exception] = {}
    item_ids_by_key: dict[tuple[object, ...], list[str]] = {}
    for item_id, key in aliases.items():
        item_ids_by_key.setdefault(key, []).append(item_id)
    for physical_key in sorted(centers, key=lambda value: tuple(str(part) for part in value)):
        item_id = min(item_ids_by_key[physical_key])
        # Exactly one metadata resolution per unique physical acquisition.  A
        # shared acquisition may serve multiple source points; every point is
        # checked against the returned full geometry below.
        longitude, latitude = sorted(centers[physical_key])[0]
        try:
            if item_id.startswith(("S2A_", "S2B_")):
                payload = client.json("POST", protocol["sentinel_2_l1c_resolution"]["search_endpoint"],
                                      body=_s2_search_body(item_id, longitude, latitude))
                if not isinstance(payload, dict) or not isinstance(payload.get("features"), list) or len(payload["features"]) >= 100:
                    raise ObservabilityAuditError("Earth Search crosswalk response malformed or truncated")
                record = select_s2_crosswalk(payload["features"], item_id, longitude=longitude, latitude=latitude)
                record["assets"] = validate_s2_assets(record)
                tileinfo = client.json("GET", record["assets"]["tileinfo"])
                validate_tileinfo(tileinfo, record)
                record["sensor"] = S2_SENSOR
            else:
                endpoint = protocol["landsat_8_9_level_1_resolution"]["item_endpoint_template"].format(item_id=quote(item_id, safe=""))
                payload = client.json("GET", endpoint)
                if not isinstance(payload, dict):
                    raise ObservabilityAuditError("USGS item response is malformed")
                record = validate_landsat_item(payload, item_id)
                record["sensor"] = LANDSAT_SENSOR
            if any(
                not reference.geometry_covers_point(record["geometry"], x, y)
                for x, y in centers[physical_key]
            ):
                raise ObservabilityAuditError("Resolved physical acquisition does not cover every frozen source point")
            resolved_physical[physical_key] = record
        except Exception as exc:
            if not continue_on_pair_failure:
                raise
            failed_physical[physical_key] = exc

    resolved_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    for row in rows:
        target_key = aliases[row["target_item_id"]]
        reference_key = aliases[row["reference_item_id"]]
        missing = [
            (side, failed_physical[key])
            for side, key in (("target", target_key), ("reference", reference_key))
            if key in failed_physical
        ]
        if missing:
            failed_rows.append(_resolution_failure(row, missing))
            continue
        resolved_rows.append({
            **row,
            "target": resolved_physical[target_key],
            "reference": resolved_physical[reference_key],
        })
    if continue_on_pair_failure:
        return resolved_rows, failed_rows
    if failed_rows:
        raise ObservabilityAuditError("Strict asset resolution dropped a frozen pair")
    return resolved_rows


def gdal_env() -> dict[str, str]:
    return {"AWS_NO_SIGN_REQUEST": "YES", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".jp2,.tif,.tiff", "GDAL_HTTP_MAX_RETRY": "5",
            "GDAL_HTTP_RETRY_DELAY": "1", "VSI_CACHE": "TRUE", "VSI_CACHE_SIZE": str(64 * 1024 * 1024)}


def _window_for_grid(source: Any, dst_crs: Any, dst_transform: Any, *, padding: int) -> Any:
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds
    bounds = rasterio.transform.array_bounds(SIZE, SIZE, dst_transform)
    if source.crs != dst_crs:
        bounds = transform_bounds(dst_crs, source.crs, *bounds, densify_pts=21)
    fractional = from_bounds(*bounds, transform=source.transform)
    c0 = max(0, math.floor(fractional.col_off) - padding)
    r0 = max(0, math.floor(fractional.row_off) - padding)
    c1 = min(source.width, math.ceil(fractional.col_off + fractional.width) + padding)
    r1 = min(source.height, math.ceil(fractional.row_off + fractional.height) + padding)
    if c1 <= c0 or r1 <= r0:
        raise ObservabilityAuditError("Crop does not overlap remote raster")
    return Window(c0, r0, c1 - c0, r1 - r0)


def read_window_to_grid(url: str, *, dst_crs: Any, dst_transform: Any, resampling: Any, dtype: str) -> np.ndarray:
    """Read only the source window needed by the frozen 200x200 destination."""
    import rasterio
    from rasterio.warp import reproject
    padding = 1 if str(resampling).lower().endswith("bilinear") or int(resampling) == 1 else 0
    with rasterio.Env(**gdal_env()), rasterio.open(url) as source:
        window = _window_for_grid(source, dst_crs, dst_transform, padding=padding)
        source_data = source.read(1, window=window, boundless=False, out_dtype=dtype)
        destination = np.zeros((SIZE, SIZE), dtype=np.dtype(dtype))
        reproject(source=source_data, destination=destination,
                  src_transform=source.window_transform(window), src_crs=source.crs,
                  src_nodata=source.nodata if source.nodata is not None else 0,
                  dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=0, resampling=resampling)
    return destination


def target_grid(url: str, longitude: float, latitude: float, sensor: str) -> tuple[Any, Any]:
    import rasterio
    from affine import Affine
    from rasterio.warp import transform as transform_points
    with rasterio.Env(**gdal_env()), rasterio.open(url) as source:
        if source.crs is None:
            raise ObservabilityAuditError("Target B02 has no CRS")
        xs, ys = transform_points("EPSG:4326", source.crs, [longitude], [latitude])
        if sensor == S2_SENSOR:
            col, row = (~source.transform) * (xs[0], ys[0])
            c0, r0 = math.floor(col) - SIZE // 2, math.floor(row) - SIZE // 2
            if c0 < 0 or r0 < 0 or c0 + SIZE > source.width or r0 + SIZE > source.height:
                raise ObservabilityAuditError("Native Sentinel target window leaves raster extent")
            transform = source.window_transform(__import__("rasterio.windows", fromlist=["Window"]).Window(c0, r0, SIZE, SIZE))
            if abs(abs(transform.a) - RESOLUTION) > 1e-6 or abs(abs(transform.e) - RESOLUTION) > 1e-6:
                raise ObservabilityAuditError("Sentinel target B02 is not native 10 m")
            return source.crs, transform
        half = SIZE * RESOLUTION / 2
        # The frozen Landsat grid is north-up and exactly centered on the
        # source point; unlike the Sentinel native grid it is not pixel-snapped.
        left = xs[0] - half
        top = ys[0] + half
        return source.crs, Affine(RESOLUTION, 0, left, 0, -RESOLUTION, top)


def training_stack(assets: dict[str, str], sensor: str, *, crs: Any, transform: Any) -> np.ndarray:
    from rasterio.enums import Resampling
    bands = S2_TRAINING_BANDS if sensor == S2_SENSOR else LANDSAT_TRAINING_BANDS
    values = []
    for band in bands:
        producer_interpolation = sensor == S2_SENSOR and band in {"B11", "B12"}
        resampling = Resampling.nearest if sensor == S2_SENSOR else Resampling.bilinear
        value = read_window_to_grid(assets[band], dst_crs=crs, dst_transform=transform,
                                    resampling=resampling, dtype="uint16")
        if producer_interpolation:
            # Reviewed MARS contract: nearest acquisition on the target grid,
            # nearest down to the native 20 m lattice, then bilinear back to 10 m.
            try:
                from tools.acquire_unep_mars_exact_crops import interpolate_s2ee_20m_bands
            except ModuleNotFoundError:
                from acquire_unep_mars_exact_crops import interpolate_s2ee_20m_bands
            value = interpolate_s2ee_20m_bands(value)
        values.append(value)
    stack = np.stack(values)
    if stack.shape != (6, SIZE, SIZE) or stack.dtype != np.uint16:
        raise ObservabilityAuditError("Training crop is not six-band 200x200 uint16")
    return stack


def radiometric_valid_fraction(target: np.ndarray, reference_frame: np.ndarray) -> float:
    if target.shape != (6, SIZE, SIZE) or reference_frame.shape != target.shape:
        raise ObservabilityAuditError("Radiometry gate requires two six-band 200x200 frames")
    valid = np.all(np.isfinite(target), axis=0) & np.all(target != 0, axis=0)
    valid &= np.all(np.isfinite(reference_frame), axis=0) & np.all(reference_frame != 0, axis=0)
    return float(np.mean(valid))


def landsat_union_clear(target_qa: np.ndarray, reference_qa: np.ndarray) -> tuple[np.ndarray, float]:
    if target_qa.shape != (SIZE, SIZE) or reference_qa.shape != target_qa.shape:
        raise ObservabilityAuditError("QA gate requires two 200x200 windows")
    bits = sum(1 << bit for bit in range(6))
    nonclear = ((target_qa.astype(np.uint16) | reference_qa.astype(np.uint16)) & bits) != 0
    return nonclear, float(np.mean(~nonclear))


def cloudsen_union_clear(target_prediction: np.ndarray, reference_prediction: np.ndarray,
                         target_stack: np.ndarray, reference_stack: np.ndarray) -> tuple[np.ndarray, float]:
    if target_prediction.shape != (SIZE, SIZE) or reference_prediction.shape != target_prediction.shape:
        raise ObservabilityAuditError("CloudSEN prediction shape mismatch")
    allowed = {0, 1, 2, 3}
    if not set(np.unique(target_prediction)).issubset(allowed) or not set(np.unique(reference_prediction)).issubset(allowed):
        raise ObservabilityAuditError("CloudSEN prediction classes changed")
    invalid = np.any(target_stack == 0, axis=0) | np.any(reference_stack == 0, axis=0)
    nonclear = (target_prediction != 0) | (reference_prediction != 0) | invalid
    return nonclear, float(np.mean(~nonclear))


def observability_gate(radiometric_fraction: float, union_clear_fraction: float) -> bool:
    return radiometric_fraction >= MIN_FRACTION and union_clear_fraction >= MIN_FRACTION


def evaluate_final_gates(rows: list[dict[str, Any]], *, all_valid: bool = True) -> tuple[dict[str, Any], dict[str, int]]:
    observations = {tuple(row[name] for name in OBSERVATION_KEY) for row in rows}
    sites = {str(row["site_ID"]) for row in rows}
    components = {str(row["component_id"]) for row in rows}
    counts = {"source_sensor_pairs": len(rows), "distinct_source_observations": len(observations),
              "distinct_sites": len(sites), "distinct_components": len(components)}
    observed = {"minimum_locally_observable_source_sensor_pairs": counts["source_sensor_pairs"],
                "minimum_distinct_locally_observable_source_observations": counts["distinct_source_observations"],
                "minimum_distinct_sites": counts["distinct_sites"],
                "minimum_novel_25km_components": counts["distinct_components"],
                "all_retained_assets_products_and_crops_valid": all_valid,
                "no_dense_supervision_created": True}
    required = load_protocol()["gates"]
    gates = {name: {"observed": observed[name], "required": threshold,
                    "pass": observed[name] is True if threshold is True else int(observed[name]) >= threshold}
             for name, threshold in required.items()}
    return gates, counts


def write_crop(path: Path, data: np.ndarray, *, crs: Any, transform: Any, descriptions: tuple[str, ...]) -> dict[str, Any]:
    import rasterio
    if data.shape != (6, SIZE, SIZE) or data.dtype != np.uint16:
        raise ObservabilityAuditError("Only six-band uint16 scene crops may be written")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with rasterio.open(temporary, "w", driver="GTiff", width=SIZE, height=SIZE, count=6,
                       dtype="uint16", crs=crs, transform=transform, nodata=0, compress="deflate",
                       predictor=2, tiled=True, blockxsize=128, blockysize=128) as output:
        output.write(data)
        for index, name in enumerate(descriptions, 1):
            output.set_band_description(index, name)
    os.replace(temporary, path)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_cloudsen_model(weights_path: Path) -> Any:
    if sha256_file(weights_path) != "218fa69aa3c7212d4e690b48af88ac6f3c976fc50d07f275b8fd623909183d7a":
        raise ObservabilityAuditError("CloudSEN12 weight SHA-256 mismatch")
    try:
        import torch
        from cloudsen12_models import cloudsen12
    except ImportError as exc:
        raise ObservabilityAuditError("cloudsen12-models and torch are required for network execution") from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cloudsen12.load_model_by_name("UNetMobV2_V2", weights_folder=str(weights_path.parent), device=device)
    if tuple(model.bands) != CLOUDSEN_BANDS:
        raise ObservabilityAuditError("CloudSEN12 model band contract changed")
    return model


def _cloudsen_stack(assets: dict[str, str], training: np.ndarray, *, crs: Any, transform: Any) -> np.ndarray:
    from rasterio.enums import Resampling
    local = {band: training[index] for index, band in enumerate(S2_TRAINING_BANDS)}
    extras = {band: read_window_to_grid(assets[band], dst_crs=crs, dst_transform=transform,
                                        resampling=Resampling.bilinear, dtype="uint16") for band in S2_EXTRA_BANDS}
    return np.stack([{**local, **extras}[band] for band in CLOUDSEN_BANDS])


def process_pair(root: Path, crop_root: Path, row: dict[str, Any], model: Any | None) -> dict[str, Any] | None:
    identity = sha256_value({name: row[name] for name in PAIR_JOIN_KEY})[:24]
    scene_dir = crop_root / identity
    try:
        # A retained directory may contain only the two crops created by this
        # execution.  This removes stale masks or superseded partial products.
        shutil.rmtree(scene_dir, ignore_errors=True)
        sensor = row["target_sensor"]
        target_assets, reference_assets = row["target"]["assets"], row["reference"]["assets"]
        crs, transform = target_grid(target_assets["B02"], float(row["representative_longitude"]),
                                     float(row["representative_latitude"]), sensor)
        target = training_stack(target_assets, sensor, crs=crs, transform=transform)
        reference_frame = training_stack(reference_assets, sensor, crs=crs, transform=transform)
        radiometric = radiometric_valid_fraction(target, reference_frame)
        if sensor == LANDSAT_SENSOR:
            from rasterio.enums import Resampling
            target_qa = read_window_to_grid(target_assets["QA_PIXEL"], dst_crs=crs, dst_transform=transform,
                                            resampling=Resampling.nearest, dtype="uint16")
            reference_qa = read_window_to_grid(reference_assets["QA_PIXEL"], dst_crs=crs, dst_transform=transform,
                                               resampling=Resampling.nearest, dtype="uint16")
            _, clear = landsat_union_clear(target_qa, reference_qa)
        else:
            if model is None:
                raise ObservabilityAuditError("CloudSEN12 model is required for Sentinel pairs")
            target_13 = _cloudsen_stack(target_assets, target, crs=crs, transform=transform)
            reference_13 = _cloudsen_stack(reference_assets, reference_frame, crs=crs, transform=transform)
            target_pred = np.asarray(model.predict(target_13.astype(np.float32) / 10000.0), dtype=np.uint8)
            reference_pred = np.asarray(model.predict(reference_13.astype(np.float32) / 10000.0), dtype=np.uint8)
            _, clear = cloudsen_union_clear(target_pred, reference_pred, target_13, reference_13)
        if not observability_gate(radiometric, clear):
            shutil.rmtree(scene_dir, ignore_errors=True)
            return None
        bands = S2_TRAINING_BANDS if sensor == S2_SENSOR else LANDSAT_TRAINING_BANDS
        target_record = write_crop(scene_dir / "target.tif", target, crs=crs, transform=transform, descriptions=bands)
        reference_record = write_crop(scene_dir / "reference.tif", reference_frame, crs=crs, transform=transform,
                                      descriptions=tuple(f"{band}_reference" for band in bands))
        return {"schema_version": 1, "scene_presence_label": 0, "supervision": "scene_head_only",
                "dense_mask": None, "dense_pixel_loss_authorized": False,
                "license": "CC BY-NC-SA 4.0",
                **{name: row[name] for name in (*PAIR_JOIN_KEY, "component_id", "reference_item_id")},
                "radiometric_valid_fraction": round(radiometric, 8), "union_clear_fraction": round(clear, 8),
                "assets": {"target": target_record, "reference": reference_record}}
    except Exception:
        shutil.rmtree(scene_dir, ignore_errors=True)
        raise


def _compact_asset_row(row: dict[str, Any]) -> dict[str, Any]:
    return row  # ignored manifest is the only location where URLs are retained


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _execute_network_audit(*, root: Path = ROOT, session: Any | None = None,
                           sleep: Callable[[float], None] = time.sleep,
                           monotonic: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    protocol = load_protocol()
    validate_frozen_inputs(protocol, root=root)
    rows = join_frozen_pairs(protocol, root=root)
    outputs = protocol["outputs"]
    ignored_root = root / outputs["ignored_root"]
    crop_root = root / outputs["ignored_crop_root"]
    for path in (ignored_root, crop_root):
        assert_ignored(root, path)
    own_session = session is None
    if own_session:
        import requests
        session = requests.Session()
    try:
        client = MetadataClient(protocol, root / outputs["ignored_request_receipts"],
                                root / outputs["ignored_response_receipts"], session=session,
                                sleep=sleep, monotonic=monotonic)
    except Exception:
        if own_session:
            session.close()
        raise
    asset_manifest = root / outputs["ignored_asset_manifest"]
    observable_manifest = root / outputs["ignored_observable_manifest"]
    try:
        resolved_result = resolve_assets(rows, protocol, client, continue_on_pair_failure=True)
        if not isinstance(resolved_result, tuple):
            raise AssertionError("failure-tolerant resolution did not return outcomes")
        resolved, resolution_failures = resolved_result
        # Rebuild the crop root from nothing. Unknown/stale files (especially a
        # dense or zero mask from another workflow) can never survive into a
        # passing cohort.
        shutil.rmtree(crop_root, ignore_errors=True)
        crop_root.mkdir(parents=True, exist_ok=True)
        weights = root / protocol["frozen_inputs"]["cloudsen12_weights"]["path"]
        model = load_cloudsen_model(weights)
        observable: list[dict[str, Any]] = []
        local_failures: list[dict[str, Any]] = []
        asset_outcomes: list[dict[str, Any]] = []
        for row in resolved:
            try:
                retained = process_pair(root, crop_root, row, model)
            except Exception as exc:
                local_failures.append({
                    "schema_version": 1,
                    "local_status": "processing_error",
                    **{name: row[name] for name in (*PAIR_JOIN_KEY, "component_id", "reference_item_id")},
                    "error_type": type(exc).__name__,
                })
                asset_outcomes.append({
                    **row, "local_status": "processing_error",
                    "local_error_type": type(exc).__name__,
                })
                continue
            if retained is not None:
                observable.append(retained)
                asset_outcomes.append({**row, "local_status": "observable"})
            else:
                asset_outcomes.append({**row, "local_status": "below_observability_gate"})
        _write_jsonl(asset_manifest, (
            _compact_asset_row(row)
            for row in (*asset_outcomes, *resolution_failures)
        ))
        _write_jsonl(observable_manifest, observable)
        gates, counts = evaluate_final_gates(observable)
        report = {"schema_version": 1,
                  "decision": "PASS" if all(gate["pass"] for gate in gates.values()) else "FAIL",
                  "claim_boundary": protocol["claim_boundary"], "counts": counts, "gates": gates,
                  "frozen_pairs_attempted": 76, "asset_urls_retained_in_compact_report": False,
                  "pair_outcomes": {
                      "asset_resolution_failed": len(resolution_failures),
                      "local_processing_failed": len(local_failures),
                      "below_observability_gate": sum(
                          row["local_status"] == "below_observability_gate" for row in asset_outcomes
                      ),
                      "observable": len(observable),
                  },
                  "dense_mask_created": False,
                  "ignored_manifests": {"assets": {"path": outputs["ignored_asset_manifest"], "sha256": sha256_file(asset_manifest)},
                                        "observable": {"path": outputs["ignored_observable_manifest"], "sha256": sha256_file(observable_manifest)}}}
        _atomic_json(root / outputs["compact_json"], report)
        markdown = ["# GHGSat windowed observability audit", "", f"**Decision: {report['decision']}**", "",
                    f"- Frozen pairs attempted: 76", f"- Observable scene-head negatives: {len(observable)}",
                    "- Dense/zero masks created: False", "", protocol["claim_boundary"], ""]
        (root / outputs["compact_markdown"]).write_text("\n".join(markdown), encoding="utf-8")
        return report
    except Exception as exc:
        asset_manifest.unlink(missing_ok=True)
        observable_manifest.unlink(missing_ok=True)
        shutil.rmtree(crop_root, ignore_errors=True)
        failure = {"schema_version": 1, "decision": "FAIL", "error_type": type(exc).__name__,
                   "error": "Windowed observability execution failed; inspect ignored receipts locally.",
                   "stale_derived_outputs_removed": True, "dense_mask_created": False,
                   "claim_boundary": protocol["claim_boundary"]}
        _atomic_json(root / outputs["compact_json"], failure)
        (root / outputs["compact_markdown"]).write_text(
            "# GHGSat windowed observability audit\n\n**Decision: FAIL**\n\n"
            + type(exc).__name__ + "; inspect ignored receipts locally.\n", encoding="utf-8")
        if isinstance(exc, ObservabilityAuditError):
            raise
        raise ObservabilityAuditError(str(exc)) from exc
    finally:
        if own_session:
            session.close()


def execute_network_audit(*, root: Path = ROOT, session: Any | None = None,
                          sleep: Callable[[float], None] = time.sleep,
                          monotonic: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Guard preflight and execution so no stale PASS can survive any failure."""
    try:
        return _execute_network_audit(
            root=root, session=session, sleep=sleep, monotonic=monotonic
        )
    except Exception as exc:
        # Use the statically frozen paths even when the protocol itself cannot
        # be trusted or parsed. Append-only request/response receipts survive;
        # all derived URL/crop/cohort outputs are invalidated.
        for name in ("ignored_asset_manifest", "ignored_observable_manifest"):
            (root / EXPECTED_OUTPUTS[name]).unlink(missing_ok=True)
        shutil.rmtree(root / EXPECTED_OUTPUTS["ignored_crop_root"], ignore_errors=True)
        failure = {
            "schema_version": 1,
            "decision": "FAIL",
            "error_type": type(exc).__name__,
            "error": "Windowed observability execution failed; inspect ignored receipts locally.",
            "stale_derived_outputs_removed": True,
            "dense_mask_created": False,
        }
        _atomic_json(root / EXPECTED_OUTPUTS["compact_json"], failure)
        (root / EXPECTED_OUTPUTS["compact_markdown"]).write_text(
            "# GHGSat windowed observability audit\n\n**Decision: FAIL**\n\n"
            + type(exc).__name__ + "; inspect ignored receipts locally.\n",
            encoding="utf-8",
        )
        if isinstance(exc, ObservabilityAuditError):
            raise
        raise ObservabilityAuditError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-network", action="store_true",
                        help="resolve all frozen assets and execute all 76 windowed audits")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute_network_audit() if args.execute_network else validation_plan()
    except (ObservabilityAuditError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("decision", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
