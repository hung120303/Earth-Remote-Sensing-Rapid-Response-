#!/usr/bin/env python3
"""Freeze outcome-blind Stanford Sentinel-2 L1C target/reference pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
L1C_COLLECTION = "sentinel-2-l1c"
L1C_BANDS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
}
REQUIRED_ASSETS = set(L1C_BANDS.values()) | {"tileinfo_metadata"}
SCENE_ID = re.compile(r"^(S2[ABC])_(\d{2}[A-Z]{3})_(\d{8})_(\d+)_L1C$")
L1C_HTTPS_PREFIX = "https://sentinel-s2-l1c.s3.amazonaws.com/"
DEFAULT_SOURCE = Path(
    ".research/stanford_controlled_release_2024_2025/cohort_manifest.jsonl"
)
DEFAULT_PROTOCOL = Path("configs/stanford_large_controlled_release_protocol.json")
DEFAULT_MANIFEST = Path(
    ".research/stanford_controlled_release_2024_2025/l1c_stress/pair_manifest.json"
)
DEFAULT_JSON = Path(
    "reports/acquisition/stanford_large_controlled_release_l1c_pairs.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/acquisition/STANFORD_LARGE_CONTROLLED_RELEASE_L1C_PAIRS.md"
)
FORBIDDEN_OUTCOME_FIELDS = {
    "metered_ch4_kgh",
    "metered_ch4_sigma",
    "truth_stratum",
    "label",
    "labels",
    "label_state",
    "release_rate",
    "release_rate_kgh",
    "outcome",
    "outcomes",
    "prediction",
    "predictions",
    "detector_score",
    "model_score",
    "isplume",
    "is_plume",
}


def repo_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    ).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_repo_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path must resolve beneath repository root: {value}")
    return path


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Datetime must include a UTC offset: {value}")
    return parsed.astimezone(timezone.utc)


def assert_no_outcome_fields(value: Any, *, path: str = "$") -> None:
    """Fail before any pair/acquisition artifact can carry a source outcome field."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_OUTCOME_FIELDS:
                raise ValueError(f"Output contains forbidden outcome field at {path}.{key}")
            assert_no_outcome_fields(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_outcome_fields(child, path=f"{path}[{index}]")


def source_targets(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Return only acquisition metadata plus the global source-target exclusion set."""
    rows = list(records)
    excluded_target_ids = {
        str(target["id"])
        for row in rows
        for target in [row.get("target")]
        if isinstance(target, dict)
        and target.get("status") == "resolved"
        and target.get("id")
    }
    targets: list[dict[str, Any]] = []
    for row in rows:
        target = row.get("target")
        if row.get("sensor") != "Sentinel-2" or not isinstance(target, dict):
            continue
        if target.get("status") != "resolved":
            continue
        scene_id = str(target.get("id", ""))
        mgrs_tile(scene_id)
        longitude = float(row["longitude"])
        latitude = float(row["latitude"])
        if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
            raise ValueError(f"Invalid source coordinates for {row.get('release_id')}")
        targets.append(
            {
                "event_id": str(row["release_id"]),
                "observed_at_utc": parse_datetime(str(row["observed_at_utc"])).isoformat(),
                "center": [longitude, latitude],
                "target_scene_id": scene_id,
                "target_datetime": parse_datetime(str(target["datetime"])).isoformat(),
            }
        )
    targets.sort(key=lambda item: (item["target_datetime"], item["event_id"]))
    assert_no_outcome_fields(targets)
    return targets, excluded_target_ids


def request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.request(
                method,
                url,
                json=payload,
                timeout=90,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "ERSRR-research/1.0",
                },
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError(f"Expected object response from {url}")
            return result
        except (requests.RequestException, ValueError) as exc:
            error = exc
            if attempt < 3:
                time.sleep(float(2**attempt))
    assert error is not None
    raise error


def item_url(scene_id: str) -> str:
    return (
        "https://earth-search.aws.element84.com/v1/collections/"
        f"{L1C_COLLECTION}/items/{scene_id}"
    )


def mgrs_tile(scene_id: str) -> str:
    match = SCENE_ID.fullmatch(scene_id)
    if match is None:
        raise ValueError(f"Expected a Sentinel-2 L1C item ID, got {scene_id!r}")
    return match.group(2)


def contains_center(item: dict[str, Any], center: list[float]) -> bool:
    bbox = item.get("bbox")
    return bool(
        isinstance(bbox, list)
        and len(bbox) >= 4
        and float(bbox[0]) <= center[0] <= float(bbox[2])
        and float(bbox[1]) <= center[1] <= float(bbox[3])
    )


def official_l1c_url(catalog_href: str) -> tuple[str, bool]:
    prefixes = (
        "s3://sentinel-s2-l1c/",
        "s3://sentinel-s2-l2a/",
        "https://sentinel-s2-l1c.s3.amazonaws.com/",
        "https://sentinel-s2-l2a.s3.amazonaws.com/",
        "https://roda.sentinel-hub.com/sentinel-s2-l1c/",
    )
    for prefix in prefixes:
        if catalog_href.startswith(prefix):
            relative = catalog_href[len(prefix) :]
            corrected = "l2a" in prefix
            return L1C_HTTPS_PREFIX + relative, corrected
    raise ValueError(f"Unsupported Sentinel-2 L1C catalog href: {catalog_href}")


def validate_catalog_item(
    item: dict[str, Any],
    *,
    expected_id: str,
    center: list[float],
) -> None:
    if item.get("id") != expected_id:
        raise ValueError(f"STAC returned {item.get('id')!r}, expected {expected_id!r}")
    if item.get("collection") not in {None, L1C_COLLECTION}:
        raise ValueError(f"Scene {expected_id} is not in {L1C_COLLECTION}")
    mgrs_tile(expected_id)
    if not contains_center(item, center):
        raise ValueError(f"Scene {expected_id} does not cover the exact source coordinate")
    missing = sorted(REQUIRED_ASSETS - set(item.get("assets", {})))
    if missing:
        raise ValueError(f"Scene {expected_id} lacks required L1C assets: {missing}")


def tileinfo_mgrs(tileinfo: dict[str, Any]) -> str:
    try:
        zone = int(tileinfo["utmZone"])
        band = str(tileinfo["latitudeBand"])
        square = str(tileinfo["gridSquare"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Official L1C tileInfo lacks MGRS components") from exc
    return f"{zone:02d}{band}{square}"


def validated_l1c_assets(item: dict[str, Any]) -> dict[str, Any]:
    scene_id = str(item["id"])
    assets = item["assets"]
    spectral: dict[str, str] = {}
    corrections = 0
    for band, role in L1C_BANDS.items():
        url, corrected = official_l1c_url(str(assets[role]["href"]))
        spectral[band] = url
        corrections += int(corrected)
    tileinfo_url, corrected = official_l1c_url(
        str(assets["tileinfo_metadata"]["href"])
    )
    corrections += int(corrected)
    tileinfo = request_json("GET", tileinfo_url)
    expected_product = str(
        item.get("properties", {}).get("s2:product_uri", "")
    ).removesuffix(".SAFE")
    observed_product = str(tileinfo.get("productName", ""))
    if not expected_product or observed_product != expected_product:
        raise ValueError(
            f"Official L1C product metadata mismatch for {scene_id}: "
            f"{observed_product!r} vs {expected_product!r}"
        )
    observed_tile = tileinfo_mgrs(tileinfo)
    expected_tile = mgrs_tile(scene_id)
    if observed_tile != expected_tile:
        raise ValueError(
            f"Official L1C tile metadata mismatch for {scene_id}: "
            f"{observed_tile!r} vs {expected_tile!r}"
        )
    return {
        "spectral_assets": spectral,
        "tileinfo_metadata": tileinfo_url,
        "product_name": observed_product,
        "tileinfo_mgrs_tile": observed_tile,
        "catalog_asset_hrefs_corrected_from_l2a_bucket": corrections,
    }


def choose_reference(
    candidates: Iterable[dict[str, Any]],
    *,
    target_time: datetime,
    target_tile: str,
    center: list[float],
    excluded_target_ids: set[str],
    min_gap_hours: float,
    max_lookback_days: int,
    seasonal_min_lookback_days: int = 334,
    seasonal_max_lookback_days: int = 410,
    seasonal_target_gap_days: int = 365,
    excluded_utc_dates: set[str] | None = None,
    max_cloud: float,
) -> dict[str, Any]:
    primary: list[tuple[float, float, str, dict[str, Any]]] = []
    seasonal: list[tuple[float, float, str, dict[str, Any]]] = []
    for item in candidates:
        scene_id = str(item.get("id", ""))
        if scene_id in excluded_target_ids:
            continue
        try:
            if mgrs_tile(scene_id) != target_tile:
                continue
        except ValueError:
            continue
        if not contains_center(item, center):
            continue
        if not set(L1C_BANDS.values()).issubset(item.get("assets", {})):
            continue
        cloud_value = item.get("properties", {}).get("eo:cloud_cover")
        if cloud_value is None:
            continue
        try:
            cloud = float(cloud_value)
            acquired = parse_datetime(str(item["properties"]["datetime"]))
        except (KeyError, TypeError, ValueError):
            continue
        if acquired.date().isoformat() in (excluded_utc_dates or set()):
            continue
        gap_hours = (target_time - acquired).total_seconds() / 3600.0
        if gap_hours < min_gap_hours - 1e-9:
            continue
        if cloud > max_cloud + 1e-9:
            continue
        gap_days = gap_hours / 24.0
        if gap_hours <= max_lookback_days * 24.0 + 1e-9:
            primary.append((gap_hours, cloud, scene_id, item))
        elif (
            seasonal_min_lookback_days - 1e-9
            <= gap_days
            <= seasonal_max_lookback_days + 1e-9
        ):
            seasonal.append(
                (abs(gap_days - seasonal_target_gap_days), cloud, scene_id, item)
            )
    if primary:
        return min(primary, key=lambda value: (value[0], value[1], value[2]))[-1]
    if seasonal:
        return min(seasonal, key=lambda value: (value[0], value[1], value[2]))[-1]
    else:
        raise ValueError(
            "No eligible prior same-tile L1C reference in the primary or "
            "seasonal-fallback window"
        )


def public_item(scene_id: str) -> dict[str, Any]:
    return request_json("GET", item_url(scene_id))


def compact_l1c_item(
    item: dict[str, Any],
    *,
    expected_id: str,
    center: list[float],
) -> dict[str, Any]:
    validate_catalog_item(item, expected_id=expected_id, center=center)
    properties = item.get("properties", {})
    acquired = parse_datetime(str(properties["datetime"]))
    assets = validated_l1c_assets(item)
    cloud_value = properties.get("eo:cloud_cover")
    return {
        "scene_id": expected_id,
        "stac_item": item_url(expected_id),
        "datetime": acquired.isoformat(),
        "mgrs_tile": mgrs_tile(expected_id),
        "catalog_scene_cloud_cover_pct": (
            None if cloud_value is None else round(float(cloud_value), 6)
        ),
        "spectral_asset_roles": dict(L1C_BANDS),
        "spectral_assets": assets["spectral_assets"],
        "tileinfo_metadata": assets["tileinfo_metadata"],
        "product_name": assets["product_name"],
        "tileinfo_mgrs_tile": assets["tileinfo_mgrs_tile"],
        "catalog_asset_hrefs_corrected_from_l2a_bucket": assets[
            "catalog_asset_hrefs_corrected_from_l2a_bucket"
        ],
    }


def search_reference_candidates(
    target: dict[str, Any],
    *,
    min_gap_hours: float,
    max_search_lookback_days: int,
    max_cloud: float,
) -> list[dict[str, Any]]:
    target_time = parse_datetime(target["target_datetime"])
    start = target_time - timedelta(days=max_search_lookback_days)
    end = target_time - timedelta(hours=min_gap_hours)
    longitude, latitude = target["center"]
    response = request_json(
        "POST",
        EARTH_SEARCH_URL,
        payload={
            "collections": [L1C_COLLECTION],
            "bbox": [
                longitude - 0.001,
                latitude - 0.001,
                longitude + 0.001,
                latitude + 0.001,
            ],
            "datetime": (
                f"{start.isoformat().replace('+00:00', 'Z')}/"
                f"{end.isoformat().replace('+00:00', 'Z')}"
            ),
            "limit": 1000,
            "query": {"eo:cloud_cover": {"lte": max_cloud}},
        },
    )
    features = response.get("features", [])
    if not isinstance(features, list):
        raise ValueError("Earth Search response has a non-list features member")
    return features


def pair_one(
    target: dict[str, Any],
    *,
    excluded_target_ids: set[str],
    min_gap_hours: float,
    max_lookback_days: int,
    seasonal_min_lookback_days: int,
    seasonal_max_lookback_days: int,
    seasonal_target_gap_days: int,
    excluded_utc_dates: set[str],
    max_cloud: float,
) -> dict[str, Any]:
    try:
        center = target["center"]
        target_id = target["target_scene_id"]
        target_time = parse_datetime(target["target_datetime"])
        target_item = public_item(target_id)
        compact_target = compact_l1c_item(
            target_item,
            expected_id=target_id,
            center=center,
        )
        fetched_target_time = parse_datetime(compact_target["datetime"])
        if abs((fetched_target_time - target_time).total_seconds()) > 1.0:
            raise ValueError(f"Exact target datetime mismatch for {target_id}")
        candidates = search_reference_candidates(
            target,
            min_gap_hours=min_gap_hours,
            max_search_lookback_days=seasonal_max_lookback_days,
            max_cloud=max_cloud,
        )
        selected = choose_reference(
            candidates,
            target_time=target_time,
            target_tile=mgrs_tile(target_id),
            center=center,
            excluded_target_ids=excluded_target_ids,
            min_gap_hours=min_gap_hours,
            max_lookback_days=max_lookback_days,
            seasonal_min_lookback_days=seasonal_min_lookback_days,
            seasonal_max_lookback_days=seasonal_max_lookback_days,
            seasonal_target_gap_days=seasonal_target_gap_days,
            excluded_utc_dates=excluded_utc_dates,
            max_cloud=max_cloud,
        )
        reference_id = str(selected["id"])
        if reference_id in excluded_target_ids:
            raise ValueError(f"Reference is a forbidden source-cohort target: {reference_id}")
        reference_item = public_item(reference_id)
        compact_reference = compact_l1c_item(
            reference_item,
            expected_id=reference_id,
            center=center,
        )
        reference_time = parse_datetime(compact_reference["datetime"])
        gap_hours = (target_time - reference_time).total_seconds() / 3600.0
        cloud = compact_reference["catalog_scene_cloud_cover_pct"]
        if cloud is None or cloud > max_cloud + 1e-9:
            raise ValueError(f"Reference cloud metadata changed for {reference_id}")
        in_primary = min_gap_hours - 1e-9 <= gap_hours <= max_lookback_days * 24 + 1e-9
        in_seasonal = (
            seasonal_min_lookback_days * 24 - 1e-9
            <= gap_hours
            <= seasonal_max_lookback_days * 24 + 1e-9
        )
        if not (in_primary or in_seasonal):
            raise ValueError(f"Reference time metadata changed for {reference_id}")
        result = {
            "status": "paired",
            "event_id": target["event_id"],
            "observed_at_utc": target["observed_at_utc"],
            "center": center,
            "target": compact_target,
            "reference": compact_reference,
            "reference_to_target_gap_hours": round(gap_hours, 6),
            "reference_tier": "primary_1_to_31_days" if in_primary else "seasonal_334_to_410_days",
        }
        assert_no_outcome_fields(result)
        return result
    except Exception as exc:
        error = {
            "status": "pair_error",
            "event_id": target.get("event_id"),
            "target_scene_id": target.get("target_scene_id"),
            "error_type": type(exc).__name__,
            "error": str(exc)[:800],
        }
        assert_no_outcome_fields(error)
        return error


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    try:
        contract = protocol["sentinel_2_l1c_stress_acquisition_contract"]
        reference = contract["reference"]
        crop = contract["crop"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Protocol lacks the frozen Sentinel-2 L1C stress contract") from exc
    expected = {
        "minimum_gap_hours": 1,
        "maximum_lookback_days": 31,
        "maximum_catalog_eo_cloud_cover_pct": 20,
    }
    for key, value in expected.items():
        if reference.get(key) != value:
            raise ValueError(f"Protocol {key} must remain frozen at {value}")
    if reference.get("same_mgrs_tile") is not True:
        raise ValueError("Protocol must require a same-MGRS reference")
    seasonal = reference.get("seasonal_fallback", {})
    seasonal_expected = {
        "enabled_only_when_primary_window_empty": True,
        "minimum_lookback_days": 334,
        "maximum_lookback_days": 410,
        "target_gap_days": 365,
    }
    for key, value in seasonal_expected.items():
        if seasonal.get(key) != value:
            raise ValueError(
                f"Protocol seasonal fallback {key} must remain frozen at {value}"
            )
    campaign_exclusions = reference.get("additional_campaign_exclusions", {})
    excluded_dates = campaign_exclusions.get("utc_dates")
    if not isinstance(excluded_dates, list) or not excluded_dates:
        raise ValueError("Protocol must freeze non-empty campaign exclusion UTC dates")
    for value in excluded_dates:
        datetime.strptime(str(value), "%Y-%m-%d")
    if crop.get("shape_pixels") != [256, 256]:
        raise ValueError("Protocol crop shape must remain 256x256")
    if crop.get("band_order") != list(L1C_BANDS):
        raise ValueError("Protocol L1C band order does not match the script")
    return contract


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    assert_no_outcome_fields(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def bindings(root: Path, source_path: Path, protocol_path: Path) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "source_manifest": {
            "path": source_path.relative_to(root).as_posix(),
            "sha256": sha256(source_path),
        },
        "protocol": {
            "path": protocol_path.relative_to(root).as_posix(),
            "sha256": sha256(protocol_path),
        },
        "script": {
            "path": script_path.relative_to(root).as_posix(),
            "sha256": sha256(script_path),
        },
    }


def build_manifest(
    root: Path,
    source_path: Path,
    protocol_path: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen = validate_protocol(protocol)
    targets, excluded_target_ids = source_targets(read_jsonl(source_path))
    reference = frozen["reference"]
    seasonal = reference["seasonal_fallback"]
    excluded_utc_dates = {
        str(value) for value in reference["additional_campaign_exclusions"]["utc_dates"]
    }
    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(
            executor.map(
                lambda target: pair_one(
                    target,
                    excluded_target_ids=excluded_target_ids,
                    min_gap_hours=float(reference["minimum_gap_hours"]),
                    max_lookback_days=int(reference["maximum_lookback_days"]),
                    seasonal_min_lookback_days=int(seasonal["minimum_lookback_days"]),
                    seasonal_max_lookback_days=int(seasonal["maximum_lookback_days"]),
                    seasonal_target_gap_days=int(seasonal["target_gap_days"]),
                    excluded_utc_dates=excluded_utc_dates,
                    max_cloud=float(reference["maximum_catalog_eo_cloud_cover_pct"]),
                ),
                targets,
            )
        )
    pairs = sorted(
        (item for item in outcomes if item["status"] == "paired"),
        key=lambda item: (item["target"]["datetime"], item["event_id"]),
    )
    errors = sorted(
        (item for item in outcomes if item["status"] != "paired"),
        key=lambda item: str(item.get("event_id")),
    )
    target_ids = {item["target"]["scene_id"] for item in pairs}
    reference_ids = {item["reference"]["scene_id"] for item in pairs}
    overlap = reference_ids & excluded_target_ids
    if overlap:
        raise ValueError(f"References include source-cohort targets: {sorted(overlap)}")
    campaign_date_overlap = sorted(
        {
            item["reference"]["datetime"][:10]
            for item in pairs
            if item["reference"]["datetime"][:10] in excluded_utc_dates
        }
    )
    if campaign_date_overlap:
        raise ValueError(
            f"References include excluded campaign dates: {campaign_date_overlap}"
        )
    tier_counts: dict[str, int] = {}
    for item in pairs:
        tier = str(item["reference_tier"])
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    manifest = {
        "schema_version": 1,
        "scope": "outcome_blind_stanford_2025_sentinel2_l1c_target_reference_pairs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": bindings(root, source_path, protocol_path),
        "contract": {
            "eligible_targets": "resolved Sentinel-2 L1C rows only",
            "reference_policy": (
                "prior-only same-MGRS L1C; 1 hour through 31 days; catalog "
                "eo:cloud_cover <= 20%; order by gap, cloud, then ID; when empty, "
                "334-410 day seasonal fallback ordered by distance from 365 days, cloud, then ID"
            ),
            "excluded_reference_ids": "every resolved source-cohort target scene ID",
            "excluded_reference_utc_dates": sorted(excluded_utc_dates),
            "spectral_product": "Sentinel-2 Level-1C top-of-atmosphere raw DN",
            "band_order": list(L1C_BANDS),
            "asset_authority": (
                "official public sentinel-s2-l1c AWS assets with exact productName "
                "and MGRS tileInfo verification"
            ),
            "landsat_status": (
                "pending exact USGS EROS Collection 2 Level-1 authentication; "
                "no Landsat L2 substitution"
            ),
        },
        "outcome_blindness": {
            "source_release_outcome_fields_selected": False,
            "source_release_outcome_fields_emitted": False,
            "detector_outcomes_accessed": False,
            "statement": "No detector outcomes were accessed.",
        },
        "summary": {
            "eligible_sentinel2_targets": len(targets),
            "complete_pairs": len(pairs),
            "pair_errors": len(errors),
            "unique_target_scene_ids": len(target_ids),
            "unique_reference_scene_ids": len(reference_ids),
            "resolved_source_target_ids_excluded": len(excluded_target_ids),
            "references_matching_source_targets": len(overlap),
            "references_matching_excluded_campaign_dates": len(campaign_date_overlap),
            "reference_tier_counts": tier_counts,
            "all_pairs_complete": len(pairs) == len(targets) and not errors,
        },
        "pairs": pairs,
        "errors": errors,
        "runtime": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "python": sys.version.split()[0],
            "requests": requests.__version__,
        },
    }
    assert_no_outcome_fields(manifest)
    return manifest


def compact_receipt(
    root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    receipt = {
        "schema_version": 1,
        "scope": manifest["scope"],
        "generated_at_utc": manifest["generated_at_utc"],
        "status": (
            "frozen_complete"
            if manifest["summary"]["all_pairs_complete"]
            else "frozen_with_pair_errors"
        ),
        "bindings": manifest["bindings"],
        "pair_manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256(manifest_path),
            "tracked": False,
        },
        "contract": manifest["contract"],
        "outcome_blindness": manifest["outcome_blindness"],
        "summary": manifest["summary"],
    }
    assert_no_outcome_fields(receipt)
    return receipt


def write_markdown(path: Path, receipt: dict[str, Any]) -> None:
    summary = receipt["summary"]
    manifest = receipt["pair_manifest"]
    lines = [
        "# Stanford controlled-release Sentinel-2 L1C pair freeze",
        "",
        f"- Status: `{receipt['status']}`",
        f"- Eligible resolved S2 L1C targets: {summary['eligible_sentinel2_targets']}",
        f"- Complete target/reference pairs: {summary['complete_pairs']}",
        f"- Pair errors: {summary['pair_errors']}",
        f"- References matching any source-cohort target: {summary['references_matching_source_targets']}",
        f"- Ignored pair manifest: `{manifest['path']}`",
        f"- Pair manifest SHA-256: `{manifest['sha256']}`",
        "",
        "References are prior-only Sentinel-2 Level-1C scenes on the same MGRS tile, from 1 hour through 31 days before the target, with catalog `eo:cloud_cover <= 20%`. Selection is deterministic by smallest time gap, then cloud cover, then item ID, after excluding every resolved source-cohort target scene ID and every listed 2024 Casa Grande controlled-release campaign UTC date. If that window is empty, a frozen 334-410 day seasonal fallback is selected by proximity to 365 days, then cloud cover, then item ID.",
        "",
        "Only official public L1C assets accepted by exact `tileInfo.json` product-name and MGRS checks are frozen. Landsat remains pending exact USGS EROS Collection 2 Level-1 authentication; no L2 substitute is allowed.",
        "",
        "Release labels/rates were neither selected nor emitted. No detector outcomes were accessed.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    root = repo_root()
    try:
        source_path = safe_repo_path(root, args.source)
        protocol_path = safe_repo_path(root, args.protocol)
        manifest_path = safe_repo_path(root, args.manifest)
        output_json = safe_repo_path(root, args.output_json)
        output_markdown = safe_repo_path(root, args.output_markdown)
        manifest = build_manifest(
            root,
            source_path,
            protocol_path,
            workers=args.workers,
        )
        write_json(manifest_path, manifest)
        receipt = compact_receipt(root, manifest_path, manifest)
        write_json(output_json, receipt)
        write_markdown(output_markdown, receipt)
    except (OSError, ValueError, requests.RequestException, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
            )
        )
        return 2
    result = {
        "ok": receipt["summary"]["all_pairs_complete"],
        "eligible_sentinel2_targets": receipt["summary"]["eligible_sentinel2_targets"],
        "complete_pairs": receipt["summary"]["complete_pairs"],
        "pair_errors": receipt["summary"]["pair_errors"],
        "manifest": manifest_path.relative_to(root).as_posix(),
        "output_json": output_json.relative_to(root).as_posix(),
        "output_markdown": output_markdown.relative_to(root).as_posix(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
