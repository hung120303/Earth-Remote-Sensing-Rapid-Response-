#!/usr/bin/env python3
"""Acquire a reproducible EMIT V002 plume-mask/Sentinel-2 pilot pair.

The protected EMIT concentration COG still requires an Earthdata login.  This
collector uses public NASA CMR metadata for the authoritative V002 plume
geometry and public Element 84 Sentinel-2 L2A COGs for the image inputs.  It
writes two timestamp stacks (before and after EMIT), physical plume masks, and
a provenance manifest into the repository's ignored acquisition directories.

It intentionally does not create a six-band ERSRR regression tile: a binary
CMR plume polygon is not interchangeable with the EMIT concentration band.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as transform_points
from rasterio.warp import transform_geom

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
CMR_COLLECTION_ID = "C3242707413-LPCLOUD"
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
S2_COLLECTION = "sentinel-2-l2a"
S2_BANDS = (
    ("B2", "blue"),
    ("B3", "green"),
    ("B4", "red"),
    ("B11", "swir16"),
    ("B12", "swir22"),
)
BAND_ORDER = tuple(name for name, _ in S2_BANDS)
SCL_NON_CLEAR = {0, 1, 3, 8, 9, 10, 11}


class AcquisitionError(RuntimeError):
    """Raised when a source cannot satisfy the acquisition contract."""


def repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def request_json(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = requests.request(
        method,
        url,
        params=params,
        json=payload,
        headers={"Accept": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise AcquisitionError(f"Expected an object from {url}")
    return result


def cmr_granule(granule_id: str) -> dict[str, Any]:
    payload = request_json(
        "GET",
        CMR_GRANULES_URL,
        params={
            "collection_concept_id": CMR_COLLECTION_ID,
            "granule_ur": granule_id,
            "page_size": 10,
        },
    )
    items = payload.get("items", [])
    matches = [item.get("umm", {}) for item in items if item.get("umm", {}).get("GranuleUR") == granule_id]
    if len(matches) != 1:
        raise AcquisitionError(f"Expected one CMR record for {granule_id}; found {len(matches)}")
    record = matches[0]
    version = record.get("CollectionReference", {}).get("Version")
    if version != "002":
        raise AcquisitionError(f"CMR returned version {version!r}, not V002")
    return record


def plume_feature(record: dict[str, Any]) -> dict[str, Any]:
    polygons = (
        record.get("SpatialExtent", {})
        .get("HorizontalSpatialDomain", {})
        .get("Geometry", {})
        .get("GPolygons", [])
    )
    if not polygons:
        raise AcquisitionError("CMR record has no plume polygon")

    rings: list[list[list[float]]] = []
    for polygon in polygons:
        points = polygon.get("Boundary", {}).get("Points", [])
        ring = [[float(point["Longitude"]), float(point["Latitude"])] for point in points]
        if len(ring) < 4:
            continue
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        rings.append(ring)
    if not rings:
        raise AcquisitionError("CMR plume polygons contain no valid rings")

    geometry: dict[str, Any]
    if len(rings) == 1:
        geometry = {"type": "Polygon", "coordinates": [rings[0]]}
    else:
        geometry = {"type": "MultiPolygon", "coordinates": [[[point for point in ring]] for ring in rings]}

    all_points = [point for ring in rings for point in ring]
    bbox = [
        min(point[0] for point in all_points),
        min(point[1] for point in all_points),
        max(point[0] for point in all_points),
        max(point[1] for point in all_points),
    ]
    additional = {
        value["Name"]: value.get("Values", [])
        for value in record.get("AdditionalAttributes", [])
        if value.get("Name")
    }
    return {
        "type": "Feature",
        "bbox": bbox,
        "geometry": geometry,
        "properties": {
            "granule_id": record["GranuleUR"],
            "datetime": record.get("TemporalExtent", {}).get("SingleDateTime"),
            "collection": "EMITL2BCH4PLM.002",
            "plume_id": (additional.get("PLUME_ID") or [None])[0],
            "source_scenes": additional.get("SOURCE_SCENES", []),
            "cloud_cover": record.get("CloudCover"),
        },
    }


def query_sentinel2(
    feature: dict[str, Any],
    observed_at: datetime,
    *,
    window_days: int,
    max_cloud: float,
) -> list[dict[str, Any]]:
    start = observed_at - timedelta(days=window_days)
    end = observed_at + timedelta(days=window_days)
    payload = {
        "collections": [S2_COLLECTION],
        "bbox": feature["bbox"],
        "datetime": f"{start.isoformat().replace('+00:00', 'Z')}/{end.isoformat().replace('+00:00', 'Z')}",
        "limit": 100,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
    }
    response = request_json("POST", EARTH_SEARCH_URL, payload=payload)
    center_lon, center_lat = bbox_center(feature["bbox"])
    candidates = []
    required_assets = {asset for _, asset in S2_BANDS} | {"scl"}
    for item in response.get("features", []):
        bbox = item.get("bbox")
        assets = item.get("assets", {})
        if not bbox or len(bbox) < 4 or not required_assets.issubset(assets):
            continue
        if not (bbox[0] <= center_lon <= bbox[2] and bbox[1] <= center_lat <= bbox[3]):
            continue
        scene_time = parse_datetime(item["properties"]["datetime"])
        item["_datetime"] = scene_time
        item["_offset_hours"] = (scene_time - observed_at).total_seconds() / 3600.0
        candidates.append(item)
    candidates.sort(
        key=lambda item: (
            abs(item["_offset_hours"]),
            float(item.get("properties", {}).get("eo:cloud_cover", 100.0)),
            item.get("id", ""),
        )
    )
    return candidates


def choose_scenes(candidates: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if not candidates:
        raise AcquisitionError("No Sentinel-2 L2A scenes met the spatial/cloud/time constraints")
    if mode == "nearest":
        return [candidates[0]]

    before = [item for item in candidates if item["_offset_hours"] <= 0]
    after = [item for item in candidates if item["_offset_hours"] >= 0]
    selected: list[dict[str, Any]] = []
    if before:
        selected.append(min(before, key=lambda item: abs(item["_offset_hours"])))
    if after:
        selected.append(min(after, key=lambda item: abs(item["_offset_hours"])))
    if not selected:
        selected.append(candidates[0])
    unique: dict[str, dict[str, Any]] = {item["id"]: item for item in selected}
    return sorted(unique.values(), key=lambda item: item["_datetime"])


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def geometry_bounds(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return bounds for a GeoJSON Polygon or MultiPolygon."""
    points: list[tuple[float, float]] = []

    def collect(value: Any) -> None:
        if (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            points.append((float(value[0]), float(value[1])))
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    collect(geometry.get("coordinates", []))
    if not points:
        raise AcquisitionError("Projected plume geometry contains no coordinates")
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def aligned_grid(
    reference_url: str,
    center: tuple[float, float],
    *,
    size: int,
    resolution: float,
) -> tuple[Any, Affine]:
    with rasterio.Env(**gdal_env()), rasterio.open(reference_url) as source:
        if source.crs is None:
            raise AcquisitionError(f"Sentinel-2 asset has no CRS: {reference_url}")
        xs, ys = transform_points("EPSG:4326", source.crs, [center[0]], [center[1]])
        half = size * resolution / 2.0
        left = math.floor((xs[0] - half) / resolution) * resolution
        top = math.ceil((ys[0] + half) / resolution) * resolution
        return source.crs, Affine(resolution, 0.0, left, 0.0, -resolution, top)


def gdal_env() -> dict[str, str]:
    return {
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "1",
    }


def read_to_grid(
    url: str,
    *,
    crs: Any,
    transform: Affine,
    size: int,
    resampling: Resampling,
) -> np.ndarray:
    with rasterio.Env(**gdal_env()), rasterio.open(url) as source:
        source_nodata = source.nodata if source.nodata is not None else 0
        with WarpedVRT(
            source,
            crs=crs,
            transform=transform,
            width=size,
            height=size,
            src_nodata=source_nodata,
            nodata=0,
            resampling=resampling,
        ) as vrt:
            return vrt.read(1, out_dtype="uint16")


def write_stack_and_mask(
    scene: dict[str, Any],
    feature: dict[str, Any],
    output_dir: Path,
    *,
    crs: Any,
    transform: Affine,
    size: int,
    resolution: float,
    overwrite: bool,
) -> dict[str, Any]:
    stack_path = output_dir / f"{scene['id']}.s2.tif"
    mask_path = output_dir / f"{scene['id']}.plume_mask.tif"
    if not overwrite and (stack_path.exists() or mask_path.exists()):
        raise AcquisitionError(f"Output exists; pass --overwrite to replace: {stack_path}")

    arrays = []
    band_urls: dict[str, str] = {}
    for band_name, asset_name in S2_BANDS:
        url = scene["assets"][asset_name]["href"]
        band_urls[band_name] = url
        arrays.append(
            read_to_grid(
                url,
                crs=crs,
                transform=transform,
                size=size,
                resampling=Resampling.bilinear,
            )
        )
    stack = np.stack(arrays)
    profile = {
        "driver": "GTiff",
        "width": size,
        "height": size,
        "count": len(S2_BANDS),
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(stack_path, "w", **profile) as target:
        target.write(stack)
        for index, (band_name, _) in enumerate(S2_BANDS, start=1):
            target.set_band_description(index, band_name)

    projected_geometry = transform_geom("EPSG:4326", crs, feature["geometry"], precision=3)
    plume_bounds = geometry_bounds(projected_geometry)
    grid_bounds = rasterio.transform.array_bounds(size, size, transform)
    plume_fully_contained = (
        grid_bounds[0] <= plume_bounds[0]
        and grid_bounds[1] <= plume_bounds[1]
        and grid_bounds[2] >= plume_bounds[2]
        and grid_bounds[3] >= plume_bounds[3]
    )
    plume_min_margin_pixels = min(
        plume_bounds[0] - grid_bounds[0],
        plume_bounds[1] - grid_bounds[1],
        grid_bounds[2] - plume_bounds[2],
        grid_bounds[3] - plume_bounds[3],
    ) / resolution
    mask = rasterize(
        [(projected_geometry, 1)],
        out_shape=(size, size),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    mask_profile = dict(profile)
    mask_profile.update(count=1, dtype="uint8", predictor=1)
    with rasterio.open(mask_path, "w", **mask_profile) as target:
        target.write(mask, 1)
        target.set_band_description(1, "EMIT_V002_CMR_PLUME_MASK")

    scl = read_to_grid(
        scene["assets"]["scl"]["href"],
        crs=crs,
        transform=transform,
        size=size,
        resampling=Resampling.nearest,
    )
    valid_scl = scl != 0
    clear = valid_scl & ~np.isin(scl, list(SCL_NON_CLEAR))
    roi_clear_pct = float(100.0 * clear.sum() / max(valid_scl.sum(), 1))

    return {
        "scene_id": scene["id"],
        "datetime": scene["_datetime"].isoformat(),
        "offset_hours": round(float(scene["_offset_hours"]), 3),
        "scene_cloud_cover_pct": float(scene["properties"].get("eo:cloud_cover", float("nan"))),
        "roi_clear_pct": round(roi_clear_pct, 3),
        "collection": S2_COLLECTION,
        "product_level": "L2A surface reflectance",
        "band_order": [band for band, _ in S2_BANDS],
        "band_urls": band_urls,
        "stack": relative_to_root(stack_path),
        "stack_sha256": sha256(stack_path),
        "mask": relative_to_root(mask_path),
        "mask_sha256": sha256(mask_path),
        "mask_positive_pixels": int(mask.sum()),
        "plume_fully_contained": plume_fully_contained,
        "plume_min_margin_pixels": round(float(plume_min_margin_pixels), 3),
        "plume_bounds": [round(float(value), 3) for value in plume_bounds],
        "grid_bounds": [round(float(value), 3) for value in grid_bounds],
        "crs": crs.to_string(),
        "resolution_m": resolution,
        "shape": [size, size],
        "transform": list(transform)[:6],
    }


def related_url(record: dict[str, Any], url_type: str, suffix: str) -> str | None:
    for item in record.get("RelatedUrls", []):
        url = item.get("URL", "")
        if item.get("Type") == url_type and url.lower().endswith(suffix.lower()):
            return url
    return None


def download_public(url: str, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("wb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    target.write(chunk)
    os.replace(temporary, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root()))
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    raw_base = (root / "EarthRemoteSensingRapidResponse" / "Data Collection" / "EMIT_Plumes").resolve()
    pair_base = (root / "EarthRemoteSensingRapidResponse" / "Data Collection" / "s2_emit_pairs").resolve()
    raw_dir = (raw_base / args.batch).resolve()
    pair_dir = (pair_base / args.batch / args.granule_id).resolve()
    if raw_base not in raw_dir.parents or pair_base not in pair_dir.parents:
        raise AcquisitionError("Resolved acquisition output escaped its intended base directory")
    record = cmr_granule(args.granule_id)
    feature = plume_feature(record)
    observed_at = parse_datetime(feature["properties"]["datetime"])

    raw_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    umm_path = raw_dir / f"{args.granule_id}.umm.json"
    plume_path = raw_dir / f"{args.granule_id}.plume.geojson"
    write_json(umm_path, record)
    write_json(plume_path, feature)

    browse_url = related_url(record, "GET RELATED VISUALIZATION", ".png")
    browse_path = raw_dir / f"{args.granule_id}.png"
    if browse_url:
        download_public(browse_url, browse_path, overwrite=args.overwrite)

    candidates = query_sentinel2(
        feature,
        observed_at,
        window_days=args.window_days,
        max_cloud=args.max_cloud,
    )
    scenes = choose_scenes(candidates, args.temporal_mode)
    grid_reference_scene = scenes[0]
    grid_crs, grid_transform = aligned_grid(
        grid_reference_scene["assets"]["swir16"]["href"],
        bbox_center(feature["bbox"]),
        size=args.tile_size,
        resolution=args.resolution,
    )
    scene_outputs = [
        write_stack_and_mask(
            scene,
            feature,
            pair_dir,
            crs=grid_crs,
            transform=grid_transform,
            size=args.tile_size,
            resolution=args.resolution,
            overwrite=args.overwrite,
        )
        for scene in scenes
    ]

    concentration_url = related_url(record, "GET DATA", ".tif")
    metadata_url = related_url(record, "GET DATA", ".json")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "granule_id": args.granule_id,
        "emit_collection": "EMITL2BCH4PLM.002",
        "emit_datetime": observed_at.isoformat(),
        "emit_cloud_cover_pct": record.get("CloudCover"),
        "label_type": "binary physical plume geometry",
        "label_source": "NASA CMR UMM SpatialExtent polygon for EMIT V002",
        "plume_geojson": relative_to_root(plume_path),
        "plume_geojson_sha256": sha256(plume_path),
        "umm_metadata": relative_to_root(umm_path),
        "umm_metadata_sha256": sha256(umm_path),
        "browse_png": relative_to_root(browse_path) if browse_path.exists() else None,
        "protected_emit_assets": {
            "concentration_cog": concentration_url,
            "companion_metadata": metadata_url,
            "status": "earthdata_authentication_required",
        },
        "sentinel2_selection": {
            "temporal_mode": args.temporal_mode,
            "window_days": args.window_days,
            "max_scene_cloud_pct": args.max_cloud,
            "rank": "absolute temporal offset, then scene cloud cover",
            "candidate_count": len(candidates),
            "shared_grid_reference_scene": grid_reference_scene["id"],
        },
        "scenes": scene_outputs,
        "contract_note": (
            "These mask pairs are segmentation data. Do not place them in the legacy six-band "
            "EMIT concentration dataset without an explicit schema migration."
        ),
    }
    manifest_path = pair_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "ok": True,
        "granule_id": args.granule_id,
        "manifest": relative_to_root(manifest_path),
        "scenes": scene_outputs,
        "protected_emit_cog_downloaded": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire public EMIT V002 plume geometry and bracketing Sentinel-2 L2A crops"
    )
    parser.add_argument("granule_id", help="Exact EMIT_L2B_CH4PLM_002_* granule ID")
    parser.add_argument("--batch", default="emit-v002-pilot")
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--temporal-mode", choices=("bracketing", "nearest"), default="bracketing")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--resolution", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.batch) is None or ".." in args.batch:
        parser.error("batch must be a safe 1-64 character slug")
    if re.fullmatch(r"EMIT_L2B_CH4PLM_002_\d{8}T\d{6}_\d{6}", args.granule_id) is None:
        parser.error("granule_id must be an exact EMIT L2B CH4PLM V002 granule ID")
    if not 0.0 <= args.max_cloud <= 100.0:
        parser.error("max-cloud must be between 0 and 100")
    if not 0 < args.window_days <= 90:
        parser.error("window-days must be between 1 and 90")
    if not 0 < args.tile_size <= 2048 or not 0 < args.resolution <= 100:
        parser.error("tile-size must be 1..2048 and resolution must be >0..100 metres")
    try:
        result = acquire(args)
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        print(json.dumps(result, separators=(",", ":") if args.compact else None, indent=None if args.compact else 2))
        return 1
    print(json.dumps(result, separators=(",", ":") if args.compact else None, indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
