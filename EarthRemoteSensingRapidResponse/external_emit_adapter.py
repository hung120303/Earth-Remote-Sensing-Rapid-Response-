"""Adapter from the frozen EMIT/Sentinel-2 cohort to the MARS v3 contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from mars_s2l_adapter import (
    CLOUD_CLASSES,
    MARS_BANDS,
    REFLECTANCE_DIVISOR,
    REFLECTANCE_MAX,
    compute_mbmp,
)
from mars_v3_model import INPUT_CHANNELS


@dataclass(frozen=True)
class ExternalEmitScene:
    group_id: str
    granule_id: str
    target_scene_id: str
    inputs: np.ndarray
    observable: np.ndarray
    plume_mask: np.ndarray
    wind_u_m_s: float
    wind_v_m_s: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_asset(root: Path, record: dict[str, Any]) -> Path:
    path = (root / record["path"]).resolve()
    if root not in path.parents:
        raise ValueError(f"External asset escapes repository root: {record['path']}")
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"External asset is missing or size-mismatched: {path}")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"External asset SHA-256 mismatch: {path}")
    return path


def build_external_inputs(
    target_raw: np.ndarray,
    reference_raw: np.ndarray,
    cloud_classes: np.ndarray,
    wind: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    target_values = np.asarray(target_raw)
    reference_values = np.asarray(reference_raw)
    cloud = np.asarray(cloud_classes)
    if target_values.shape != reference_values.shape or target_values.ndim != 3:
        raise ValueError("External target and reference must be matching band-first arrays")
    if target_values.shape[0] != len(MARS_BANDS):
        raise ValueError(f"Expected {len(MARS_BANDS)} MARS bands, got {target_values.shape[0]}")
    if cloud.shape != target_values.shape[1:]:
        raise ValueError("External cloud mask does not match the spectral grid")
    unknown = set(int(value) for value in np.unique(cloud)) - set(CLOUD_CLASSES)
    if unknown:
        raise ValueError(f"External CloudSEN12 mask has unknown classes: {sorted(unknown)}")
    wind_values = np.asarray(wind, dtype=np.float32)
    if wind_values.shape != (2,) or not np.all(np.isfinite(wind_values)):
        raise ValueError("External wind must contain finite u/v components")
    wind_values = np.clip(wind_values, -20.0, 20.0)
    raw_pair = np.concatenate([target_values, reference_values], axis=0)
    reflectance = np.clip(
        raw_pair.astype(np.float32) / REFLECTANCE_DIVISOR,
        0.0,
        REFLECTANCE_MAX,
    )
    target = reflectance[: len(MARS_BANDS)]
    reference = reflectance[len(MARS_BANDS) :]
    radiometric_valid = np.all(target_values != 0, axis=0) & np.all(
        reference_values != 0, axis=0
    )
    observable = radiometric_valid & (cloud == 0)
    mbmp = compute_mbmp(target, reference)
    wind_channels = np.broadcast_to(
        (wind_values / 8.0)[:, None, None],
        (2, target_values.shape[1], target_values.shape[2]),
    ).copy()
    cloud_binary = (cloud > 0).astype(np.float32)
    inputs = np.concatenate(
        [mbmp[None, ...], reflectance, wind_channels, cloud_binary[None, ...]], axis=0
    ).astype(np.float32)
    if inputs.shape[0] != len(INPUT_CHANNELS):
        raise ValueError("External input does not match the frozen v3 channel contract")
    return inputs, observable


def _read_raster(
    path: Path, *, count: int, descriptions: tuple[str, ...]
) -> tuple[np.ndarray, tuple[Any, ...]]:
    with rasterio.open(path) as source:
        if source.count != count:
            raise ValueError(f"Expected {count} bands in {path}, got {source.count}")
        if tuple(source.descriptions) != descriptions:
            raise ValueError(
                f"Unexpected band descriptions in {path}: {source.descriptions}"
            )
        grid = (source.width, source.height, source.crs, tuple(source.transform)[:6])
        return source.read(), grid


def load_external_scene(
    root: Path,
    crop_manifest_path: Path,
    cloud_manifest_path: Path,
    wind_record: dict[str, Any],
) -> ExternalEmitScene:
    crop = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
    cloud_manifest = json.loads(cloud_manifest_path.read_text(encoding="utf-8"))
    if not crop["quality"]["gate_pass"]:
        raise ValueError(f"External scene is not gate-pass: {crop['group_id']}")
    identities = {
        crop["group_id"],
        cloud_manifest["group_id"],
        wind_record["group_id"],
    }
    if len(identities) != 1:
        raise ValueError(f"External scene identities disagree: {sorted(identities)}")
    target_path = verified_asset(root, crop["assets"]["target_l1c"])
    reference_path = verified_asset(root, crop["assets"]["reference_l1c"])
    plume_path = verified_asset(root, crop["assets"]["plume_mask"])
    cloud_path = verified_asset(root, cloud_manifest["asset"])
    target, target_grid = _read_raster(
        target_path, count=len(MARS_BANDS), descriptions=MARS_BANDS
    )
    reference, reference_grid = _read_raster(
        reference_path,
        count=len(MARS_BANDS),
        descriptions=tuple(f"{band}_reference" for band in MARS_BANDS),
    )
    cloud, cloud_grid = _read_raster(
        cloud_path, count=1, descriptions=("CloudSEN12_UNetMobV2_V2",)
    )
    plume, plume_grid = _read_raster(
        plume_path, count=1, descriptions=("EMIT_V002_CMR_PLUME_MASK",)
    )
    if len({target_grid, reference_grid, cloud_grid, plume_grid}) != 1:
        raise ValueError(f"External scene rasters are not co-registered: {crop['group_id']}")
    inputs, observable = build_external_inputs(
        target,
        reference,
        cloud[0],
        (float(wind_record["wind_u_m_s"]), float(wind_record["wind_v_m_s"])),
    )
    plume_mask = plume[0].astype(bool)
    if not np.any(plume_mask):
        raise ValueError(f"External positive has an empty plume mask: {crop['group_id']}")
    return ExternalEmitScene(
        group_id=crop["group_id"],
        granule_id=crop["granule_id"],
        target_scene_id=crop["target_scene_id"],
        inputs=inputs,
        observable=observable,
        plume_mask=plume_mask,
        wind_u_m_s=float(wind_record["wind_u_m_s"]),
        wind_v_m_s=float(wind_record["wind_v_m_s"]),
    )
