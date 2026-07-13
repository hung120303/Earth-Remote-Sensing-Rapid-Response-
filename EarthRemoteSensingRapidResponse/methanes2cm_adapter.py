"""Validated adapter for the pinned MethaneS2CM L2A 32x32 contract.

MethaneS2CM writes each 12-band Sentinel-2 array as twelve TIFF pages rather
than twelve GDAL bands.  Rasterio therefore exposes only the first page.  This
module deliberately uses tifffile and fails closed on any other layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import tifffile

from mars_s2l_adapter import compute_mbmp

S2_L2A_BAND_ORDER = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)
MODEL_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
MODEL_BAND_INDICES = tuple(S2_L2A_BAND_ORDER.index(band) for band in MODEL_BANDS)
REFLECTANCE_DIVISOR = 10_000.0
REFLECTANCE_MAX = 2.0
PATCH_SHAPE = (32, 32)
STACK_SHAPE = (len(S2_L2A_BAND_ORDER), *PATCH_SHAPE)
IMPUTED_WIND_MPS = (4.0, 4.0)
INPUT_CHANNELS = (
    "mbmp_T_over_Tminus90",
    *(f"target_{band}" for band in MODEL_BANDS),
    *(f"reference90_{band}" for band in MODEL_BANDS),
    "wind_u_div8",
    "wind_v_div8",
    "cloud_binary_unavailable_zero",
)
V5_INPUT_CHANNELS = (
    "mbmp_T_over_Tminus90",
    "mbmp_T_over_Tminus365",
    *(f"target_{band}" for band in MODEL_BANDS),
    *(f"reference90_{band}" for band in MODEL_BANDS),
    *(f"reference365_{band}" for band in MODEL_BANDS),
)


@dataclass(frozen=True)
class MethaneS2CMSample:
    sample_id: str
    label: int
    group_id: str
    target: np.ndarray
    reference90: np.ndarray
    reference365: np.ndarray
    observable_mask: np.ndarray
    plume_mask: np.ndarray


def coordinate_group_id(record: dict[str, Any]) -> str:
    """Return the exact-location identity without lossy float reformatting."""
    latitude = str(record["latitude"]).strip()
    longitude = str(record["longitude"]).strip()
    if not latitude or not longitude:
        raise ValueError("MethaneS2CM record is missing coordinates")
    return f"latlon:{latitude},{longitude}"


def safe_sample_path(split_dir: Path, relative_path: str, expected_name: str) -> Path:
    """Resolve one CSV-declared sample asset beneath the split directory."""
    relative = PurePosixPath(str(relative_path))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.name != expected_name
        or not relative.parts[0].isdigit()
    ):
        raise ValueError(f"Unsafe MethaneS2CM asset path: {relative_path!r}")
    # The strict two-component contract already proves lexical containment and
    # avoids a filesystem round trip for every asset on Windows-mounted data.
    base = Path(os.path.abspath(split_dir))
    return base / relative.parts[0] / relative.parts[1]


def read_stack(path: Path) -> np.ndarray:
    """Read and validate a 12-page Sentinel-2 TIFF stack."""
    values = np.asarray(tifffile.imread(path))
    if values.shape != STACK_SHAPE or values.dtype != np.uint16:
        raise ValueError(
            f"Expected uint16 MethaneS2CM stack {STACK_SHAPE}, got {values.shape} {values.dtype}"
        )
    return values


def normalize_binary_mask(values: np.ndarray) -> np.ndarray:
    """Validate a source mask and return the canonical packed uint8 form.

    MethaneS2CM stores the released plume TIFFs as float64 even though their
    values are binary.  Validate the semantic contract rather than silently
    depending on the storage dtype, then canonicalize before training.
    """
    values = np.asarray(values)
    if values.shape != PATCH_SHAPE or not np.issubdtype(values.dtype, np.number):
        raise ValueError(
            f"Expected numeric MethaneS2CM mask {PATCH_SHAPE}, got "
            f"{values.shape} {values.dtype}"
        )
    if not np.all(np.isfinite(values)) or not np.all(np.isin(np.unique(values), (0, 1))):
        raise ValueError("MethaneS2CM plume mask is not finite and binary")
    return values.astype(np.uint8, copy=False)


def read_mask(path: Path) -> np.ndarray:
    values = normalize_binary_mask(np.asarray(tifffile.imread(path)))
    return values.astype(bool, copy=False)


def _reflectance(stack: np.ndarray) -> np.ndarray:
    selected = stack[np.asarray(MODEL_BAND_INDICES)].astype(np.float32)
    selected /= REFLECTANCE_DIVISOR
    np.clip(selected, 0.0, REFLECTANCE_MAX, out=selected)
    return selected


def load_sample(split_dir: Path, record: dict[str, Any]) -> MethaneS2CMSample:
    identifier = str(record["id"]).strip()
    if not identifier.isdigit():
        raise ValueError(f"Invalid MethaneS2CM sample id: {identifier!r}")
    declared = {
        "s2_path": ("s2.tif", "target"),
        "s2_pre_path": ("s2_pre.tif", "reference90"),
        "s2_pre_pre_path": ("s2_pre_pre.tif", "reference365"),
        "plume_mask_path": ("plume.tif", "mask"),
    }
    paths = {
        role: safe_sample_path(split_dir, str(record[column]), filename)
        for column, (filename, role) in declared.items()
    }
    target_raw = read_stack(paths["target"])
    reference90_raw = read_stack(paths["reference90"])
    reference365_raw = read_stack(paths["reference365"])
    plume = read_mask(paths["mask"])
    try:
        label = int(record["label"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MethaneS2CM label for {identifier}") from exc
    if label not in (0, 1) or bool(label) != bool(np.any(plume)):
        raise ValueError(f"Label/mask disagreement for MethaneS2CM sample {identifier}")

    raw_selected = np.concatenate(
        [
            target_raw[np.asarray(MODEL_BAND_INDICES)],
            reference90_raw[np.asarray(MODEL_BAND_INDICES)],
            reference365_raw[np.asarray(MODEL_BAND_INDICES)],
        ]
    )
    observable = np.all(raw_selected != 0, axis=0)
    return MethaneS2CMSample(
        sample_id=identifier,
        label=label,
        group_id=coordinate_group_id(record),
        target=_reflectance(target_raw),
        reference90=_reflectance(reference90_raw),
        reference365=_reflectance(reference365_raw),
        observable_mask=observable,
        plume_mask=plume,
    )


def v4_input(
    sample: MethaneS2CMSample,
    *,
    wind: tuple[float, float] = IMPUTED_WIND_MPS,
) -> np.ndarray:
    """Construct the frozen 16-channel v4/released-MARS compatibility input."""
    mbmp = compute_mbmp(sample.target, sample.reference90, valid_mask=sample.observable_mask)
    spectral = np.concatenate([sample.target, sample.reference90])
    height, width = PATCH_SHAPE
    wind_channels = np.broadcast_to(
        np.asarray(wind, dtype=np.float32)[:, None, None] / 8.0,
        (2, height, width),
    ).copy()
    cloud = np.zeros((1, height, width), dtype=np.float32)
    values = np.concatenate([mbmp[None], spectral, wind_channels, cloud]).astype(np.float32)
    if values.shape != (len(INPUT_CHANNELS), *PATCH_SHAPE):
        raise ValueError("Constructed MethaneS2CM v4 input violates its channel contract")
    return values


def v5_input(sample: MethaneS2CMSample) -> np.ndarray:
    """Construct the predeclared tri-temporal v5 input."""
    mbmp90 = compute_mbmp(
        sample.target, sample.reference90, valid_mask=sample.observable_mask
    )
    mbmp365 = compute_mbmp(
        sample.target, sample.reference365, valid_mask=sample.observable_mask
    )
    values = np.concatenate(
        [
            mbmp90[None],
            mbmp365[None],
            sample.target,
            sample.reference90,
            sample.reference365,
        ]
    ).astype(np.float32)
    if values.shape != (len(V5_INPUT_CHANNELS), *PATCH_SHAPE):
        raise ValueError("Constructed MethaneS2CM v5 input violates its channel contract")
    values[:, ~sample.observable_mask] = 0.0
    values[0:2, ~sample.observable_mask] = 1.0
    return values
