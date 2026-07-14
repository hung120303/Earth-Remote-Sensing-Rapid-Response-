"""Native, validity-aware adapter for the pinned mixed-sensor MARS-S2L contract.

This module deliberately does not depend on the legacy ERSRR five-band model.
MARS-S2L stores six target bands followed by the corresponding six background
bands. Ancillary roles come from the release manifest because some rasters omit
descriptive tags and cloud-mask nodata metadata can overlap encoded classes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

import numpy as np
import rasterio

MARS_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
MARS_BACKGROUND_BANDS = tuple(f"{band}_bg" for band in MARS_BANDS)
MARS_IMAGE_BANDS = MARS_BANDS + MARS_BACKGROUND_BANDS
DEVELOPMENT_RESEARCH_ROLES = frozenset(
    {"development_training", "development_validation"}
)
SEALED_RESEARCH_ROLES = frozenset({"sealed_paper_test"})
REFLECTANCE_DIVISOR = 5000.0
REFLECTANCE_MAX = 2.0
MBMP_NEUTRAL = 1.0
MBMP_MAX = 10.0
CLOUD_CLASSES = {
    0: "clear",
    1: "thick_cloud",
    2: "thin_cloud",
    3: "cloud_shadow",
    4: "nodata_or_invalid",
}
ENHANCEMENT_UNIT_STATUS = (
    "conflict: GeoTIFF tags may say DeltaCH4(ppm), while the pinned MARS-S2L "
    "README describes enhancement values as ppb; do not use quantitative units "
    "until reconciled with the data producer"
)


@dataclass(frozen=True)
class MarsS2Sample:
    """In-memory representation used by research models and baseline runners."""

    sample_id: str
    split: str
    sensor_family: str
    satellite: str
    label_state: str
    location_id: str
    target_scene_id: str
    reference_scene_id: str
    raw_pair: np.ndarray
    reflectance_pair: np.ndarray
    target: np.ndarray
    reference: np.ndarray
    cloud_classes: np.ndarray
    clear_mask: np.ndarray
    radiometric_valid_mask: np.ndarray
    observable_mask: np.ndarray
    plume_mask: np.ndarray
    methane_enhancement_raw: np.ndarray | None
    mbmp_release_compatible: np.ndarray
    mbmp_valid_aware: np.ndarray
    crs: str
    transform: tuple[float, ...]

    @property
    def presence(self) -> int:
        return int(self.label_state == "PLUME")


@lru_cache(maxsize=8192)
def _validate_parent_chain(base_value: str, parent_value: str) -> None:
    """Cache immutable acquisition-directory symlink checks by parent folder."""
    base = Path(base_value)
    parent = Path(parent_value)
    while parent != base:
        if parent.exists() and parent.is_symlink():
            raise ValueError(f"MARS-S2L asset traverses a symlink: {parent}")
        parent = parent.parent


def safe_asset_path(base_dir: Path, relative_path: str) -> Path:
    """Resolve a release-relative asset without allowing acquisition-root escape."""
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe MARS-S2L asset path: {relative_path}")
    base = base_dir.resolve()
    result = Path(os.path.abspath(os.path.join(str(base), *relative.parts)))
    if os.path.normcase(os.path.commonpath([str(base), str(result)])) != os.path.normcase(str(base)):
        raise ValueError(f"MARS-S2L asset escapes base directory: {relative_path}")
    _validate_parent_chain(str(base), str(result.parent))
    return result


def role_paths(record: dict[str, Any]) -> dict[str, str]:
    """Return a unique semantic-role to path mapping from a manifest record."""
    result: dict[str, str] = {}
    for item in record["assets"]:
        role = str(item["role"])
        path = str(item["path"])
        if role in result:
            raise ValueError(f"Duplicate asset role {role!r} for {record_id(record)}")
        result[role] = path
    return result


def validate_image_band_order(
    record: dict[str, Any], descriptions: tuple[str | None, ...]
) -> str:
    """Validate embedded band labels or a narrowly scoped manifest fallback."""
    declared = tuple(record.get("band_order") or ())
    if descriptions == MARS_IMAGE_BANDS or descriptions == declared:
        return "embedded_descriptions"
    if all(value is None for value in descriptions) and len(declared) == 12:
        # A small producer-side tranche omits TIFF band descriptions. The
        # frozen, hash-bound MARS manifest still declares the exact 12-band
        # order; accept only the all-missing case, never partial/mixed labels.
        return "frozen_manifest_declaration"
    raise ValueError(
        f"Unexpected image band order for {record_id(record)}: embedded={descriptions}, "
        f"manifest={declared}"
    )


def record_id(record: dict[str, Any]) -> str:
    return str(record.get("sample_id") or record.get("id_loc_image") or "<unknown>")


def label_state(record: dict[str, Any]) -> str:
    value = str(record.get("label_state") or record.get("label") or "").upper()
    aliases = {"POSITIVE": "PLUME", "NEGATIVE": "NO_PLUME"}
    value = aliases.get(value, value)
    if value not in {"PLUME", "NO_PLUME"}:
        raise ValueError(f"Unsupported MARS-S2L label state {value!r} for {record_id(record)}")
    return value


def iter_manifest(path: Path) -> Iterator[dict[str, Any]]:
    """Read either the acquisition JSON manifest or frozen cohort JSONL."""
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid manifest JSONL at line {line_number}") from exc
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"JSON manifest does not contain a samples list: {path}")
    yield from samples


def iter_development_manifest(path: Path) -> Iterator[dict[str, Any]]:
    """Yield only successor-development rows and reject any sealed-test row.

    The paper-v3 cohort builder writes a physically separate development
    manifest. This validation is a second line of defense against accidentally
    passing the combined acquisition or sealed-test manifest to training code.
    """
    for record in iter_manifest(path):
        role = str(record.get("research_role") or "")
        if role in SEALED_RESEARCH_ROLES:
            raise ValueError(
                f"Development loader refuses sealed role {role!r} in {path}"
            )
        if role not in DEVELOPMENT_RESEARCH_ROLES:
            raise ValueError(
                f"Unsupported development role {role!r} in {path}"
            )
        yield record


def _normalized_ratio(
    b11: np.ndarray, b12: np.ndarray, normalization_mask: np.ndarray
) -> np.ndarray:
    ratio = np.full(b11.shape, MBMP_NEUTRAL, dtype=np.float32)
    usable = normalization_mask & np.isfinite(b11) & np.isfinite(b12) & (b11 != 0)
    ratio[usable] = b12[usable] / b11[usable]
    values = ratio[usable]
    median = float(np.median(values)) if values.size else MBMP_NEUTRAL
    if not np.isfinite(median) or abs(median) < 1e-8:
        median = MBMP_NEUTRAL
    ratio[usable] /= median
    np.clip(ratio, 0.0, MBMP_MAX, out=ratio)
    return ratio


def compute_mbmp(
    target: np.ndarray,
    reference: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the release MBMP ratio, optionally using a validity-aware median.

    The released implementation normalizes each B12/B11 ratio by its scene
    median, divides target by background, fills non-finite values with one, and
    clips to ten. Supplying ``valid_mask`` restricts both medians and output to
    observable pixels; omitting it reproduces the released nonzero-B11 logic.
    """
    target_values = np.asarray(target, dtype=np.float32)
    reference_values = np.asarray(reference, dtype=np.float32)
    if target_values.shape != reference_values.shape or target_values.shape[0] != 6:
        raise ValueError(
            f"Expected matching six-band target/reference arrays; got "
            f"{target_values.shape} and {reference_values.shape}"
        )
    if valid_mask is None:
        target_normalization = target_values[4] != 0
        reference_normalization = reference_values[4] != 0
        output_valid = np.ones(target_values.shape[1:], dtype=bool)
    else:
        output_valid = np.asarray(valid_mask, dtype=bool)
        if output_valid.shape != target_values.shape[1:]:
            raise ValueError("MBMP valid mask does not match the raster shape")
        target_normalization = output_valid & (target_values[4] != 0)
        reference_normalization = output_valid & (reference_values[4] != 0)
    target_ratio = _normalized_ratio(target_values[4], target_values[5], target_normalization)
    reference_ratio = _normalized_ratio(
        reference_values[4], reference_values[5], reference_normalization
    )
    mbmp = np.full(target_ratio.shape, MBMP_NEUTRAL, dtype=np.float32)
    np.divide(target_ratio, reference_ratio, out=mbmp, where=reference_ratio != 0)
    mbmp[~np.isfinite(mbmp)] = MBMP_NEUTRAL
    np.clip(mbmp, 0.0, MBMP_MAX, out=mbmp)
    mbmp[~output_valid] = MBMP_NEUTRAL
    return mbmp


def _grid_signature(source: rasterio.io.DatasetReader) -> tuple[Any, ...]:
    return (source.width, source.height, source.crs, tuple(source.transform)[:6])


def _read_single_band(path: Path, image_grid: tuple[Any, ...]) -> tuple[np.ndarray, str | None]:
    with rasterio.open(path) as source:
        if source.count != 1:
            raise ValueError(f"Expected one band in {path}, got {source.count}")
        if _grid_signature(source) != image_grid:
            raise ValueError(f"Raster grid does not match image grid: {path}")
        return source.read(1), source.descriptions[0]


def validate_positive_mask(
    mask: np.ndarray, identifier: str, *, allow_empty: bool
) -> np.ndarray:
    """Validate a public positive mask while making empty-label policy explicit."""
    unique = set(int(value) for value in np.unique(mask))
    if not unique.issubset({0, 1}):
        raise ValueError(f"Plume mask is not binary for {identifier}")
    result = mask.astype(bool)
    if not np.any(result) and not allow_empty:
        raise ValueError(f"Positive sample has an empty plume mask: {identifier}")
    return result


def load_sample(
    base_dir: Path,
    record: dict[str, Any],
    *,
    require_enhancement: bool = True,
    allow_empty_positive_mask: bool = False,
    allow_missing_positive_mask: bool = False,
) -> MarsS2Sample:
    """Load and validate one pinned MARS-S2L sample.

    ``require_enhancement=False`` supports detector-only manifests that retain
    plume masks but intentionally omit unit-ambiguous enhancement rasters.
    """
    identifier = record_id(record)
    state = label_state(record)
    paths = role_paths(record)
    required = {"image", "cloud_mask"}
    allowed = set(required)
    missing_positive_truth = (
        state == "PLUME"
        and not bool(record.get("pixel_truth_available", True))
    )
    if state == "PLUME":
        allowed |= {"plume_mask", "methane_enhancement"}
        if missing_positive_truth:
            if not allow_missing_positive_mask:
                raise ValueError(
                    f"Positive sample has no pixel truth: {identifier}"
                )
        else:
            required.add("plume_mask")
        if require_enhancement and not missing_positive_truth:
            required.add("methane_enhancement")
    if not required.issubset(paths) or not set(paths).issubset(allowed):
        raise ValueError(
            f"Asset roles for {identifier} are {sorted(paths)}; required {sorted(required)} "
            f"and allowed {sorted(allowed)}"
        )

    image_path = safe_asset_path(base_dir, paths["image"])
    with rasterio.open(image_path) as source:
        if source.count != 12:
            raise ValueError(f"Expected 12 image bands for {identifier}, got {source.count}")
        descriptions = tuple(source.descriptions)
        validate_image_band_order(record, descriptions)
        if set(source.dtypes) != {"uint16"}:
            raise ValueError(f"Expected uint16 image for {identifier}, got {source.dtypes}")
        raw_pair = source.read()
        image_grid = _grid_signature(source)
        crs = "" if source.crs is None else source.crs.to_string()
        transform = tuple(float(value) for value in source.transform)[:6]

    cloud, _ = _read_single_band(safe_asset_path(base_dir, paths["cloud_mask"]), image_grid)
    observed_cloud_values = set(int(value) for value in np.unique(cloud))
    unknown_cloud_values = observed_cloud_values - set(CLOUD_CLASSES)
    if unknown_cloud_values:
        raise ValueError(
            f"Unknown cloud classes for {identifier}: {sorted(unknown_cloud_values)}"
        )

    raw_target = raw_pair[:6]
    raw_reference = raw_pair[6:]
    radiometric_valid = np.all(raw_target != 0, axis=0) & np.all(raw_reference != 0, axis=0)
    clear = cloud == 0
    observable = radiometric_valid & clear
    reflectance_pair = np.clip(
        raw_pair.astype(np.float32) / REFLECTANCE_DIVISOR, 0.0, REFLECTANCE_MAX
    )
    target = reflectance_pair[:6]
    reference = reflectance_pair[6:]

    enhancement: np.ndarray | None = None
    if state == "PLUME" and not missing_positive_truth:
        plume, _ = _read_single_band(
            safe_asset_path(base_dir, paths["plume_mask"]), image_grid
        )
        plume_mask = validate_positive_mask(
            plume, identifier, allow_empty=allow_empty_positive_mask
        )
        if "methane_enhancement" in paths:
            enhancement, _ = _read_single_band(
                safe_asset_path(base_dir, paths["methane_enhancement"]), image_grid
            )
            enhancement = enhancement.astype(np.float32)
            if not np.all(np.isfinite(enhancement)):
                raise ValueError(f"Enhancement raster contains non-finite values: {identifier}")
    else:
        plume_mask = np.zeros(raw_pair.shape[1:], dtype=bool)

    return MarsS2Sample(
        sample_id=identifier,
        split=str(record["split"]),
        sensor_family=str(
            record.get("sensor_family")
            or ("Sentinel-2" if str(record.get("satellite") or "").startswith("S2") else "Landsat")
        ),
        satellite=str(record.get("satellite") or ""),
        label_state=state,
        location_id=str(record.get("physical_location_id") or record.get("id_location") or ""),
        target_scene_id=str(record.get("target_scene_id") or record.get("scene_id") or ""),
        reference_scene_id=str(
            record.get("reference_scene_id") or record.get("background_scene_id") or ""
        ),
        raw_pair=raw_pair,
        reflectance_pair=reflectance_pair,
        target=target,
        reference=reference,
        cloud_classes=cloud,
        clear_mask=clear,
        radiometric_valid_mask=radiometric_valid,
        observable_mask=observable,
        plume_mask=plume_mask,
        methane_enhancement_raw=enhancement,
        mbmp_release_compatible=compute_mbmp(target, reference),
        mbmp_valid_aware=compute_mbmp(target, reference, valid_mask=observable),
        crs=crs,
        transform=transform,
    )


def load_samples(
    base_dir: Path, records: Iterable[dict[str, Any]]
) -> list[MarsS2Sample]:
    return [load_sample(base_dir, record) for record in records]
