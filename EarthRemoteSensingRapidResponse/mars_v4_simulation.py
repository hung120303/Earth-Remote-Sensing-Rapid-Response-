"""Radiative-transfer plume injection for ERSRR v4 training.

The interpolation contract follows UNEP-IMEO MARS-S2L's public
``TransmittanceCH4InterpolationFromDict`` implementation.  This local module
keeps the training path small and deterministic while using the exact released
integrated-transmittance lookup table.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import interpolate, ndimage
from skimage.transform import rotate

DEFAULT_BACKGROUND_CH4_PPB = 1895.0
MAX_DELTA_CH4_PPB = 16_000.0


def air_mass_factor(solar_zenith_degrees: float, view_zenith_degrees: float) -> float:
    """Return the MARS-S2L geometric air-mass factor."""
    return float(
        1.0 / np.cos(np.radians(view_zenith_degrees))
        + 1.0 / np.cos(np.radians(solar_zenith_degrees))
    )


def counterclockwise_wind_angle(
    source_wind: tuple[float, float], target_wind: tuple[float, float]
) -> float:
    """Return the signed rotation from source-plume wind to target-scene wind."""
    source = np.asarray(source_wind, dtype=np.float64)
    target = np.asarray(target_wind, dtype=np.float64)
    if source.shape != (2,) or target.shape != (2,):
        raise ValueError("Wind vectors must each have two components")
    if np.linalg.norm(source) < 1e-6 or np.linalg.norm(target) < 1e-6:
        return 0.0
    dot = float(np.dot(source, target))
    determinant = float(source[0] * target[1] - source[1] * target[0])
    return float(np.degrees(np.arctan2(determinant, dot)))


def injection_slices(
    image_shape: tuple[int, int],
    plume_shape: tuple[int, int],
    upper_left: tuple[int, int],
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Return clipped image/plume slices for an injection that may cross an edge."""
    image_height, image_width = image_shape
    plume_height, plume_width = plume_shape
    start_row_image = max(0, upper_left[0])
    start_col_image = max(0, upper_left[1])
    end_row_image = min(image_height, upper_left[0] + plume_height)
    end_col_image = min(image_width, upper_left[1] + plume_width)
    if end_row_image <= start_row_image or end_col_image <= start_col_image:
        raise ValueError("Plume injection does not intersect the target image")
    start_row_plume = start_row_image - upper_left[0]
    start_col_plume = start_col_image - upper_left[1]
    end_row_plume = start_row_plume + (end_row_image - start_row_image)
    end_col_plume = start_col_plume + (end_col_image - start_col_image)
    return (
        (slice(start_row_image, end_row_image), slice(start_col_image, end_col_image)),
        (slice(start_row_plume, end_row_plume), slice(start_col_plume, end_col_plume)),
    )


@dataclass(frozen=True)
class SimulatedPlume:
    target: np.ndarray
    mask: np.ndarray
    delta_ch4: np.ndarray
    scale: float
    rotation_degrees: float


class MarsTransmittanceLut:
    """Interpolate B11/B12 transmittances from the public MARS-S2L LUT."""

    def __init__(self, path: Path) -> None:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        self.path = path
        self.amf = np.asarray(data["amf_arr"], dtype=np.float64)
        self.methane = np.asarray(data["mr_ch4_arr"], dtype=np.float64)
        self.background = float(data.get("background_concentration", DEFAULT_BACKGROUND_CH4_PPB))
        self.satellites = {
            key: {
                name: np.asarray(value, dtype=np.float64)
                for name, value in payload.items()
            }
            for key, payload in data.items()
            if key.startswith("S2") and isinstance(payload, dict)
        }
        if not self.satellites:
            raise ValueError("Transmittance LUT contains no Sentinel-2 entries")

    def transmittance(
        self,
        satellite: str,
        solar_zenith_degrees: float,
        view_zenith_degrees: float,
        delta_ch4_ppb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if satellite not in self.satellites:
            raise ValueError(f"Unsupported Sentinel-2 platform: {satellite}")
        amf = min(
            air_mass_factor(solar_zenith_degrees, view_zenith_degrees),
            float(np.max(self.amf)),
        )
        values = self.satellites[satellite]
        methane_at_amf = interpolate.interp1d(
            self.amf, self.methane, axis=0, kind="cubic"
        )(amf)
        b12_at_amf = interpolate.interp1d(
            self.amf, values["transmittance_b12"], axis=0, kind="cubic"
        )(amf)
        b11_at_amf = interpolate.interp1d(
            self.amf, values["transmittance_b11"], axis=0, kind="cubic"
        )(amf)
        b12_background = interpolate.interp1d(
            self.amf, values["transmittance_b12_bg"], axis=0, kind="cubic"
        )(amf)
        b11_background = interpolate.interp1d(
            self.amf, values["transmittance_b11_bg"], axis=0, kind="cubic"
        )(amf)
        b12 = interpolate.interp1d(
            methane_at_amf,
            b12_at_amf,
            axis=0,
            fill_value="extrapolate",
            kind="cubic",
        )(np.asarray(delta_ch4_ppb, dtype=np.float64) + self.background)
        b11 = interpolate.interp1d(
            b12_at_amf,
            b11_at_amf,
            axis=0,
            fill_value="extrapolate",
            kind="cubic",
        )(b12)
        return (
            np.asarray(b12 / b12_background, dtype=np.float32),
            np.asarray(b11 / b11_background, dtype=np.float32),
        )


class MarsPlumeSimulator:
    """Inject a real enhancement field into a no-plume Sentinel-2 target."""

    def __init__(self, lut_path: Path, padding: int = 20) -> None:
        self.lut = MarsTransmittanceLut(lut_path)
        self.padding = padding

    def simulate(
        self,
        raw_target: np.ndarray,
        ch4: np.ndarray,
        plume_mask: np.ndarray,
        *,
        source_wind: tuple[float, float],
        target_wind: tuple[float, float],
        satellite: str,
        solar_zenith_degrees: float,
        view_zenith_degrees: float,
        rng: np.random.Generator,
    ) -> SimulatedPlume:
        target = np.asarray(raw_target)
        enhancement = np.asarray(ch4, dtype=np.float32)
        mask = np.asarray(plume_mask, dtype=bool)
        if target.ndim != 3 or target.shape[0] != 6 or target.dtype != np.uint16:
            raise ValueError("Target must be a six-band uint16 Sentinel-2 array")
        if enhancement.shape != mask.shape or enhancement.ndim != 2 or not np.any(mask):
            raise ValueError("CH4 enhancement and non-empty mask must be matching 2D arrays")

        distance = ndimage.distance_transform_edt(mask)
        mean_distance = max(float(np.mean(distance[mask])), 1.0)
        scale = float(rng.uniform(0.5, 1.5))
        enhancement = np.clip(
            enhancement * np.clip(distance / mean_distance, 0.0, 1.0) * scale,
            0.0,
            MAX_DELTA_CH4_PPB,
        )
        angle = counterclockwise_wind_angle(source_wind, target_wind)
        if abs(angle) > 1.0:
            enhancement = rotate(
                enhancement,
                angle=angle,
                resize=True,
                order=1,
                mode="constant",
                cval=0,
                preserve_range=True,
            ).astype(np.float32)
            mask = rotate(
                mask,
                angle=angle,
                resize=True,
                order=0,
                mode="constant",
                cval=False,
                preserve_range=True,
            ).astype(bool)

        b12, b11 = self.lut.transmittance(
            satellite,
            solar_zenith_degrees,
            view_zenith_degrees,
            enhancement,
        )
        height, width = target.shape[-2:]
        if height <= 2 * self.padding or width <= 2 * self.padding:
            raise ValueError("Target is too small for the configured injection padding")
        injection_center = (
            int(rng.integers(self.padding, height - self.padding)),
            int(rng.integers(self.padding, width - self.padding)),
        )
        upper_left = (
            injection_center[0] - mask.shape[0] // 2,
            injection_center[1] - mask.shape[1] // 2,
        )
        image_slice, plume_slice = injection_slices(
            (height, width), mask.shape, upper_left
        )
        full_mask = np.zeros((height, width), dtype=bool)
        full_ch4 = np.zeros((height, width), dtype=np.float32)
        full_b11 = np.ones((height, width), dtype=np.float32)
        full_b12 = np.ones((height, width), dtype=np.float32)
        full_mask[image_slice] = mask[plume_slice]
        full_ch4[image_slice] = enhancement[plume_slice]
        full_b11[image_slice] = b11[plume_slice]
        full_b12[image_slice] = b12[plume_slice]
        if not np.any(full_mask):
            raise ValueError("Simulated plume was clipped out of the target")

        simulated = target.copy()
        simulated[4] = np.round(simulated[4].astype(np.float32) * full_b11).astype(np.uint16)
        simulated[5] = np.round(simulated[5].astype(np.float32) * full_b12).astype(np.uint16)
        return SimulatedPlume(
            target=simulated,
            mask=full_mask,
            delta_ch4=full_ch4,
            scale=scale,
            rotation_degrees=angle,
        )
