"""Analytical Gaussian-plume synthesis for mixed-sensor MARS training.

This module implements the inexpensive plume family used conceptually by
Rouet-Leduc and Hulbert (Nature Communications, 2024): a steady, vertically
integrated Gaussian plume with correlated two-dimensional perturbations.  It
then applies the released MARS-S2L sensor-specific integrated-transmittance LUT
to B11/B12.  The generator is training-only; benchmark images remain real.

The MARS enhancement raster has an unresolved ppm/ppb metadata conflict, so
this implementation deliberately parameterizes spectral strength as peak
delta-CH4 in the units expected by the released LUT.  It does not claim a
physical emission-rate inversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from mars_v4_simulation import MAX_DELTA_CH4_PPB, MarsTransmittanceLut


@dataclass(frozen=True)
class GaussianPlumeParameters:
    """Frozen parameters for one analytical column-enhancement field."""

    source_row: float
    source_col: float
    wind_u_m_s: float
    wind_v_m_s: float
    peak_delta_ch4_ppb: float
    plume_length_m: float
    initial_sigma_m: float
    sigma_at_1km_m: float
    dispersion_exponent: float = 0.894
    turbulence_fraction: float = 0.35
    turbulence_correlation_m: float = 60.0
    mask_peak_fraction: float = 0.05
    wind_direction_offset_degrees: float = 0.0

    def validate(self, image_shape: tuple[int, int]) -> None:
        height, width = image_shape
        if height <= 0 or width <= 0:
            raise ValueError("Image shape must be positive")
        if not (0 <= self.source_row < height and 0 <= self.source_col < width):
            raise ValueError("Plume source must lie inside the image")
        if np.hypot(self.wind_u_m_s, self.wind_v_m_s) < 0.5:
            raise ValueError("Gaussian plume requires wind speed of at least 0.5 m/s")
        if not (0 < self.peak_delta_ch4_ppb <= MAX_DELTA_CH4_PPB):
            raise ValueError("Peak delta-CH4 is outside the released LUT safety range")
        if self.plume_length_m <= 0 or self.initial_sigma_m <= 0:
            raise ValueError("Plume length and initial sigma must be positive")
        if self.sigma_at_1km_m <= 0 or self.dispersion_exponent <= 0:
            raise ValueError("Dispersion parameters must be positive")
        if not (0 <= self.turbulence_fraction <= 1.5):
            raise ValueError("Turbulence fraction must be in [0, 1.5]")
        if self.turbulence_correlation_m <= 0:
            raise ValueError("Turbulence correlation length must be positive")
        if not (0 < self.mask_peak_fraction < 1):
            raise ValueError("Mask peak fraction must be in (0, 1)")
        if not (-90 <= self.wind_direction_offset_degrees <= 90):
            raise ValueError("Wind-direction offset must be in [-90, 90] degrees")


@dataclass(frozen=True)
class GaussianPlumeField:
    """Analytical enhancement and its dense training support."""

    delta_ch4: np.ndarray
    mask: np.ndarray
    along_wind_m: np.ndarray
    cross_wind_m: np.ndarray
    parameters: GaussianPlumeParameters


@dataclass(frozen=True)
class SimulatedGaussianPlume:
    """Sensor-aware plume injection result."""

    target: np.ndarray
    mask: np.ndarray
    delta_ch4: np.ndarray
    b11_factor: np.ndarray
    b12_factor: np.ndarray
    parameters: GaussianPlumeParameters


@dataclass(frozen=True)
class GaussianPlumeSamplingRanges:
    """Predeclared bank distribution fit to authorized real-mask geometry."""

    source_margin_fraction: float = 0.075
    peak_delta_ch4_ppb: tuple[float, float] = (500.0, 10_000.0)
    common_length_m: tuple[float, float] = (300.0, 1500.0)
    common_initial_sigma_m: tuple[float, float] = (8.0, 30.0)
    common_sigma_at_1km_m: tuple[float, float] = (60.0, 230.0)
    broad_tail_probability: float = 0.12
    broad_length_m: tuple[float, float] = (1200.0, 2300.0)
    broad_initial_sigma_m: tuple[float, float] = (20.0, 50.0)
    broad_sigma_at_1km_m: tuple[float, float] = (220.0, 400.0)
    turbulence_fraction: tuple[float, float] = (0.15, 0.50)
    turbulence_correlation_m: tuple[float, float] = (40.0, 120.0)
    direction_core_probability: float = 0.90
    direction_core_sigma_degrees: float = 24.0
    direction_tail_abs_degrees: tuple[float, float] = (45.0, 85.0)
    mask_peak_fraction: float = 0.05


def _log_uniform(
    bounds: tuple[float, float], rng: np.random.Generator
) -> float:
    low, high = map(float, bounds)
    if low <= 0 or high <= low:
        raise ValueError("Log-uniform bounds must satisfy 0 < low < high")
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_gaussian_parameters(
    image_shape: tuple[int, int],
    wind: tuple[float, float],
    rng: np.random.Generator,
    ranges: GaussianPlumeSamplingRanges = GaussianPlumeSamplingRanges(),
) -> GaussianPlumeParameters:
    """Sample one deterministic bank member from frozen morphology ranges."""
    height, width = image_shape
    row_margin = float(ranges.source_margin_fraction) * height
    col_margin = float(ranges.source_margin_fraction) * width
    if 2 * row_margin >= height or 2 * col_margin >= width:
        raise ValueError("Source margin leaves no valid image interior")
    broad = bool(rng.random() < float(ranges.broad_tail_probability))
    if broad:
        length = _log_uniform(ranges.broad_length_m, rng)
        initial_sigma = float(rng.uniform(*ranges.broad_initial_sigma_m))
        sigma_at_1km = _log_uniform(ranges.broad_sigma_at_1km_m, rng)
    else:
        length = _log_uniform(ranges.common_length_m, rng)
        initial_sigma = float(rng.uniform(*ranges.common_initial_sigma_m))
        sigma_at_1km = _log_uniform(ranges.common_sigma_at_1km_m, rng)
    if rng.random() < float(ranges.direction_core_probability):
        direction_offset = float(
            np.clip(
                rng.normal(0.0, float(ranges.direction_core_sigma_degrees)),
                -float(ranges.direction_tail_abs_degrees[1]),
                float(ranges.direction_tail_abs_degrees[1]),
            )
        )
    else:
        direction_offset = float(
            rng.choice((-1.0, 1.0))
            * rng.uniform(*ranges.direction_tail_abs_degrees)
        )
    return GaussianPlumeParameters(
        source_row=float(rng.uniform(row_margin, height - row_margin)),
        source_col=float(rng.uniform(col_margin, width - col_margin)),
        wind_u_m_s=float(wind[0]),
        wind_v_m_s=float(wind[1]),
        peak_delta_ch4_ppb=_log_uniform(ranges.peak_delta_ch4_ppb, rng),
        plume_length_m=length,
        initial_sigma_m=initial_sigma,
        sigma_at_1km_m=sigma_at_1km,
        turbulence_fraction=float(rng.uniform(*ranges.turbulence_fraction)),
        turbulence_correlation_m=_log_uniform(
            ranges.turbulence_correlation_m, rng
        ),
        mask_peak_fraction=float(ranges.mask_peak_fraction),
        wind_direction_offset_degrees=direction_offset,
    )


def _colored_noise(
    shape: tuple[int, int],
    correlation_pixels: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return deterministic zero-mean, unit-scale 2-D correlated noise."""
    white = rng.standard_normal(shape).astype(np.float32)
    colored = ndimage.gaussian_filter(
        white,
        sigma=max(float(correlation_pixels) / 2.0, 0.5),
        mode="reflect",
    )
    colored -= float(np.mean(colored))
    scale = float(np.std(colored))
    if scale < 1e-6:
        return np.zeros(shape, dtype=np.float32)
    return np.asarray(colored / scale, dtype=np.float32)


def analytical_gaussian_plume(
    image_shape: tuple[int, int],
    resolution_m: float,
    parameters: GaussianPlumeParameters,
    rng: np.random.Generator,
) -> GaussianPlumeField:
    """Generate a wind-aligned vertically integrated Gaussian plume.

    ``wind_u`` is positive east and ``wind_v`` positive north. Array rows grow
    southward, so the northing coordinate carries the opposite row sign. The
    cross-wind width follows ``sigma_y(x) = sigma_0 + sigma_1km *
    (x / 1000)^k``. A finite exponential residence length prevents an
    unphysical infinite tail inside large training chips.
    """
    if not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("Pixel resolution must be positive and finite")
    parameters.validate(image_shape)
    height, width = image_shape
    rows, cols = np.indices((height, width), dtype=np.float32)
    east_m = (cols - float(parameters.source_col)) * float(resolution_m)
    north_m = -(rows - float(parameters.source_row)) * float(resolution_m)
    speed = float(np.hypot(parameters.wind_u_m_s, parameters.wind_v_m_s))
    unit_east = float(parameters.wind_u_m_s) / speed
    unit_north = float(parameters.wind_v_m_s) / speed
    # ERA5/reanalysis wind is an imperfect proxy for instantaneous plume-level
    # transport. A frozen synthetic bank may therefore include a bounded
    # direction offset while retaining the unperturbed wind as model metadata.
    angle = np.radians(float(parameters.wind_direction_offset_degrees))
    cos_angle = float(np.cos(angle))
    sin_angle = float(np.sin(angle))
    unit_east, unit_north = (
        cos_angle * unit_east - sin_angle * unit_north,
        sin_angle * unit_east + cos_angle * unit_north,
    )
    along = east_m * unit_east + north_m * unit_north
    cross = -east_m * unit_north + north_m * unit_east

    downwind = along >= 0
    effective_along = np.maximum(along, float(resolution_m) / 2.0)
    sigma = float(parameters.initial_sigma_m) + float(parameters.sigma_at_1km_m) * np.power(
        effective_along / 1000.0,
        float(parameters.dispersion_exponent),
    )
    # The vertically integrated steady plume is proportional to
    # 1/(u*sigma_y) exp(-y^2/(2*sigma_y^2)). Wind speed is retained here so
    # parameter banks can sample source strength and wind independently; peak
    # normalization below converts the unresolved absolute units to LUT units.
    base = np.exp(-0.5 * np.square(cross / sigma)) / (speed * sigma)
    base *= np.exp(-effective_along / float(parameters.plume_length_m))
    base[~downwind] = 0.0
    maximum = float(np.max(base))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Gaussian plume has no downwind support")
    enhancement = base / maximum

    if parameters.turbulence_fraction > 0:
        noise = _colored_noise(
            image_shape,
            float(parameters.turbulence_correlation_m) / float(resolution_m),
            rng,
        )
        # A log-normal modulation preserves non-negativity. The centering term
        # keeps its expected multiplier near one before the final peak scale.
        strength = float(parameters.turbulence_fraction)
        enhancement *= np.exp(strength * noise - 0.5 * strength * strength)
    peak = float(np.max(enhancement))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("Turbulence removed all plume support")
    enhancement = np.clip(
        enhancement / peak * float(parameters.peak_delta_ch4_ppb),
        0.0,
        MAX_DELTA_CH4_PPB,
    ).astype(np.float32)
    mask = enhancement >= (
        float(parameters.mask_peak_fraction) * float(parameters.peak_delta_ch4_ppb)
    )
    mask &= downwind
    # Colored perturbations can form tiny detached threshold islands. Public
    # MARS masks are overwhelmingly one component; retain the source-connected
    # component represented by the enhancement maximum as the dense target.
    components, _ = ndimage.label(mask)
    peak_row, peak_col = np.unravel_index(int(np.argmax(enhancement)), image_shape)
    peak_component = int(components[peak_row, peak_col])
    if peak_component > 0:
        mask = components == peak_component
    if not np.any(mask):
        raise ValueError("Gaussian plume mask is empty")
    return GaussianPlumeField(
        delta_ch4=enhancement,
        mask=np.asarray(mask, dtype=bool),
        along_wind_m=np.asarray(along, dtype=np.float32),
        cross_wind_m=np.asarray(cross, dtype=np.float32),
        parameters=parameters,
    )


class MarsGaussianPlumeSimulator:
    """Inject analytical plumes through the exact mixed-sensor MARS LUT."""

    def __init__(self, lut_path: Path) -> None:
        self.lut = MarsTransmittanceLut(lut_path)

    def simulate(
        self,
        raw_target: np.ndarray,
        *,
        satellite: str,
        solar_zenith_degrees: float,
        view_zenith_degrees: float,
        resolution_m: float,
        parameters: GaussianPlumeParameters,
        rng: np.random.Generator,
    ) -> SimulatedGaussianPlume:
        """Return a copy with only target B11/B12 attenuated by methane."""
        target = np.asarray(raw_target)
        if target.ndim != 3 or target.shape[0] != 6 or target.dtype != np.uint16:
            raise ValueError("Target must be a six-band uint16 S2/Landsat array")
        field = analytical_gaussian_plume(
            target.shape[-2:], resolution_m, parameters, rng
        )
        b12, b11 = self.lut.transmittance(
            satellite,
            solar_zenith_degrees,
            view_zenith_degrees,
            field.delta_ch4,
        )
        # The analytical tail is nonzero to the image boundary. Exactly neutral
        # factors outside dense support avoid changing nominal negative pixels.
        active = field.delta_ch4 >= (
            0.005 * float(parameters.peak_delta_ch4_ppb)
        )
        b11 = np.where(active, b11, 1.0).astype(np.float32)
        b12 = np.where(active, b12, 1.0).astype(np.float32)
        simulated = target.copy()
        uint16_max = np.iinfo(np.uint16).max
        simulated[4] = np.rint(
            np.clip(target[4].astype(np.float32) * b11, 0, uint16_max)
        ).astype(np.uint16)
        simulated[5] = np.rint(
            np.clip(target[5].astype(np.float32) * b12, 0, uint16_max)
        ).astype(np.uint16)
        return SimulatedGaussianPlume(
            target=simulated,
            mask=field.mask,
            delta_ch4=field.delta_ch4,
            b11_factor=b11,
            b12_factor=b12,
            parameters=parameters,
        )
