from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_gaussian_plume import (  # noqa: E402
    GaussianPlumeParameters,
    MarsGaussianPlumeSimulator,
    analytical_gaussian_plume,
    sample_gaussian_parameters,
)


LUT = ROOT / "configs" / "mars_s2_integrated_transmittances.json"


def parameters(**changes: float) -> GaussianPlumeParameters:
    values = {
        "source_row": 48.0,
        "source_col": 24.0,
        "wind_u_m_s": 4.0,
        "wind_v_m_s": 0.0,
        "peak_delta_ch4_ppb": 4000.0,
        "plume_length_m": 900.0,
        "initial_sigma_m": 20.0,
        "sigma_at_1km_m": 140.0,
        "turbulence_fraction": 0.3,
        "turbulence_correlation_m": 60.0,
    }
    values.update(changes)
    return GaussianPlumeParameters(**values)


def test_gaussian_field_is_downwind_and_widens() -> None:
    field = analytical_gaussian_plume(
        (96, 128), 10.0, parameters(turbulence_fraction=0.0), np.random.default_rng(3)
    )
    assert field.delta_ch4.shape == (96, 128)
    assert field.mask.dtype == np.bool_
    assert not np.any(field.mask[:, :24])
    assert np.any(field.mask[:, 25:])
    near_width = np.count_nonzero(field.mask[:, 35])
    far_width = np.count_nonzero(field.mask[:, 90])
    assert far_width > near_width


def test_colored_turbulence_is_seed_deterministic() -> None:
    first = analytical_gaussian_plume(
        (96, 128), 10.0, parameters(), np.random.default_rng(71)
    )
    second = analytical_gaussian_plume(
        (96, 128), 10.0, parameters(), np.random.default_rng(71)
    )
    third = analytical_gaussian_plume(
        (96, 128), 10.0, parameters(), np.random.default_rng(72)
    )
    assert np.array_equal(first.delta_ch4, second.delta_ch4)
    assert not np.array_equal(first.delta_ch4, third.delta_ch4)
    assert np.all(first.delta_ch4 >= 0)
    assert np.isclose(float(first.delta_ch4.max()), 4000.0)
    assert ndimage.label(first.mask)[1] == 1


def test_bounded_wind_offset_models_reanalysis_direction_error() -> None:
    no_offset = analytical_gaussian_plume(
        (96, 128), 10.0, parameters(turbulence_fraction=0.0), np.random.default_rng(5)
    )
    offset = analytical_gaussian_plume(
        (96, 128),
        10.0,
        parameters(turbulence_fraction=0.0, wind_direction_offset_degrees=45.0),
        np.random.default_rng(5),
    )
    assert not np.array_equal(no_offset.mask, offset.mask)
    assert np.any(offset.mask[:48, 25:])


def test_northward_wind_respects_array_row_sign() -> None:
    field = analytical_gaussian_plume(
        (96, 96),
        10.0,
        parameters(
            source_row=70.0,
            source_col=48.0,
            wind_u_m_s=0.0,
            wind_v_m_s=3.0,
            turbulence_fraction=0.0,
        ),
        np.random.default_rng(9),
    )
    assert not np.any(field.mask[71:, :])
    assert np.any(field.mask[:70, :])


def test_sensor_aware_injection_changes_only_swir() -> None:
    raw = np.full((6, 96, 128), 5000, dtype=np.uint16)
    simulator = MarsGaussianPlumeSimulator(LUT)
    s2 = simulator.simulate(
        raw,
        satellite="S2A",
        solar_zenith_degrees=30.0,
        view_zenith_degrees=5.0,
        resolution_m=10.0,
        parameters=parameters(turbulence_fraction=0.0),
        rng=np.random.default_rng(11),
    )
    landsat = simulator.simulate(
        raw,
        satellite="LC08",
        solar_zenith_degrees=30.0,
        view_zenith_degrees=5.0,
        resolution_m=10.0,
        parameters=parameters(turbulence_fraction=0.0),
        rng=np.random.default_rng(11),
    )
    assert s2.target.dtype == np.uint16
    assert np.array_equal(s2.target[:4], raw[:4])
    assert np.any(s2.target[5] != raw[5])
    assert np.any(landsat.target[5] != raw[5])
    assert not np.array_equal(s2.b12_factor, landsat.b12_factor)
    assert np.all(s2.b12_factor[s2.delta_ch4 == 0] == 1.0)


def test_invalid_calm_wind_is_rejected() -> None:
    try:
        analytical_gaussian_plume(
            (64, 64),
            10.0,
            parameters(wind_u_m_s=0.1, wind_v_m_s=0.1),
            np.random.default_rng(1),
        )
    except ValueError as error:
        assert "wind speed" in str(error)
    else:
        raise AssertionError("Calm-wind plume should have been rejected")


def test_parameter_bank_sampling_is_bounded_and_deterministic() -> None:
    first_rng = np.random.default_rng(99)
    second_rng = np.random.default_rng(99)
    first = sample_gaussian_parameters((200, 200), (3.0, -2.0), first_rng)
    second = sample_gaussian_parameters((200, 200), (3.0, -2.0), second_rng)
    assert first == second
    assert 15 <= first.source_row <= 185
    assert 15 <= first.source_col <= 185
    assert 500 <= first.peak_delta_ch4_ppb <= 10_000
    assert -85 <= first.wind_direction_offset_degrees <= 85
