from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/build_cloudsen12_spatial_pilot.py"
SPEC = importlib.util.spec_from_file_location("build_cloudsen_pilot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_haversine_minimum_distance() -> None:
    distance = MODULE.haversine_min_km(
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([0.0]),
        np.asarray([0.0]),
    )
    assert distance[0] == 0.0
    assert 111.0 < distance[1] < 112.0


def test_partition_selection_is_bounded_deterministic_and_balanced() -> None:
    frame = pd.DataFrame(
        {
            "id_loc_image": [f"id-{index}" for index in range(12)],
            "country": ["a", "b", "c"] * 4,
            "MBMP_std": np.arange(12, dtype=float),
            "MBMP_max": np.arange(12, dtype=float) / 2,
        }
    )
    first = MODULE.select_partition(frame, 8, 0.5, "seed")
    second = MODULE.select_partition(frame, 8, 0.5, "seed")
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 8
    assert first["id_loc_image"].nunique() == 8
    assert first["selection_stratum"].value_counts().to_dict() == {
        "hard_mbmp": 4,
        "country_diverse": 4,
    }
    assert set(first.nlargest(4, "MBMP_std")["id_loc_image"]) == {
        "id-8", "id-9", "id-10", "id-11"
    }
