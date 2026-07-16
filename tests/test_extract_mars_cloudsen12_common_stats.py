from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/extract_mars_cloudsen12_common_stats.py"
SPEC = importlib.util.spec_from_file_location("extract_cloudsen", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_feature_frame_fills_absent_cloud_class_only(tmp_path: Path) -> None:
    source = tmp_path / "stats.csv"
    pd.DataFrame(
        {
            "id_loc_image": ["a", "b"],
            "wind_u": [1.0, 2.0],
            "cloudmask_0.0": [40_000.0, np.nan],
            "cloudmask_1.0": [np.nan, 40_000.0],
        }
    ).to_csv(source, index=False)
    names = ["wind_u", "cloudmask_0.0", "cloudmask_1.0"]

    frame = MODULE.feature_frame(source, names)

    assert frame.loc["a", "cloudmask_1.0"] == 0.0
    assert frame.loc["b", "cloudmask_0.0"] == 0.0
    np.testing.assert_allclose(
        MODULE.ordered_features(frame, np.asarray(["b", "a"]), names),
        [[2.0, 0.0, 40_000.0], [1.0, 40_000.0, 0.0]],
    )


def test_ordered_features_rejects_missing_identifier() -> None:
    frame = pd.DataFrame({"x": [1.0]}, index=["present"])
    with pytest.raises(ValueError, match="Missing 1"):
        MODULE.ordered_features(frame, np.asarray(["missing"]), ["x"])


def test_cloud_metadata_identity_can_differ_from_statistics_key() -> None:
    frame = pd.DataFrame({"x": [3.0]}, index=["ROI_00001__date_tile"])
    values = MODULE.ordered_features(
        frame, np.asarray(["ROI_00001__date_tile"]), ["x"]
    )
    np.testing.assert_allclose(values, [[3.0]])
