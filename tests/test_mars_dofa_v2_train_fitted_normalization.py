from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from confirm_mars_dofa_v2_train_fitted_normalization import (  # noqa: E402
    source_fitted_normalize,
)


def test_global_normalization_uses_only_source_statistics() -> None:
    source = np.asarray([[0.0, 0.0], [2.0, 4.0]], dtype=np.float32)
    target = np.asarray([[101.0, 102.0], [103.0, 106.0]], dtype=np.float32)
    normalized_source, normalized_target = source_fitted_normalize(
        source,
        target,
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        mode="global_train_fitted",
    )
    np.testing.assert_allclose(normalized_source.mean(0), 0.0, atol=1e-7)
    np.testing.assert_allclose(normalized_source.std(0), 1.0, atol=1e-7)
    np.testing.assert_allclose(normalized_target, [[100.0, 50.0], [102.0, 52.0]])
    assert not np.allclose(normalized_target.mean(0), 0.0)


def test_sensor_normalization_applies_each_source_sensor_transform() -> None:
    source = np.asarray([[0.0], [2.0], [10.0], [14.0]], dtype=np.float32)
    target = np.asarray([[3.0], [18.0]], dtype=np.float32)
    normalized_source, normalized_target = source_fitted_normalize(
        source,
        target,
        np.asarray([0, 0, 1, 1]),
        np.asarray([0, 1]),
        mode="sensor_train_fitted",
    )
    np.testing.assert_allclose(normalized_source[:, 0], [-1.0, 1.0, -1.0, 1.0])
    np.testing.assert_allclose(normalized_target[:, 0], [2.0, 3.0])


def test_sensor_normalization_rejects_unseen_sensor() -> None:
    with pytest.raises(ValueError, match="absent from fit"):
        source_fitted_normalize(
            np.asarray([[0.0], [1.0]]),
            np.asarray([[2.0]]),
            np.asarray([0, 0]),
            np.asarray([1]),
            mode="sensor_train_fitted",
        )
