from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_mars_dofa_v2_protected_fusion import protected_logit_blend  # noqa: E402


def test_protected_logit_blend_keeps_operating_region_separate() -> None:
    current = np.asarray([0.05, 0.2, 0.499, 0.5, 0.7, 0.95])
    dofa = np.asarray([0.99, 0.99, 0.99, 0.01, 0.8, 0.2])
    observed = protected_logit_blend(current, dofa, gate=0.5, weight=0.2)
    np.testing.assert_array_equal(observed[:3], current[:3])
    assert np.all(observed[3:] >= 0.5)
    assert float(observed[:3].max()) < float(observed[3:].min())


def test_protected_logit_blend_weight_zero_is_identity() -> None:
    current = np.asarray([0.1, 0.5, 0.75, 0.9])
    observed = protected_logit_blend(
        current, np.asarray([0.9, 0.1, 0.2, 0.3]), gate=0.5, weight=0.0
    )
    np.testing.assert_allclose(observed, current, rtol=0.0, atol=1e-16)


def test_protected_logit_blend_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        protected_logit_blend(np.ones(2), np.ones(3), gate=0.5, weight=0.1)
    with pytest.raises(ValueError, match="strictly inside"):
        protected_logit_blend(np.ones(2), np.ones(2), gate=1.0, weight=0.1)
    with pytest.raises(ValueError, match="finite probabilities"):
        protected_logit_blend(
            np.asarray([0.2, np.nan]), np.ones(2), gate=0.5, weight=0.1
        )
