from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from evaluate_mars_dofa_anchored_protected_ensemble import (  # noqa: E402
    protected_residual_ensemble,
)


def test_protected_ensemble_preserves_low_region_and_gate() -> None:
    current = np.asarray([0.1, 0.249, 0.3, 0.8])
    anchored = np.asarray([0.99, 0.99, 0.4, 0.7])
    dofa = np.asarray([0.99, 0.99, 0.3, 0.9])
    result = protected_residual_ensemble(
        current, anchored, dofa, gate=0.25, anchored_multiplier=1.0
    )
    assert np.array_equal(result[:2], current[:2])
    assert np.all(result[2:] >= 0.25)


def test_independent_positive_residuals_combine() -> None:
    current = np.asarray([0.6])
    anchored = np.asarray([0.7])
    dofa = np.asarray([0.8])
    result = protected_residual_ensemble(
        current, anchored, dofa, gate=0.25, anchored_multiplier=0.5
    )
    assert result[0] > dofa[0]


def test_component_crossing_gate_is_rejected() -> None:
    with pytest.raises(ValueError, match="component crossed"):
        protected_residual_ensemble(
            np.asarray([0.6]), np.asarray([0.1]), np.asarray([0.7]),
            gate=0.25, anchored_multiplier=1.0,
        )
