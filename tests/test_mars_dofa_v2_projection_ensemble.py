from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from confirm_mars_dofa_v2_projection_ensemble import (  # noqa: E402
    mean_logit_probabilities,
)


def test_mean_logit_probabilities_matches_geometric_odds_mean() -> None:
    first = np.asarray([0.2, 0.8], dtype=np.float64)
    second = np.asarray([0.5, 0.5], dtype=np.float64)
    observed = mean_logit_probabilities([first, second])
    expected_odds = np.sqrt((first / (1.0 - first)) * (second / (1.0 - second)))
    expected = expected_odds / (1.0 + expected_odds)
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_mean_logit_probabilities_rejects_invalid_collection() -> None:
    with pytest.raises(ValueError, match="At least one"):
        mean_logit_probabilities([])
    with pytest.raises(ValueError, match="identical shapes"):
        mean_logit_probabilities([np.ones(2), np.ones(3)])
