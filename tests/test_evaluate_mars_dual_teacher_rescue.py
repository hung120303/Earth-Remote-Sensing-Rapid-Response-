from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from evaluate_mars_dual_teacher_rescue import anchor_score, rescue_scores  # noqa: E402


def test_anchor_scores_follow_frozen_contract() -> None:
    primary = np.asarray([0.25, 0.81])
    released = np.asarray([1.0, 0.49])
    top = np.asarray([0.64, 1.0])
    np.testing.assert_allclose(
        anchor_score("geometric_dense", primary, released, top), [0.5, 0.63]
    )
    np.testing.assert_allclose(
        anchor_score("conservative_dense", primary, released, top), [0.25, 0.49]
    )
    np.testing.assert_allclose(
        anchor_score("topology_consensus", primary, released, top),
        np.cbrt(primary * released * top),
    )


def test_rescue_only_raises_inside_route() -> None:
    champion = np.asarray([0.1, 0.3, 0.1])
    released = np.asarray([0.8, 0.8, 0.4])
    anchor = np.asarray([0.6, 0.9, 0.9])
    candidate, route = rescue_scores(champion, released, anchor, weight=0.5)
    np.testing.assert_array_equal(route, [True, False, False])
    np.testing.assert_allclose(candidate, [0.3, 0.3, 0.1])
    assert np.all(candidate >= champion)


def test_rescue_rejects_unknown_anchor_and_invalid_weight() -> None:
    values = np.asarray([0.5])
    with pytest.raises(ValueError, match="Unknown rescue anchor"):
        anchor_score("unknown", values, values, values)
    with pytest.raises(ValueError, match="Rescue weight"):
        rescue_scores(values, values, values, weight=0.0)
