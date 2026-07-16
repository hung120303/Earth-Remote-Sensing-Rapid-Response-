from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_mars_geospatial_site_prior import (  # noqa: E402
    crossfit_prior,
    haversine_km,
)


def test_haversine_one_degree_latitude() -> None:
    distance = haversine_km(np.asarray([[0.0, 0.0]]), np.asarray([[1.0, 0.0]]))
    assert distance.shape == (1, 1)
    assert 111.0 < distance[0, 0] < 112.0


def test_crossfit_prior_never_uses_held_labels() -> None:
    values = {
        "groups": np.asarray(["a", "b", "c", "d"]),
        "labels": np.asarray([1, 0, 1, 0], dtype=np.uint8),
        "folds": np.asarray([0, 1, 2, 3], dtype=np.uint8),
    }
    # Add a fifth fold row because the production contract always crossfits 0..4.
    values = {key: np.append(value, {"groups": "e", "labels": 1, "folds": 4}[key]) for key, value in values.items()}
    coordinates = np.asarray([[0.0, float(index)] for index in range(5)])
    first, _ = crossfit_prior(values, coordinates, neighbors=2, scale_km=500.0)
    changed = {key: value.copy() for key, value in values.items()}
    changed["labels"] = 1 - changed["labels"]
    second, _ = crossfit_prior(changed, coordinates, neighbors=2, scale_km=500.0)
    # Changing every label changes source priors, so isolate each held fold in turn.
    for fold in range(5):
        local = {key: value.copy() for key, value in values.items()}
        local["labels"][local["folds"] == fold] ^= 1
        local_scores, _ = crossfit_prior(local, coordinates, neighbors=2, scale_km=500.0)
        held = values["folds"] == fold
        assert np.array_equal(first[held], local_scores[held])
    assert not np.array_equal(first, second)
