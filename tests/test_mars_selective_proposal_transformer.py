from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mars_selective_proposal_transformer import SelectiveProposalTransformer  # noqa: E402
from train_mars_selective_proposal_transformer import (  # noqa: E402
    balanced_cell_weights,
    candidate_scores,
    seed_everything,
    validate_crossfit_alignment,
    validate_spatial_cache,
)


def test_selective_transformer_forward_contract() -> None:
    model = SelectiveProposalTransformer(dropout=0.0)
    values = torch.zeros(2, 9, 64, 64)
    proposal = torch.full((2, 1, 64, 64), 0.1)
    observable = torch.ones_like(proposal)
    logits = model(values, torch.tensor([0, 1]), proposal, observable)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_selective_transformer_rejects_wrong_shape() -> None:
    model = SelectiveProposalTransformer()
    with pytest.raises(ValueError, match="frozen schema"):
        model(
            torch.zeros(1, 8, 64, 64),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, 1, 64, 64),
        )


def test_balanced_weights_assign_equal_label_mass() -> None:
    labels = np.asarray([0, 0, 0, 1], dtype=np.uint8)
    weights = balanced_cell_weights(
        np.asarray(["a", "a", "b", "c"]),
        labels,
        np.asarray([0, 0, 1, 0], dtype=np.uint8),
    )
    assert weights[labels == 0].sum() == pytest.approx(weights[labels == 1].sum())


def test_candidate_scores_only_raise_routed_rows() -> None:
    champion = np.asarray([0.1, 0.3, 0.1])
    released = np.asarray([0.8, 0.8, 0.4])
    verifier = np.asarray([0.9, 0.9, 0.9])
    candidate, route = candidate_scores(champion, released, verifier, 0.5)
    np.testing.assert_array_equal(route, [True, False, False])
    np.testing.assert_allclose(candidate, [0.45, 0.3, 0.1])
    assert np.all(candidate >= champion)


def test_seed_configuration_requires_deterministic_algorithms() -> None:
    seed_everything(17)
    assert torch.are_deterministic_algorithms_enabled()


def test_spatial_cache_validation_checks_schema_and_binding() -> None:
    images = np.zeros((2, 9, 64, 64), dtype=np.float16)
    names = np.asarray(
        [
            "released_probability_meanpool",
            "released_probability_maxpool",
            "mbmp_centered",
            "target_reference_B11_difference",
            "target_reference_B12_difference",
            "target_reference_B11_normalized_difference",
            "target_reference_B12_normalized_difference",
            "cloud_fraction",
            "observable_fraction",
        ]
    )
    validate_spatial_cache(images, np.asarray(["a", "b"]), names, "abc", "abc")
    with pytest.raises(ValueError, match="not bound"):
        validate_spatial_cache(images, np.asarray(["a", "b"]), names, "bad", "abc")


def test_crossfit_alignment_rejects_group_crossing_folds() -> None:
    ids = np.asarray(["a", "b"])
    labels = np.asarray([0, 1], dtype=np.uint8)
    sensors = np.asarray([0, 1], dtype=np.uint8)
    groups = np.asarray(["shared", "shared"])
    folds = np.asarray([3, 4], dtype=np.uint8)
    with pytest.raises(ValueError, match="crosses"):
        validate_crossfit_alignment(
            ids,
            labels,
            sensors,
            groups,
            folds,
            labels,
            sensors,
            groups,
            folds,
            source="test",
        )
