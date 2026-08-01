from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_mars_anchored_full_finetune_pilot import (  # noqa: E402
    resolve_fold_contract,
    write_endpoint_state_cache,
    write_scene_prediction_cache,
)


def raw() -> dict:
    return {
        "labels": [1, 0],
        "sensors": [0, 1],
        "groups": ["g1", "g2"],
        "sample_ids": ["a", "b"],
        "folds": [3, 4],
        "base_scores": [0.8, 0.1],
        "candidate_scores": {"0.1": [0.81, 0.09], "0.5": [0.9, 0.05]},
    }


def test_scene_cache_is_pickle_free_and_identity_aligned(tmp_path: Path) -> None:
    path = tmp_path / "scores.npz"
    receipt = write_scene_prediction_cache(
        path, raw(), [0.1, 0.5], protocol_sha256="a" * 64
    )
    with np.load(path, allow_pickle=False) as cache:
        assert cache["sample_ids"].tolist() == ["a", "b"]
        assert cache["groups"].tolist() == ["g1", "g2"]
        assert cache["candidate_1"].tolist() == [0.9, 0.05]
        assert cache["protocol_sha256"].item() == "a" * 64
    assert receipt["rows"] == 2
    assert receipt["contains_dense_pixels"] is False


def test_scene_cache_rejects_misaligned_candidate(tmp_path: Path) -> None:
    values = raw()
    values["candidate_scores"]["0.5"] = [0.9]
    with pytest.raises(ValueError, match="Candidate scene scores"):
        write_scene_prediction_cache(
            tmp_path / "scores.npz", values, [0.1, 0.5], protocol_sha256="b" * 64
        )


def test_endpoint_state_cache_is_marked_research_only(tmp_path: Path) -> None:
    import torch

    path = tmp_path / "states.pt"
    receipt = write_endpoint_state_cache(
        path,
        {"3": {"weight": torch.tensor([1.0])}},
        strengths=[0.1, 0.5],
        protocol_sha256="c" * 64,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["research_only_until_downstream_gates_pass"] is True
    assert payload["protocol_sha256"] == "c" * 64
    assert receipt["tracked"] is False


def test_explicit_single_holdout_fit_contract() -> None:
    evaluation, authorized, mapping = resolve_fold_contract(
        {"folds": [2], "fit_folds_by_held": {"2": [3, 4]}}
    )
    assert evaluation == {2}
    assert authorized == {2, 3, 4}
    assert mapping == {2: {3, 4}}


@pytest.mark.parametrize(
    "mapping",
    ({"3": [3, 4]}, {"2": []}, {"3": [4]}),
)
def test_invalid_explicit_fit_contract_is_rejected(mapping: dict[str, list[int]]) -> None:
    with pytest.raises(ValueError):
        resolve_fold_contract({"folds": [2], "fit_folds_by_held": mapping})
