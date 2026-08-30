from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mars_sensor_ordinal import MarsSensorOrdinalUNet, monotone_cumulative_logits, pixel_loss
from train_mars_sensor_ordinal import (
    SiteBalancedBatcher,
    _random_valid_crop_origin,
    apply_shared_spatial_transform,
    align_comparator,
    deterministic_inner_split,
    main,
    metric_gates,
    ordinal_levels,
    scene_learning_rate,
    validate_requested_folds,
    verify_protocol,
)


def test_monotone_cumulative_logits_are_decreasing() -> None:
    logits = monotone_cumulative_logits(torch.randn(3, 4, 7, 5))
    assert logits.shape == (3, 4, 7, 5)
    assert torch.all(logits[:, 1:] < logits[:, :-1])


def test_branch_only_quantile_ties_assign_to_lower_level() -> None:
    enhancement = np.asarray([[0.0, 1.0, 2.0, 3.0, 4.0, np.nan]])
    plume = np.asarray([[False, True, True, True, True, True]])
    observable = np.ones_like(plume)
    levels, support = ordinal_levels(enhancement, plume, observable, np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(levels, [[0, 1, 2, 3, 4, 0]])
    np.testing.assert_array_equal(support, [[True, True, True, True, True, False]])


def test_invalid_pixels_have_zero_dense_and_ordinal_weight() -> None:
    output = {"binary_logit": torch.tensor([[[[100.0, -100.0]]]], requires_grad=True), "ordinal_logits": torch.tensor([[[[100.0, -100.0]], [[100.0, -100.0]], [[100.0, -100.0]], [[100.0, -100.0]]]], requires_grad=True)}
    plume = torch.tensor([[[0.0, 1.0]]])
    observable = torch.tensor([[[True, False]]])
    levels = torch.tensor([[[0, 4]]])
    support = torch.tensor([[[True, False]]])
    losses = pixel_loss(output, plume, observable, levels, support)
    losses["loss"].backward()
    assert output["binary_logit"].grad[0, 0, 0, 1] == 0
    assert torch.count_nonzero(output["ordinal_logits"].grad[..., 1]) == 0
    assert torch.isfinite(losses["loss"])


def test_exact_14_channels_and_independent_sensor_stems() -> None:
    model = MarsSensorOrdinalUNet().eval()
    inputs = torch.randn(2, 14, 32, 40)
    observable = torch.ones(2, 1, 32, 40)
    output = model(inputs, torch.tensor([0, 1]), observable)
    assert output["binary_logit"].shape == (2, 1, 32, 40)
    assert output["ordinal_logits"].shape == (2, 4, 32, 40)
    assert output["scene_descriptor"].shape == (2, 26)
    assert model.sensor_stems[0][0].weight.data_ptr() != model.sensor_stems[1][0].weight.data_ptr()
    with pytest.raises(ValueError, match="Bx14"):
        model(torch.randn(1, 13, 16, 16), torch.tensor([0]), torch.ones(1, 1, 16, 16))


def test_deterministic_inner_split_is_group_disjoint_and_repeatable() -> None:
    rows = []
    for label in ("PLUME", "NO_PLUME"):
        for index in range(12):
            rows.append({"group_id": f"{label}-{index}", "label_state": label})
    first = deterministic_inner_split(rows)
    second = deterministic_inner_split(list(reversed(rows)))
    assert first == second
    assert first[0].isdisjoint(first[1])
    validation_labels = {row["label_state"] for row in rows if row["group_id"] in first[1]}
    assert validation_labels == {"PLUME", "NO_PLUME"}


def test_augmentation_uses_one_shared_transform_for_every_spatial_tensor() -> None:
    marker = np.arange(12).reshape(3, 4)
    values = {
        "inputs": np.stack((marker, marker + 100)),
        "observable": marker,
        "plume": marker + 200,
        "ordinal_level": marker + 300,
        "ordinal_support": marker + 400,
    }
    transformed = apply_shared_spatial_transform(
        values, turns=1, horizontal=True, vertical=False
    )
    expected = np.flip(np.rot90(marker, 1), axis=-1)
    np.testing.assert_array_equal(transformed["inputs"][0], expected)
    np.testing.assert_array_equal(transformed["inputs"][1] - 100, expected)
    np.testing.assert_array_equal(transformed["observable"], expected)
    np.testing.assert_array_equal(transformed["plume"] - 200, expected)
    np.testing.assert_array_equal(transformed["ordinal_level"] - 300, expected)
    np.testing.assert_array_equal(transformed["ordinal_support"] - 400, expected)
    assert all(value.flags.c_contiguous for value in transformed.values())


def test_negative_crop_origin_is_random_and_never_violates_support_gate() -> None:
    observable = np.ones((200, 200), dtype=bool)
    observable[:50] = False
    origins = {
        _random_valid_crop_origin(
            observable,
            size=256,
            minimum_valid_fraction=0.70,
            rng=np.random.default_rng(seed),
        )
        for seed in range(8)
    }
    assert len(origins) > 1
    for top, left in origins:
        view = observable[max(top, 0):min(top + 256, 200), max(left, 0):min(left + 256, 200)]
        assert view.mean() >= 0.70
    with pytest.raises(ValueError, match="no 256x256 crop"):
        _random_valid_crop_origin(
            np.zeros((200, 200), dtype=bool),
            size=256,
            minimum_valid_fraction=0.70,
            rng=np.random.default_rng(0),
        )


def test_site_balancing_never_draws_one_group_as_both_labels() -> None:
    records = [
        {"group_id": "mixed", "label_state": "PLUME", "sample_id": "p"},
        {"group_id": "mixed", "label_state": "NO_PLUME", "sample_id": "n-mixed"},
        {"group_id": "negative", "label_state": "NO_PLUME", "sample_id": "n"},
    ]
    batcher = SiteBalancedBatcher(
        Path("unused"), records, np.asarray([1.0, 2.0, 3.0]), np.random.default_rng(4)
    )
    rows = batcher.rows(1, 1)
    assert {row["group_id"] for row in rows} == {"mixed", "negative"}
    assert next(row for row in rows if row["group_id"] == "mixed")["label_state"] == "PLUME"


def test_scene_epoch_five_uses_one_epoch_step_warmup() -> None:
    assert scene_learning_rate(4, 24) == 0.0
    assert scene_learning_rate(5, 24, warmup_step=1, warmup_steps=150) == pytest.approx(1e-3 / 150)
    assert scene_learning_rate(5, 24, warmup_step=150, warmup_steps=150) == pytest.approx(1e-3)
    assert scene_learning_rate(24, 24) == pytest.approx(1e-4)


def test_scene_gradient_is_isolated_from_pixel_network() -> None:
    model = MarsSensorOrdinalUNet()
    output = model(torch.randn(2, 14, 32, 32), torch.tensor([0, 1]), torch.ones(2, 1, 32, 32))
    output["scene_logit"].sum().backward()
    scene_names = ("scene_projection.", "scene_mlp.")
    assert any(parameter.grad is not None for name, parameter in model.named_parameters() if name.startswith(scene_names))
    assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0 for name, parameter in model.named_parameters() if not name.startswith(scene_names))


def test_metric_and_bootstrap_gates_pass_clear_superiority() -> None:
    labels = np.tile([0, 1], 8)
    folds = np.repeat([3, 4], 8)
    sensors = np.tile(["Sentinel-2", "Sentinel-2", "Landsat", "Landsat"], 4)
    groups = np.asarray([f"g{fold}-{index//2}" for index, fold in enumerate(folds)])
    comparator = np.tile([0.6, 0.4], 8)
    candidate = np.tile([0.01, 0.99], 8)
    base_dense = np.tile([[1, 2, 2]], (16, 1))
    candidate_dense = np.tile([[4, 0, 1]], (16, 1))
    result = metric_gates(labels, candidate, comparator, folds, sensors, groups, candidate_dense, base_dense, replicates=20, ap_seed=4, dense_seed=5)
    assert result["passed"]
    assert all(result["checks"].values())


def test_protected_fold_rejection_and_comparator_identity_gate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Protected fold"):
        validate_requested_folds([2, 3])
    candidate = {"sample_ids": np.asarray(["a"]), "folds": np.asarray([3]), "sensors": np.asarray([0]), "groups": np.asarray(["g"])}
    path = tmp_path / "comparator.npz"
    np.savez(path, sample_ids=np.asarray(["a"]), folds=np.asarray([4]), sensors=np.asarray([0]), groups=np.asarray(["g"]), scores=np.asarray([0.5]))
    with pytest.raises(ValueError, match="folds differ"):
        align_comparator(candidate, path)


def test_nonfrozen_held_run_and_frozen_hash_mismatch_are_rejected(tmp_path: Path) -> None:
    protocol_path = ROOT / "configs/mars_sensor_ordinal_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    draft = dict(protocol)
    draft["status"] = "draft_pending_code_and_smoke_hashes"
    with pytest.raises(ValueError, match="requires a frozen protocol"):
        verify_protocol(draft, protocol_path, smoke=False)
    frozen = dict(protocol)
    frozen["status"] = "frozen_before_held_outcomes"
    frozen["trainer"] = {**protocol["trainer"], "sha256": "0" * 64}
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_protocol(frozen, protocol_path, smoke=False)


def test_cli_refuses_to_open_held_outcomes_without_explicit_flag() -> None:
    with pytest.raises(RuntimeError, match="explicit --run-held-folds"):
        main([])
