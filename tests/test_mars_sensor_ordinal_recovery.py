from __future__ import annotations

import copy
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_mars_sensor_ordinal import (  # noqa: E402
    AccessLedger,
    RECOVERY_SCHEMA_VERSION,
    RecoveryStore,
    _nested_exact_equal,
    authorize_comparator_decode,
    capture_rng_state,
    evaluate_or_reuse_prediction_part,
    preflight_comparator_hashes,
    restore_rng_state,
    scene_learning_rate,
    seal_or_reuse_prediction_part,
)


def _optimizers() -> tuple[torch.nn.Module, torch.optim.AdamW, torch.optim.AdamW]:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    pixel = torch.optim.AdamW(model[0].parameters(), lr=0.0123, weight_decay=1e-4)
    scene = torch.optim.AdamW(model[1].parameters(), lr=0.0456, weight_decay=1e-4)
    x = torch.tensor([[0.25, -0.5]])
    model(x).sum().backward()
    pixel.step()
    scene.step()
    return model, pixel, scene


def _payload(identity: dict[str, object], *, completed_epoch: int = 4) -> dict[str, object]:
    model, pixel, scene = _optimizers()
    live = {name: value.detach().clone() for name, value in model.state_dict().items()}
    best = {name: value.detach().clone() - 1 for name, value in model.state_dict().items()}
    batcher = SimpleNamespace(rng=np.random.default_rng(123))
    pixel_state = copy.deepcopy(pixel.state_dict())
    scene_state = copy.deepcopy(scene.state_dict())
    identity_ledger = identity.get("access_ledger")
    comparator_hashed = bool(
        identity_ledger.get("comparator_integrity_bytes_hashed", False)
        if isinstance(identity_ledger, dict)
        else False
    )
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "identity": identity,
        "live_model_state": live,
        "best_model_state": best,
        "pixel_optimizer_state": pixel_state,
        "scene_optimizer_state": scene_state,
        "pixel_optimizer_param_groups": copy.deepcopy(pixel_state["param_groups"]),
        "scene_optimizer_param_groups": copy.deepcopy(scene_state["param_groups"]),
        "pixel_optimizer_lrs": [0.0123],
        "scene_optimizer_lrs": [0.0456],
        "best_rank": [0.7, 0.2, -0.4],
        "best_epoch": 3,
        "history": [{"epoch": epoch} for epoch in range(1, completed_epoch + 1)],
        "cutpoints": np.asarray([1.0, 2.0, 3.0]),
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "held_fold": 3,
        "fit_fold": 4,
        "rng_state": capture_rng_state(batcher),
        "access_ledger": AccessLedger(
            comparator_integrity_bytes_hashed=comparator_hashed,
            inner_validation_outcomes_opened=True,
        ).snapshot(),
    }


def _part(fold: int) -> dict[str, np.ndarray]:
    return {
        "sample_ids": np.asarray([f"sample-{fold}"]),
        "labels": np.asarray([1], dtype=np.uint8),
        "sensors": np.asarray([0], dtype=np.uint8),
        "groups": np.asarray([f"group-{fold}"]),
        "folds": np.asarray([fold], dtype=np.uint8),
        "scores": np.asarray([0.25], dtype=np.float64),
        "dense_counts": np.asarray([[1, 2, 3]], dtype=np.int64),
    }


def _binding(fold: int) -> dict[str, object]:
    import train_mars_sensor_ordinal as trainer

    return {
        "protocol_identity": "p",
        "scientific_digest": "s",
        "held_fold": fold,
        "fit_fold": 7 - fold,
        "endpoint_recovery_identity_sha256": f"endpoint-{fold}",
        "ordered_held_records_sha256": f"records-{fold}",
        "ordered_sample_ids_sha256": trainer.canonical_json_hash([f"sample-{fold}"]),
        "access_before_open": {
            "comparator_integrity_bytes_hashed": True,
            "comparator_values_decoded": False,
            "folds_0_1_2_opened": False,
            "external_or_official_evidence_opened": False,
        },
    }


def test_complete_state_coverage_and_live_best_optimizer_lr_roundtrip(tmp_path: Path) -> None:
    identity = {"runtime_signature": {"platform": "Windows"}, "scientific_digest": "frozen"}
    payload = _payload(identity)
    store = RecoveryStore(tmp_path, identity, torch.device("cpu"))
    store.save(payload)
    loaded = store.load()
    assert loaded is not None
    assert not _nested_exact_equal(loaded["live_model_state"], loaded["best_model_state"])
    assert loaded["pixel_optimizer_param_groups"] == loaded["pixel_optimizer_state"]["param_groups"]
    assert loaded["scene_optimizer_param_groups"] == loaded["scene_optimizer_state"]["param_groups"]
    assert loaded["pixel_optimizer_lrs"] == [0.0123]
    assert loaded["scene_optimizer_lrs"] == [0.0456]
    incomplete = _payload(identity)
    incomplete.pop("scene_optimizer_state")
    with pytest.raises(RuntimeError, match="missing required state"):
        store.save(incomplete)


def test_batcher_generator_and_all_global_rngs_restore_exactly() -> None:
    random.seed(10)
    np.random.seed(11)
    torch.manual_seed(12)
    batcher = SimpleNamespace(rng=np.random.default_rng(13))
    state = capture_rng_state(batcher)
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
        batcher.rng.integers(0, 1000, size=8),
    )
    for _ in range(7):
        random.random()
        np.random.random()
        torch.rand(1)
        batcher.rng.integers(0, 1000)
    restore_rng_state(state, batcher)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
        batcher.rng.integers(0, 1000, size=8),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert np.array_equal(actual[3], expected[3])


def test_interrupted_continuation_matches_uninterrupted_through_epoch_five_warmup(tmp_path: Path) -> None:
    def construct(seed: int):
        torch.manual_seed(seed)
        model = torch.nn.ParameterDict({
            "pixel": torch.nn.Parameter(torch.tensor([0.5])),
            "scene": torch.nn.Parameter(torch.tensor([-0.25])),
        })
        pixel = torch.optim.AdamW([model["pixel"]], lr=0.0, weight_decay=1e-4)
        scene = torch.optim.AdamW([model["scene"]], lr=0.0, weight_decay=1e-4)
        batcher = SimpleNamespace(rng=np.random.default_rng(seed))
        return model, pixel, scene, batcher

    def advance(model, pixel, scene, batcher, start: int, stop: int):
        observed: list[tuple[int, float]] = []
        for epoch in range(start, stop + 1):
            pixel.param_groups[0]["lr"] = 3e-4
            pixel.zero_grad()
            noise = random.random() + float(np.random.random()) + float(torch.rand(())) + float(batcher.rng.random())
            (model["pixel"] * noise).sum().backward()
            pixel.step()
            if epoch >= 5:
                for step in range(1, 4):
                    lr = scene_learning_rate(epoch, 24, warmup_step=step if epoch == 5 else None, warmup_steps=3)
                    scene.param_groups[0]["lr"] = lr
                    scene.zero_grad()
                    (model["scene"] * (step + float(batcher.rng.random()))).sum().backward()
                    scene.step()
                    observed.append((epoch, lr))
        return observed

    random.seed(21)
    np.random.seed(22)
    torch.manual_seed(23)
    model, pixel, scene, batcher = construct(24)
    advance(model, pixel, scene, batcher, 1, 4)
    identity = {"runtime_signature": {"platform": "Windows", "driver": "595.79"}}
    payload = _payload(identity, completed_epoch=4)
    payload.update({
        "live_model_state": copy.deepcopy(model.state_dict()),
        "best_model_state": copy.deepcopy(model.state_dict()),
        "pixel_optimizer_state": copy.deepcopy(pixel.state_dict()),
        "scene_optimizer_state": copy.deepcopy(scene.state_dict()),
        "pixel_optimizer_param_groups": copy.deepcopy(pixel.state_dict()["param_groups"]),
        "scene_optimizer_param_groups": copy.deepcopy(scene.state_dict()["param_groups"]),
        "pixel_optimizer_lrs": [float(pixel.param_groups[0]["lr"])],
        "scene_optimizer_lrs": [float(scene.param_groups[0]["lr"])],
        "rng_state": capture_rng_state(batcher),
    })
    store = RecoveryStore(tmp_path, identity, torch.device("cpu"))
    store.save(payload)
    expected_warmup = advance(model, pixel, scene, batcher, 5, 6)
    expected_model = copy.deepcopy(model.state_dict())
    expected_pixel = copy.deepcopy(pixel.state_dict())
    expected_scene = copy.deepcopy(scene.state_dict())

    recovered_model, recovered_pixel, recovered_scene, recovered_batcher = construct(999)
    loaded = store.load()
    assert loaded is not None and loaded["next_epoch"] == 5
    recovered_model.load_state_dict(loaded["live_model_state"])
    recovered_pixel.load_state_dict(loaded["pixel_optimizer_state"])
    recovered_scene.load_state_dict(loaded["scene_optimizer_state"])
    restore_rng_state(loaded["rng_state"], recovered_batcher)  # deliberately last
    actual_warmup = advance(recovered_model, recovered_pixel, recovered_scene, recovered_batcher, 5, 6)
    assert actual_warmup == expected_warmup
    assert actual_warmup[:3] == [(5, pytest.approx(1e-3 / 3)), (5, pytest.approx(2e-3 / 3)), (5, pytest.approx(1e-3))]
    assert _nested_exact_equal(expected_model, recovered_model.state_dict())
    assert _nested_exact_equal(expected_pixel, recovered_pixel.state_dict())
    assert _nested_exact_equal(expected_scene, recovered_scene.state_dict())


def test_corrupt_latest_generation_falls_back_to_previous(tmp_path: Path) -> None:
    identity = {"runtime_signature": {"platform": "Windows"}}
    store = RecoveryStore(tmp_path, identity, torch.device("cpu"))
    store.save(_payload(identity, completed_epoch=1))
    store.save(_payload(identity, completed_epoch=2))
    pointer = json.loads((tmp_path / "latest.json").read_text())
    (tmp_path / pointer["generations"][0]["checkpoint"]).write_bytes(b"corrupt")
    loaded = store.load()
    assert loaded is not None
    assert loaded["completed_epoch"] == 1
    assert loaded["next_epoch"] == 2


def test_identity_runtime_and_forbidden_access_mismatches_are_rejected(tmp_path: Path) -> None:
    identity = {"runtime_signature": {"driver": "595.79"}, "protocol_identity": "p"}
    RecoveryStore(tmp_path, identity, torch.device("cpu")).save(_payload(identity))
    changed_runtime = {"runtime_signature": {"driver": "different"}, "protocol_identity": "p"}
    with pytest.raises(RuntimeError, match="identity/runtime/access mismatch"):
        RecoveryStore(tmp_path, changed_runtime, torch.device("cpu")).load()
    forbidden = _payload(identity)
    forbidden["access_ledger"] = {**forbidden["access_ledger"], "comparator_values_decoded": True}
    with pytest.raises(RuntimeError, match="forbidden evidence"):
        RecoveryStore(tmp_path / "forbidden", identity, torch.device("cpu")).save(forbidden)

    second_endpoint_identity = {
        "runtime_signature": {"driver": "595.79"},
        "held_fold": 4,
        "access_ledger": AccessLedger(
            comparator_integrity_bytes_hashed=True,
            inner_validation_outcomes_opened=True,
            held_folds_opened=(3,),
        ).snapshot(),
    }
    second_payload = _payload(second_endpoint_identity)
    second_payload["access_ledger"] = copy.deepcopy(second_endpoint_identity["access_ledger"])
    second_store = RecoveryStore(tmp_path / "second-endpoint", second_endpoint_identity, torch.device("cpu"))
    second_store.save(second_payload)
    assert second_store.load()["access_ledger"]["held_folds_opened"] == [3]


def test_partial_epoch_is_not_persisted_and_last_boundary_replays(tmp_path: Path) -> None:
    identity = {"protocol_identity": "p"}
    payload = _payload(identity, completed_epoch=3)
    store = RecoveryStore(tmp_path, identity, torch.device("cpu"))
    store.save(payload)
    payload["live_model_state"][next(iter(payload["live_model_state"]))].add_(1000)  # unsaved partial epoch
    loaded = store.load()
    assert loaded is not None
    assert loaded["completed_epoch"] == 3
    assert loaded["next_epoch"] == 4
    assert not _nested_exact_equal(loaded["live_model_state"], payload["live_model_state"])


def test_existing_immutable_held_part_is_reused_without_evaluation(tmp_path: Path) -> None:
    path = tmp_path / "held-3.part.json"
    binding = _binding(3)
    expected = seal_or_reuse_prediction_part(path, _part(3), binding)
    assert not (path.stat().st_mode & 0o222)
    calls = 0

    def forbidden_evaluation() -> dict[str, np.ndarray]:
        nonlocal calls
        calls += 1
        raise AssertionError("held fold was evaluated twice")

    reused, was_reused = evaluate_or_reuse_prediction_part(path, binding, forbidden_evaluation)
    assert was_reused
    assert calls == 0
    assert _nested_exact_equal(expected, reused)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        seal_or_reuse_prediction_part(path, None, {**binding, "scientific_digest": "changed"})


def test_comparator_preflight_hashes_opaque_bytes_without_decode(tmp_path: Path) -> None:
    paths = {}
    for name in ("champion_scene_cache", "gaussian_dense_state", "gaussian_protocol", "released_dense_checkpoint"):
        path = tmp_path / name
        path.write_bytes(b"not a numpy or torch container")
        paths[name] = path
    ledger = AccessLedger()
    hashes = preflight_comparator_hashes(paths, ledger)
    assert set(hashes) == set(paths)
    assert ledger.comparator_integrity_bytes_hashed
    assert not ledger.comparator_values_decoded


def test_semantic_comparator_decode_requires_two_parts_and_final_candidate_immutable(tmp_path: Path) -> None:
    parts = [tmp_path / "part-3.json", tmp_path / "part-4.json"]
    candidate = tmp_path / "candidate.npz"
    for path in (*parts, candidate):
        path.write_bytes(b"sealed")
    ledger = AccessLedger(comparator_integrity_bytes_hashed=True)
    ledger.open_held_fold(3)
    ledger.open_held_fold(4)
    with pytest.raises(RuntimeError, match="mutable or missing"):
        authorize_comparator_decode(candidate, parts, ledger)
    for path in (*parts, candidate):
        os.chmod(path, 0o444)
    authorize_comparator_decode(candidate, parts, ledger)
    assert ledger.comparator_values_decoded


def test_native_windows_runtime_is_rejected_before_data_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import train_mars_sensor_ordinal as trainer

    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("CUDA_MODULE_LOADING", "LAZY")
    monkeypatch.setattr(trainer, "runtime_signature", lambda: {
        **trainer.REQUIRED_NATIVE_WINDOWS_RUNTIME,
        "platform": "Linux",
        "environment": dict(trainer.REQUIRED_RUNTIME_ENV),
    })
    with pytest.raises(RuntimeError, match="Native-Windows production runtime mismatch"):
        trainer.verify_runtime_environment(require_native_windows=True)

    assert trainer.REQUIRED_NATIVE_WINDOWS_RUNTIME == {
        "platform": "Windows",
        "gpu": "NVIDIA GeForce RTX 5070",
        "nvidia_driver": "595.79",
        "torch": "2.11.0+cu128",
        "numpy": "2.4.4",
        "rasterio": "1.4.4",
        "scikit-learn": "1.9.0",
        "scipy": "1.17.1",
    }
    exact = {
        **trainer.REQUIRED_NATIVE_WINDOWS_RUNTIME,
        "environment": dict(trainer.REQUIRED_RUNTIME_ENV),
    }
    monkeypatch.setattr(trainer, "runtime_signature", lambda: exact)
    assert trainer.verify_runtime_environment(require_native_windows=True) == exact
