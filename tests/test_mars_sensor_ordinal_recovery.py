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
    _verify_recovery_output_phase,
    authorize_comparator_decode,
    capture_rng_state,
    evaluate_or_reuse_prediction_part,
    merge_predictions,
    preflight_comparator_hashes,
    restore_rng_state,
    scene_learning_rate,
    seal_or_reuse_prediction_part,
    seal_or_validate_final_candidate,
    seal_or_validate_text_artifact,
    seal_or_validate_torch_artifact,
    sha256,
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


def test_relocated_torch_rng_entries_are_restored_as_byte_exact_cpu_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RelocatedTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, value: torch.Tensor):
            return torch.Tensor._make_subclass(cls, value, False)

        @property
        def device(self) -> torch.device:
            return torch.device("cuda:0")

        def to(self, *args, **kwargs) -> torch.Tensor:
            return self.as_subclass(torch.Tensor).to(*args, **kwargs)

    batcher = SimpleNamespace(rng=np.random.default_rng(13))
    state = capture_rng_state(batcher)
    cpu_bytes = state["torch_cpu"].clone()
    cuda_bytes = torch.arange(0, 64, dtype=torch.uint8)
    state["torch_cpu"] = RelocatedTensor(cpu_bytes)
    state["torch_cuda_all"] = [RelocatedTensor(cuda_bytes)]
    restored: dict[str, object] = {}

    monkeypatch.setattr(torch, "set_rng_state", lambda value: restored.__setitem__("cpu", value))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda values: restored.__setitem__("cuda", values))

    restore_rng_state(state, batcher)

    restored_cpu = restored["cpu"]
    restored_cuda = restored["cuda"]
    assert isinstance(restored_cpu, torch.Tensor)
    assert restored_cpu.device.type == "cpu" and restored_cpu.dtype == torch.uint8
    assert torch.equal(restored_cpu, cpu_bytes)
    assert isinstance(restored_cuda, list) and len(restored_cuda) == 1
    assert restored_cuda[0].device.type == "cpu" and restored_cuda[0].dtype == torch.uint8
    assert torch.equal(restored_cuda[0], cuda_bytes)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("torch_cpu", torch.zeros(8, dtype=torch.int64), "torch_cpu.*uint8"),
        ("torch_cpu", torch.zeros((2, 4), dtype=torch.uint8), "torch_cpu.*one-dimensional"),
        ("torch_cuda_all", [torch.zeros(8, dtype=torch.int64)], r"torch_cuda_all\[0\].*uint8"),
    ],
)
def test_invalid_torch_rng_entries_are_rejected(field: str, value: object, message: str) -> None:
    batcher = SimpleNamespace(rng=np.random.default_rng(13))
    state = capture_rng_state(batcher)
    state[field] = value
    with pytest.raises(RuntimeError, match=message):
        restore_rng_state(state, batcher)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the recovery device audit")
def test_cpu_loaded_checkpoint_restores_exact_cuda_next_step_and_usable_optimizer(tmp_path: Path) -> None:
    device = torch.device("cuda")

    def construct(seed: int):
        torch.manual_seed(seed)
        model = torch.nn.Linear(2, 1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        batcher = SimpleNamespace(rng=np.random.default_rng(seed))
        return model, optimizer, batcher

    def advance(model, optimizer, batcher) -> float:
        scale = (
            random.random()
            + float(np.random.random())
            + float(torch.rand(()))
            + float(torch.rand((), device=device))
            + float(batcher.rng.random())
        )
        optimizer.zero_grad()
        loss = model(torch.tensor([[0.25, -0.5]], device=device)).sum() * scale
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    model, optimizer, batcher = construct(34)
    advance(model, optimizer, batcher)
    identity = {"runtime_signature": {"platform": "Windows", "device": "cuda"}}
    payload = _payload(identity, completed_epoch=1)
    payload.update({
        "live_model_state": copy.deepcopy(model.state_dict()),
        "best_model_state": copy.deepcopy(model.state_dict()),
        "pixel_optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "scene_optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "pixel_optimizer_param_groups": copy.deepcopy(optimizer.state_dict()["param_groups"]),
        "scene_optimizer_param_groups": copy.deepcopy(optimizer.state_dict()["param_groups"]),
        "pixel_optimizer_lrs": [float(optimizer.param_groups[0]["lr"])],
        "scene_optimizer_lrs": [float(optimizer.param_groups[0]["lr"])],
        "rng_state": capture_rng_state(batcher),
    })
    store = RecoveryStore(tmp_path, identity, device)
    store.save(payload)

    expected_loss = advance(model, optimizer, batcher)
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    recovered_model, recovered_optimizer, recovered_batcher = construct(999)
    loaded = store.load()
    assert loaded is not None
    assert all(value.device.type == "cpu" for value in loaded["live_model_state"].values())
    assert all(value.device.type == "cpu" for value in loaded["best_model_state"].values())
    assert loaded["rng_state"]["torch_cpu"].device.type == "cpu"
    assert all(value.device.type == "cpu" for value in loaded["rng_state"]["torch_cuda_all"])

    recovered_model.load_state_dict(loaded["live_model_state"])
    recovered_optimizer.load_state_dict(loaded["pixel_optimizer_state"])
    restore_rng_state(loaded["rng_state"], recovered_batcher)
    actual_loss = advance(recovered_model, recovered_optimizer, recovered_batcher)

    assert actual_loss == expected_loss
    assert _nested_exact_equal(recovered_model.state_dict(), expected_model)
    assert _nested_exact_equal(recovered_optimizer.state_dict(), expected_optimizer)
    for optimizer_state in recovered_optimizer.state.values():
        assert optimizer_state["exp_avg"].device.type == "cuda"
        assert optimizer_state["exp_avg_sq"].device.type == "cuda"


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
    expected, was_reused = evaluate_or_reuse_prediction_part(path, binding, lambda: _part(3))
    assert not was_reused
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


def test_access_start_is_durable_before_evaluator_and_uncertain_start_never_repeats(
    tmp_path: Path,
) -> None:
    import train_mars_sensor_ordinal as trainer

    path = tmp_path / "held-3.part.json"
    binding = _binding(3)
    start_path, completion_path = trainer._receipt_paths(path)
    calls = 0

    def interrupted() -> dict[str, np.ndarray]:
        nonlocal calls
        calls += 1
        assert start_path.is_file()
        assert not (start_path.stat().st_mode & 0o222)
        assert not path.exists()
        assert not completion_path.exists()
        raise OSError("crash after durable start")

    with pytest.raises(OSError, match="durable start"):
        evaluate_or_reuse_prediction_part(path, binding, interrupted)
    with pytest.raises(RuntimeError, match="refusing repeat evaluation"):
        evaluate_or_reuse_prediction_part(path, binding, interrupted)
    assert calls == 1


def test_part_sealed_before_completion_recovers_without_repeating_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import train_mars_sensor_ordinal as trainer

    path = tmp_path / "held-3.part.json"
    binding = _binding(3)
    original = trainer._seal_or_validate_immutable_json
    calls = 0

    def crash_before_completion(receipt_path: Path, expected: dict[str, object]) -> None:
        if expected.get("kind") == "held_prediction_part_complete":
            raise OSError("crash before completion receipt")
        original(receipt_path, expected)

    def evaluate() -> dict[str, np.ndarray]:
        nonlocal calls
        calls += 1
        return _part(3)

    monkeypatch.setattr(trainer, "_seal_or_validate_immutable_json", crash_before_completion)
    with pytest.raises(OSError, match="completion receipt"):
        evaluate_or_reuse_prediction_part(path, binding, evaluate)
    assert path.is_file() and not (path.stat().st_mode & 0o222)
    monkeypatch.setattr(trainer, "_seal_or_validate_immutable_json", original)
    recovered, reused = evaluate_or_reuse_prediction_part(path, binding, evaluate)
    assert reused and calls == 1
    assert _nested_exact_equal(recovered, _part(3))
    _, completion_path = trainer._receipt_paths(path)
    assert completion_path.is_file() and not (completion_path.stat().st_mode & 0o222)


def test_part_or_completion_without_start_and_receipt_hash_mismatch_are_rejected(
    tmp_path: Path,
) -> None:
    import train_mars_sensor_ordinal as trainer

    orphan = tmp_path / "orphan.part.json"
    seal_or_reuse_prediction_part(orphan, _part(3), _binding(3))
    with pytest.raises(RuntimeError, match="without a durable access-start"):
        evaluate_or_reuse_prediction_part(orphan, _binding(3), lambda: _part(3))

    path = tmp_path / "held-3.part.json"
    evaluate_or_reuse_prediction_part(path, _binding(3), lambda: _part(3))
    _, completion = trainer._receipt_paths(path)
    os.chmod(completion, 0o644)
    payload = json.loads(completion.read_text())
    payload["part_sha256"] = "0" * 64
    completion.write_text(json.dumps(payload))
    os.chmod(completion, 0o444)
    with pytest.raises(RuntimeError, match="binding mismatch"):
        evaluate_or_reuse_prediction_part(path, _binding(3), lambda: _part(3))


def test_final_candidate_and_outputs_are_idempotent_but_never_overwritten(tmp_path: Path) -> None:
    part_paths = [tmp_path / f"held-{fold}.part.json" for fold in (3, 4)]
    bindings = [_binding(fold) for fold in (3, 4)]
    parts = []
    for path, binding, fold in zip(part_paths, bindings, (3, 4)):
        part, reused = evaluate_or_reuse_prediction_part(path, binding, lambda fold=fold: _part(fold))
        assert not reused
        parts.append(part)
    candidate = merge_predictions(parts)
    candidate_path = tmp_path / "candidate.npz"
    assert not seal_or_validate_final_candidate(candidate_path, candidate, part_paths, bindings)
    candidate_hash = sha256(candidate_path)
    assert seal_or_validate_final_candidate(candidate_path, candidate, part_paths, bindings)
    assert sha256(candidate_path) == candidate_hash

    changed = {key: value.copy() for key, value in candidate.items()}
    changed["scores"][0] += 0.1
    with pytest.raises(RuntimeError, match="array mismatch"):
        seal_or_validate_final_candidate(candidate_path, changed, part_paths, bindings)
    changed_dtype = {key: value.copy() for key, value in candidate.items()}
    changed_dtype["labels"] = changed_dtype["labels"].astype(np.int64)
    with pytest.raises(RuntimeError, match="array mismatch"):
        seal_or_validate_final_candidate(candidate_path, changed_dtype, part_paths, bindings)
    with pytest.raises(RuntimeError, match="binding mismatch"):
        seal_or_validate_final_candidate(
            candidate_path, candidate, part_paths, [{**bindings[0], "scientific_digest": "wrong"}, bindings[1]]
        )

    state_path = tmp_path / "states.pt"
    state = {"folds": {"3": torch.tensor([1.0]), "4": torch.tensor([2.0])}}
    assert not seal_or_validate_torch_artifact(state_path, state)
    assert seal_or_validate_torch_artifact(state_path, state)
    with pytest.raises(RuntimeError, match="contents mismatch"):
        seal_or_validate_torch_artifact(state_path, {"folds": {"3": torch.tensor([9.0])}})

    json_path, markdown_path = tmp_path / "report.json", tmp_path / "report.md"
    report = {"passed": True, "metric": 0.5}
    text = json.dumps(report, indent=2) + "\n"
    assert not seal_or_validate_text_artifact(json_path, text)
    assert seal_or_validate_text_artifact(json_path, text)
    assert not seal_or_validate_text_artifact(markdown_path, "# exact\n")
    assert seal_or_validate_text_artifact(markdown_path, "# exact\n")
    with pytest.raises(RuntimeError, match="contents mismatch"):
        seal_or_validate_text_artifact(markdown_path, "# changed\n")


def test_protocol_recovery_phase_accepts_only_consistent_immutable_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import train_mars_sensor_ordinal as trainer

    protocol_path = ROOT / "configs/mars_sensor_ordinal_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["outputs"] = {
        "candidate_predictions": str(tmp_path / "out/candidate.npz"),
        "endpoint_states": str(tmp_path / "out/states.pt"),
        "json": str(tmp_path / "out/report.json"),
        "markdown": str(tmp_path / "out/report.md"),
    }
    protocol["trainer"]["sha256"] = trainer.sha256(ROOT / protocol["trainer"]["path"])
    protocol["protocol_sha256_self_excluding_field"] = trainer.protocol_identity(protocol)
    candidate_path = tmp_path / "out/candidate.npz"
    part_paths = [tmp_path / f"out/candidate.held-{fold}.part.json" for fold in (3, 4)]
    bindings = [
        {
            **_binding(fold),
            "protocol_identity": trainer.protocol_identity(protocol),
            "scientific_digest": trainer.scientific_digest(protocol),
        }
        for fold in (3, 4)
    ]
    parts = [
        evaluate_or_reuse_prediction_part(path, binding, lambda fold=fold: _part(fold))[0]
        for path, binding, fold in zip(part_paths, bindings, (3, 4))
    ]
    candidate = merge_predictions(parts)
    seal_or_validate_final_candidate(candidate_path, candidate, part_paths, bindings)
    _verify_recovery_output_phase(protocol)
    assert trainer.verify_protocol(protocol, protocol_path, smoke=False)

    start_path, _ = trainer._receipt_paths(part_paths[0])
    os.chmod(start_path, 0o644)
    wrong = json.loads(start_path.read_text())
    wrong["binding"]["scientific_digest"] = "wrong-science"
    start_path.write_text(json.dumps(wrong, indent=2, sort_keys=True) + "\n")
    os.chmod(start_path, 0o444)
    with pytest.raises(RuntimeError, match="binding mismatch|fold binding mismatch"):
        trainer.verify_protocol(protocol, protocol_path, smoke=False)

    os.chmod(start_path, 0o644)
    start_path.unlink()
    trainer._seal_or_validate_immutable_json(
        start_path, trainer._held_access_start_receipt(bindings[0])
    )
    markdown_path = tmp_path / "out/report.md"
    seal_or_validate_text_artifact(markdown_path, "unrelated\n")
    with pytest.raises(RuntimeError, match="unrelated or partial"):
        _verify_recovery_output_phase(protocol)
    os.chmod(markdown_path, 0o644)
    markdown_path.unlink()

    os.chmod(candidate_path, 0o644)
    with pytest.raises(RuntimeError, match="mutable"):
        _verify_recovery_output_phase(protocol)
    os.chmod(candidate_path, 0o444)

    os.chmod(candidate_path, 0o644)
    with np.load(candidate_path, allow_pickle=False) as source:
        wrong_candidate = {key: source[key].copy() for key in source.files}
    wrong_candidate["scores"][0] += 0.5
    np.savez_compressed(candidate_path, **wrong_candidate)
    os.chmod(candidate_path, 0o444)
    with pytest.raises(RuntimeError, match="array mismatch"):
        _verify_recovery_output_phase(protocol)


@pytest.mark.parametrize(
    "boundary",
    ["candidate", "metrics", "endpoint_state", "json", "markdown"],
)
def test_finalization_crash_boundaries_resume_without_training_or_held_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    import train_mars_sensor_ordinal as trainer

    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = {
        "outer_folds": [3, 4], "seed": trainer.SEED,
        "architecture": {}, "ordinal_targets": {}, "inner_split": {}, "training": {},
        "evaluation": {}, "bootstrap": {"replicates": 1, "ap_seed": 2, "dense_seed": 3},
        "gates": {},
        "outputs": {
            "candidate_predictions": "out/candidate.npz",
            "endpoint_states": "out/states.pt",
            "json": "out/report.json",
            "markdown": "out/report.md",
        },
    }
    protocol_path.write_text(json.dumps(protocol))
    records = [
        {"sample_id": f"sample-{fold}", "group_id": f"group-{fold}", "fold": fold}
        for fold in (3, 4)
    ]
    groups = {f"group-{fold}": fold for fold in (3, 4)}
    monkeypatch.setattr(trainer, "fold_lookup", lambda _path: groups)
    monkeypatch.setattr(trainer, "records_for_folds", lambda *_args, **_kwargs: records)

    endpoint_entries = 0
    training_epoch_calls = 0
    held_calls = 0

    def recovered_endpoint(_protocol, _paths, _fit, held_fold, _device, **_kwargs):
        nonlocal endpoint_entries, training_epoch_calls
        # This fixture represents already-sealed endpoints: no epoch body executes.
        endpoint_entries += 1
        training_epoch_calls += 0
        return (
            {
                "held_fold": held_fold, "fit_fold": 7 - held_fold,
                "selected_epoch": held_fold, "cutpoints": [1.0, 2.0, 3.0],
                "recovery_identity_sha256": f"endpoint-{held_fold}", "history": [],
            },
            {"weight": torch.tensor([float(held_fold)])},
            np.asarray([1.0, 2.0, 3.0]),
        )

    class DummyModel:
        def to(self, _device): return self
        def load_state_dict(self, _state, strict=True): return None

    monkeypatch.setattr(trainer, "train_endpoint", recovered_endpoint)
    monkeypatch.setattr(trainer, "MarsSensorOrdinalUNet", DummyModel)
    def forbidden_held_evaluation(*_args, **_kwargs):
        nonlocal held_calls
        held_calls += 1
        raise AssertionError("held evaluation repeated")

    monkeypatch.setattr(trainer, "evaluate_candidate", forbidden_held_evaluation)
    monkeypatch.setattr(trainer, "align_comparator", lambda candidate, _path: {
        "champion_scores": np.asarray([0.1, 0.2]),
        "spatial_prithvi_scores": np.asarray([0.1, 0.2]),
    })
    monkeypatch.setattr(trainer, "reconstruct_dense_comparator", lambda candidate, *_a, **_k: np.zeros_like(candidate["dense_counts"]))
    metrics = {
        "passed": True, "pooled_ap_delta": 0.1, "ap_bootstrap_lower": 0.1,
        "matched_fpr_recall_delta": 0.1, "dense_iou_delta": 0.1,
        "dense_bootstrap_lower": 0.1, "checks": {"frozen": True},
    }
    metric_calls = 0

    def fixed_metrics(*_args, **_kwargs):
        nonlocal metric_calls
        metric_calls += 1
        return copy.deepcopy(metrics)

    monkeypatch.setattr(trainer, "metric_gates", fixed_metrics)
    paths = {"fold_protocol": tmp_path / "folds.json", "manifest": tmp_path / "manifest.jsonl",
             "metadata_root": tmp_path}
    for name in ("champion_scene_cache", "gaussian_dense_state", "gaussian_protocol", "released_dense_checkpoint"):
        paths[name] = tmp_path / name
        paths[name].write_bytes(name.encode())
    candidate_path = tmp_path / "out/candidate.npz"
    ledger = AccessLedger(comparator_integrity_bytes_hashed=True)
    for fold in (3, 4):
        binding = {
            "protocol_identity": trainer.protocol_identity(protocol),
            "scientific_digest": trainer.scientific_digest(protocol),
            "held_fold": fold, "fit_fold": 7 - fold,
            "endpoint_recovery_identity_sha256": f"endpoint-{fold}",
            "ordered_held_records_sha256": trainer.ordered_records_hash([records[fold - 3]]),
            "ordered_sample_ids_sha256": trainer.canonical_json_hash([f"sample-{fold}"]),
            "access_before_open": {
                "comparator_integrity_bytes_hashed": True, "comparator_values_decoded": False,
                "previous_immutable_held_folds": list(ledger.held_folds_opened),
                "folds_0_1_2_opened": False, "external_or_official_evidence_opened": False,
            },
        }
        part_path = candidate_path.with_name(f"{candidate_path.stem}.held-{fold}.part.json")
        evaluate_or_reuse_prediction_part(part_path, binding, lambda fold=fold: _part(fold))
        ledger.open_held_fold(fold)

    original_candidate = trainer.seal_or_validate_final_candidate
    original_state = trainer.seal_or_validate_torch_artifact
    original_text = trainer.seal_or_validate_text_artifact
    crashed = False

    def candidate_boundary(*args, **kwargs):
        nonlocal crashed
        result = original_candidate(*args, **kwargs)
        if boundary == "candidate" and not crashed:
            crashed = True
            raise OSError("crash after candidate seal")
        return result

    def state_boundary(*args, **kwargs):
        nonlocal crashed
        if boundary == "metrics" and not crashed:
            crashed = True
            raise OSError("crash after comparator metrics")
        result = original_state(*args, **kwargs)
        if boundary == "endpoint_state" and not crashed:
            crashed = True
            raise OSError("crash after endpoint state")
        return result

    def text_boundary(path, text):
        nonlocal crashed
        result = original_text(path, text)
        target = "json" if path.suffix == ".json" else "markdown"
        if boundary == target and not crashed:
            crashed = True
            raise OSError(f"crash after {target}")
        return result

    monkeypatch.setattr(trainer, "seal_or_validate_final_candidate", candidate_boundary)
    monkeypatch.setattr(trainer, "seal_or_validate_torch_artifact", state_boundary)
    monkeypatch.setattr(trainer, "seal_or_validate_text_artifact", text_boundary)
    with pytest.raises(OSError, match="crash after"):
        trainer.run_full(protocol, protocol_path, paths, runtime={})
    sealed_before = {
        path.name: path.read_bytes()
        for path in (candidate_path, tmp_path / "out/states.pt", tmp_path / "out/report.json", tmp_path / "out/report.md")
        if path.exists()
    }
    report = trainer.run_full(protocol, protocol_path, paths, runtime={})
    assert report["metrics"] == metrics
    assert endpoint_entries == 4
    assert training_epoch_calls == held_calls == 0
    for path in (candidate_path, tmp_path / "out/states.pt", tmp_path / "out/report.json", tmp_path / "out/report.md"):
        assert path.is_file() and not (path.stat().st_mode & 0o222)
        if path.name in sealed_before:
            assert path.read_bytes() == sealed_before[path.name]


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
