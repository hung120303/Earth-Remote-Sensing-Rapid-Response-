from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_mars_sensor_ordinal as trainer  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recovery_payload(identity: dict[str, object]) -> dict[str, object]:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    pixel = torch.optim.AdamW(model[0].parameters(), lr=0.0123)
    scene = torch.optim.AdamW(model[1].parameters(), lr=0.0456)
    pixel_state = copy.deepcopy(pixel.state_dict())
    scene_state = copy.deepcopy(scene.state_dict())
    return {
        "schema_version": trainer.RECOVERY_SCHEMA_VERSION,
        "identity": identity,
        "live_model_state": copy.deepcopy(model.state_dict()),
        "best_model_state": copy.deepcopy(model.state_dict()),
        "pixel_optimizer_state": pixel_state,
        "scene_optimizer_state": scene_state,
        "pixel_optimizer_param_groups": copy.deepcopy(pixel_state["param_groups"]),
        "scene_optimizer_param_groups": copy.deepcopy(scene_state["param_groups"]),
        "pixel_optimizer_lrs": [0.0123],
        "scene_optimizer_lrs": [0.0456],
        "best_rank": [0.1, 0.2, -0.3],
        "best_epoch": 1,
        "history": [{"epoch": 1}],
        "cutpoints": np.asarray([1.0, 2.0, 3.0]),
        "completed_epoch": 1,
        "next_epoch": 2,
        "held_fold": 3,
        "fit_fold": 4,
        "rng_state": {
            "python": None,
            "numpy_legacy": None,
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda_all": [],
            "site_balanced_batcher_bit_generator": {},
        },
        "access_ledger": trainer.AccessLedger().snapshot(),
    }


def test_protocol_bound_text_retains_hashes_with_windows_autocrlf(tmp_path: Path) -> None:
    """A simulated clean Windows checkout must preserve protocol-bound LF bytes."""
    tracked = (
        Path("configs/mars_sensor_ordinal_protocol.json"),
        Path("tools/train_mars_sensor_ordinal.py"),
        Path("EarthRemoteSensingRapidResponse/mars_sensor_ordinal.py"),
    )
    repository = tmp_path / "repository"
    checkout = tmp_path / "checkout"
    repository.mkdir()
    shutil.copy2(ROOT / ".gitattributes", repository / ".gitattributes")
    for relative in tracked:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=repository, check=True)
    subprocess.run(["git", "add", ".gitattributes", *map(str, tracked)], cwd=repository, check=True)
    checkout.mkdir()
    subprocess.run(
        ["git", "checkout-index", "--force", f"--prefix={checkout.as_posix()}/", *map(str, tracked)],
        cwd=repository,
        check=True,
    )

    for relative in tracked:
        assert _sha256(checkout / relative) == _sha256(ROOT / relative)
        assert b"\r\n" not in (checkout / relative).read_bytes()

    protocol_path = checkout / tracked[0]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert trainer.protocol_identity(protocol) == protocol["protocol_sha256_self_excluding_field"]
    assert _sha256(checkout / tracked[1]) == protocol["trainer"]["sha256"]
    assert _sha256(checkout / tracked[2]) == protocol["model"]["sha256"]


def test_binary_artifacts_are_exempt_from_text_normalization() -> None:
    completed = subprocess.run(
        [
            "git", "check-attr", "text", "eol", "--",
            "fixture.model", "fixture.npz", "fixture.pt", "fixture.tif",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    assert len(lines) == 8
    assert all(line.endswith(": unset") or line.endswith(": unspecified") for line in lines)


def test_all_atomic_writers_and_recovery_fsync_writable_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Emulate Windows EBADF when fsync receives a read-only descriptor."""
    original_open = Path.open
    original_fsync = trainer.os.fsync
    descriptor_modes: dict[int, str] = {}
    fsynced_modes: list[str] = []

    def tracked_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        stream = original_open(path, mode, *args, **kwargs)
        descriptor_modes[stream.fileno()] = mode
        return stream

    def windows_fsync(descriptor: int) -> None:
        mode = descriptor_modes.get(descriptor, "")
        if not any(marker in mode for marker in ("w", "a", "+")):
            raise OSError(9, "Bad file descriptor")
        fsynced_modes.append(mode)
        original_fsync(descriptor)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(trainer.os, "fsync", windows_fsync)
    monkeypatch.setattr(trainer, "_fsync_directory", lambda _path: None)

    npz_path = tmp_path / "arrays.npz"
    torch_path = tmp_path / "state.pt"
    text_path = tmp_path / "receipt.json"
    pointer_path = tmp_path / "pointer.json"
    trainer.atomic_npz(npz_path, values=np.asarray([1, 2, 3]))
    trainer.atomic_torch(torch_path, {"value": torch.tensor([4.0])})
    trainer.atomic_text(text_path, "exact\n")
    trainer.atomic_replace_json(pointer_path, {"generation": 1})

    identity = {"access_ledger": trainer.AccessLedger().snapshot()}
    store = trainer.RecoveryStore(tmp_path / "recovery", identity, torch.device("cpu"))
    store.save(_recovery_payload(identity))

    with np.load(npz_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["values"], [1, 2, 3])
    assert torch.equal(torch.load(torch_path, weights_only=True)["value"], torch.tensor([4.0]))
    assert text_path.read_text(encoding="utf-8") == "exact\n"
    assert json.loads(pointer_path.read_text(encoding="utf-8")) == {"generation": 1}
    assert store.load() is not None
    assert len(fsynced_modes) == 7
    assert all(any(marker in mode for marker in ("w", "a", "+")) for mode in fsynced_modes)
    for path in (npz_path, torch_path, text_path):
        assert not (path.stat().st_mode & 0o222)
