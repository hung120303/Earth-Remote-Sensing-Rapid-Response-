from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_protected_bisensor_finetune import (  # noqa: E402
    PROTECTION_GATE,
    ProtectedBisensorFinetune,
)


def test_zero_strength_is_exact_identity() -> None:
    base = torch.tensor([0.1, 0.4, 0.9])
    result = ProtectedBisensorFinetune.fuse_scene_score(
        base, torch.tensor([10.0, -10.0, 1.0]), torch.tensor([0, 1, 0]), 0.0
    )
    assert torch.equal(result, base)


def test_below_gate_is_identity_and_high_scores_stay_high() -> None:
    base = torch.tensor([0.01, 0.249, 0.25, 0.7, 0.99])
    delta = torch.tensor([100.0, -100.0, -100.0, -100.0, 100.0])
    result = ProtectedBisensorFinetune.fuse_scene_score(
        base, delta, torch.tensor([0, 1, 0, 1, 0]), 1.0, sentinel_only=True
    )
    assert torch.equal(result[:2], base[:2])
    assert bool((result[2:] >= PROTECTION_GATE).all())


def test_both_sensor_families_receive_evidence() -> None:
    base = torch.tensor([0.6, 0.6])
    result = ProtectedBisensorFinetune.fuse_scene_score(
        base, torch.tensor([1.0, 1.0]), torch.tensor([0, 1]), 0.5, sentinel_only=True
    )
    assert result[0] > base[0]
    assert result[1] > base[1]
    assert torch.equal(result[0], result[1])
