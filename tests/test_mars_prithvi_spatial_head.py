from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_mars_prithvi_spatial_head import (  # noqa: E402
    GRID_SIZE,
    SpatialPrithviHead,
    patch_supervision_loss,
)


def test_spatial_prithvi_head_shapes() -> None:
    model = SpatialPrithviHead(dropout=0.0)
    result = model(
        torch.randn(3, 768, GRID_SIZE, GRID_SIZE),
        torch.ones(3, 9, GRID_SIZE, GRID_SIZE),
        torch.tensor([0, 1, 0]),
    )
    assert result["scene_logits"].shape == (3,)
    assert result["patch_logits"].shape == (3, 1, GRID_SIZE, GRID_SIZE)
    assert torch.isfinite(result["scene_logits"]).all()


def test_patch_supervision_excludes_missing_positive_truth() -> None:
    logits = torch.zeros(3, 1, GRID_SIZE, GRID_SIZE)
    targets = torch.zeros(3, 2, GRID_SIZE, GRID_SIZE)
    targets[:, 1] = 1
    targets[0, 0, 0, 0] = 1
    targets[1, 0, 0, 0] = 1
    labels = torch.tensor([1.0, 1.0, 0.0])
    available = torch.tensor([True, False, True])
    loss = patch_supervision_loss(logits, targets, labels, available)
    assert loss.shape == (3,)
    assert loss[0] > 0
    assert loss[1] == 0
    assert loss[2] > 0
