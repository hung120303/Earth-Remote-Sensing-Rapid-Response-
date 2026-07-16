from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from extract_mars_prithvi_spatial_features import (  # noqa: E402
    BLOCKS,
    EMBED_DIM,
    GRID_SIZE,
    feature_names,
    pooled_patch_targets,
    spatial_patch_differences,
)


def test_spatial_patch_differences_preserve_depth_and_temporal_sign() -> None:
    outputs: list[torch.Tensor] = []
    tokens = 2 * GRID_SIZE * GRID_SIZE
    for depth in range(1, 13):
        values = torch.zeros(2, tokens + 1, EMBED_DIM)
        frames = values[:, 1:].reshape(2, 2, GRID_SIZE, GRID_SIZE, EMBED_DIM)
        frames[:, 0] = float(depth)
        frames[:, 1] = float(depth * 3)
        outputs.append(values)
    result = spatial_patch_differences(outputs)
    assert result.shape == (2, len(BLOCKS) * EMBED_DIM, GRID_SIZE, GRID_SIZE)
    for offset, depth in enumerate(BLOCKS):
        block = result[:, offset * EMBED_DIM : (offset + 1) * EMBED_DIM]
        assert torch.all(block == float(depth * 2))
    assert len(feature_names()) == len(BLOCKS) * EMBED_DIM


def test_pooled_patch_targets_keep_fraction_and_observability_separate() -> None:
    mask = torch.zeros(1, 1, 16, 16)
    observable = torch.ones_like(mask)
    mask[:, :, :2, :2] = 1
    observable[:, :, :8, 8:] = 0
    result = pooled_patch_targets(mask, observable)
    assert result.shape == (1, 2, GRID_SIZE, GRID_SIZE)
    assert torch.isclose(result[0, 0].sum(), torch.tensor(1.0))
    assert torch.isclose(result[0, 1].mean(), torch.tensor(0.75))
