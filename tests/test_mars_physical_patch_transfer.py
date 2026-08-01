from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_physical_patch_transfer import (  # noqa: E402
    PhysicalPatchTransferDetector,
    gradient_reverse,
    mars_tile_to_canonical,
    physical_tile_starts,
)


def test_mars_physical_conversion_has_exact_contract() -> None:
    values = torch.zeros(2, 16, 64, 64)
    values[:, 1:13] = 0.2
    values[:, 5] = 0.4
    values[:, 6] = 0.2
    values[:, 11] = 0.3
    values[:, 12] = 0.2
    values[:, 13] = 0.5
    values[:, 14] = -0.25
    values[:, 15, 0, 0] = 1.0
    observable = torch.ones(2, 1, 64, 64)
    canonical, auxiliary, pooled = mars_tile_to_canonical(values, observable)
    assert canonical.shape == (2, 20, 32, 32)
    assert auxiliary.shape == (2, 3, 32, 32)
    assert pooled.shape == (2, 1, 32, 32)
    assert torch.equal(canonical[:, 0], canonical[:, 1])
    assert torch.equal(canonical[:, 8:14], canonical[:, 14:20])
    assert torch.all(auxiliary[:, 2, 0, 0] == 1.0)
    assert torch.isfinite(canonical).all()


def test_edge_aligned_physical_tiles_cover_every_scene_pixel() -> None:
    starts = physical_tile_starts(200)
    assert starts == (0, 32, 64, 96, 128, 136)
    coverage = torch.zeros(200, 200, dtype=torch.int64)
    for y in starts:
        for x in starts:
            coverage[y : y + 64, x : x + 64] += 1
    assert int(coverage.min()) >= 1


def test_model_outputs_and_frozen_scene_routing() -> None:
    torch.manual_seed(31)
    model = PhysicalPatchTransferDetector().eval()
    inputs = torch.rand(2, 20, 32, 32)
    auxiliary = torch.rand(2, 3, 32, 32)
    observable = torch.ones(2, 1, 32, 32)
    sensors = torch.tensor([0, 1])
    with torch.no_grad():
        output = model(inputs, auxiliary, observable, sensors)
    assert output["segmentation_logits"].shape == (2, 1, 32, 32)
    assert output["scene_logit"].shape == (2,)
    assert output["domain_logit"].shape == (2,)
    assert torch.isfinite(output["segmentation_logits"]).all()
    base = torch.tensor([0.2, 0.8])
    candidate = model.fuse_scene_score(base, output["scene_logit"], sensors, 0.25)
    assert candidate[1] == base[1]
    assert model.aggregate_tile_scene_logits(torch.randn(2, 36)).shape == (2,)


def test_gradient_reversal_changes_only_backward_sign() -> None:
    values = torch.tensor([1.0, -2.0], requires_grad=True)
    reversed_values = gradient_reverse(values, 0.25)
    assert torch.equal(values, reversed_values)
    reversed_values.sum().backward()
    assert torch.equal(values.grad, torch.full_like(values, -0.25))
