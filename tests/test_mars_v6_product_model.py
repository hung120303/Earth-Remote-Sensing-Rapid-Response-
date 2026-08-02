from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_v6_product_model import (  # noqa: E402
    DENSE_PHYSICS_CHANNELS,
    ProductAffineHarmonizer,
    ProductHarmonizedMultiCohortV6,
    canonicalize_mars,
    canonicalize_methanes2cm,
    dense_physics_features,
)


class DummyPairEncoder(nn.Module):
    width = 12

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(6, self.width, 1)

    def forward(
        self, values: torch.Tensor, temporal: torch.Tensor, location: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        del temporal, location
        target = values[:, :, 1]
        grid = torch.nn.functional.adaptive_avg_pool2d(self.projection(target), (4, 4))
        return {"cls": grid.mean(dim=(2, 3)), "target_grid": grid}


def test_mars_and_methanes2cm_share_physical_contract() -> None:
    mars = torch.zeros(2, 16, 32, 32)
    mars[:, 1:13] = 1.0
    observable = torch.ones(2, 1, 32, 32)
    mars_batch = canonicalize_mars(
        mars,
        observable,
        torch.tensor([0, 1]),
        reference90_available=torch.tensor([1.0, 0.0]),
    )
    assert mars_batch.frames.shape == (2, 3, 6, 32, 32)
    assert torch.allclose(mars_batch.frames[:, 2], torch.full_like(mars_batch.frames[:, 2], 0.5))
    assert mars_batch.product_index.tolist() == [0, 1]
    assert mars_batch.reference_available.tolist() == [[1.0, 0.0], [0.0, 0.0]]

    methane = torch.zeros(2, 20, 32, 32)
    methane[:, 2:20] = 0.5
    methane_batch = canonicalize_methanes2cm(methane, observable)
    assert torch.allclose(methane_batch.frames, mars_batch.frames)
    assert methane_batch.product_index.tolist() == [2, 2]
    assert methane_batch.reference_available.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_harmonizer_is_identity_at_initialization_and_missing_is_explicit() -> None:
    inputs = torch.zeros(1, 20, 32, 32)
    inputs[:, 2:20] = 0.4
    batch = canonicalize_methanes2cm(
        inputs,
        torch.ones(1, 1, 32, 32),
        reference365_available=torch.zeros(1),
    )
    harmonized = ProductAffineHarmonizer()(batch)
    assert torch.allclose(harmonized, batch.frames)
    assert batch.reference_available.tolist() == [[1.0, 0.0]]


def test_dense_contract_has_declared_channel_count() -> None:
    inputs = torch.zeros(1, 20, 32, 32)
    inputs[:, 2:20] = torch.linspace(0.1, 0.6, 18)[None, :, None, None]
    batch = canonicalize_methanes2cm(inputs, torch.ones(1, 1, 32, 32))
    features, evidence = dense_physics_features(
        batch.frames, batch.observable, batch.reference_available
    )
    assert features.shape == (1, DENSE_PHYSICS_CHANNELS, 32, 32)
    assert evidence.shape == (1, 2, 32, 32)
    assert torch.isfinite(features).all()


def test_dual_branch_model_shapes_and_phase_isolation() -> None:
    inputs = torch.zeros(2, 20, 32, 32)
    inputs[:, 2:20] = 0.25
    batch = canonicalize_methanes2cm(inputs, torch.ones(2, 1, 32, 32))
    model = ProductHarmonizedMultiCohortV6(
        DummyPairEncoder(),
        DummyPairEncoder(),
        torch.zeros(6),
        torch.ones(6),
        scene_hidden=32,
    )
    temporal = torch.tensor(
        [[[2025.0, 1.0], [2024.0, 1.0], [2025.0, 91.0]]] * 2
    )
    location = torch.zeros(2, 2)
    output = model(batch, temporal, location)
    assert output["scene_logit"].shape == (2,)
    assert output["dense_logits"].shape == (2, 1, 32, 32)
    assert torch.isfinite(output["scene_logit"]).all()
    assert torch.isfinite(output["dense_logits"]).all()

    model.set_trainable_phase("scene")
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable
    assert all(name.startswith("scene_") for name in trainable)
    model.set_trainable_phase("dense")
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable
    assert not any(name.startswith("scene_") for name in trainable)


def test_phase_toggle_does_not_unfreeze_frozen_encoder_weights() -> None:
    scene = DummyPairEncoder()
    dense = DummyPairEncoder()
    scene.projection.weight.requires_grad_(False)
    dense.projection.weight.requires_grad_(False)
    model = ProductHarmonizedMultiCohortV6(
        scene, dense, torch.zeros(6), torch.ones(6), scene_hidden=16
    )
    model.set_trainable_phase("all")
    assert model.scene_encoder.projection.weight.requires_grad is False
    assert model.dense_encoder.projection.weight.requires_grad is False


def test_protected_scene_score_is_exact_below_gate() -> None:
    baseline = torch.tensor([0.1, 0.25, 0.8])
    residual = torch.tensor([10.0, -2.0, 1.0])
    candidate = ProductHarmonizedMultiCohortV6.protected_scene_score(
        baseline, residual, strength=0.1, protection_gate=0.25
    )
    assert candidate[0].item() == pytest.approx(baseline[0].item(), abs=0.0)
    assert candidate[1].item() >= 0.25
    assert candidate[1].item() < 0.25001
    assert candidate[2].item() > baseline[2].item()
