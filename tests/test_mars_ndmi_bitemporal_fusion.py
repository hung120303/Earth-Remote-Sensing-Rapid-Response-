from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_ndmi_bitemporal_fusion import (  # noqa: E402
    GUIDE_CHANNELS,
    PATCH_GRID_SIZE,
    MethaneChangeGuide,
    NdmiBitemporalFusionAdapter,
)


def test_methane_guide_is_finite_and_has_frozen_channel_count() -> None:
    values = torch.zeros(2, 16, 32, 32)
    values[:, 1:13] = 0.2
    values[:, 5] = 0.4
    values[:, 6] = 0.2
    values[:, 11] = 0.3
    values[:, 12] = 0.2
    observable = torch.ones(2, 1, 32, 32)
    sensor = torch.tensor([0, 1])
    guide = MethaneChangeGuide.inputs(values, observable, sensor)
    assert guide.shape == (2, GUIDE_CHANNELS, 32, 32)
    assert torch.isfinite(guide).all()
    expected_target_ndmi = (0.4 - 0.2) / (0.4 + 0.2 + 0.02)
    assert torch.allclose(guide[:, 1], torch.full_like(guide[:, 1], expected_target_ndmi))


def test_zero_initialized_dense_path_is_exact_teacher_identity() -> None:
    torch.manual_seed(7)
    model = NdmiBitemporalFusionAdapter().eval()
    values = torch.rand(2, 16, 32, 32)
    observable = torch.ones(2, 1, 32, 32)
    sensor = torch.tensor([0, 1])
    with torch.no_grad():
        output = model(values, observable, sensor)
    assert output["segmentation_logits"].shape == (2, 1, 32, 32)
    assert output["patch_logits"].shape == (2, 1, PATCH_GRID_SIZE, PATCH_GRID_SIZE)
    assert output["scene_logit"].shape == (2,)
    assert torch.equal(output["segmentation_logits"], output["baseline_logits"])
    assert torch.count_nonzero(output["correction_logits"]) == 0


def test_scene_fusion_has_exact_zero_strength_and_landsat_identity() -> None:
    base = torch.tensor([0.2, 0.4, 0.8, 0.6])
    evidence = torch.tensor([3.0, -2.0, 1.0, -4.0])
    sensors = torch.tensor([0, 1, 0, 1])
    zero = NdmiBitemporalFusionAdapter.fuse_scene_score(
        base, evidence, sensors, 0.0
    )
    candidate = NdmiBitemporalFusionAdapter.fuse_scene_score(
        base, evidence, sensors, 0.5, sentinel_only=True
    )
    assert torch.equal(zero, base)
    assert torch.equal(candidate[sensors == 1], base[sensors == 1])
    assert torch.all((candidate >= 0.0) & (candidate <= 1.0))
