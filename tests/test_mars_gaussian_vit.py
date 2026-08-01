from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_gaussian_vit import GaussianPretrainedViTUNet  # noqa: E402


def test_vit_unet_preserves_training_and_full_scene_grids() -> None:
    torch.manual_seed(7)
    model = GaussianPretrainedViTUNet(dimension=64, depth=2, heads=4).eval()
    for size in (160, 200):
        inputs = torch.rand(2, 16, size, size)
        observable = torch.ones(2, 1, size, size)
        sensors = torch.tensor([0, 1])
        with torch.no_grad():
            output = model(inputs, observable, sensors)
        assert output["segmentation_logits"].shape == (2, 1, size, size)
        assert output["scene_logit"].shape == (2,)
        assert torch.isfinite(output["segmentation_logits"]).all()
        assert torch.isfinite(output["scene_logit"]).all()


def test_scene_fusion_is_bounded_and_identity_at_zero_strength() -> None:
    baseline = torch.tensor([0.1, 0.5, 0.9])
    evidence = torch.tensor([-4.0, 0.0, 4.0])
    identity = GaussianPretrainedViTUNet.fuse_scene_score(baseline, evidence, 0.0)
    corrected = GaussianPretrainedViTUNet.fuse_scene_score(baseline, evidence, 0.5)
    assert torch.allclose(identity, baseline)
    assert torch.all((corrected > 0) & (corrected < 1))
    assert corrected[0] < baseline[0]
    assert corrected[2] > baseline[2]
