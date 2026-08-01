from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_gaussian_contrast_vit import (  # noqa: E402
    CONTRAST_CHANNELS,
    GaussianContrastViTUNet,
    methane_contrast_features,
)


def test_contrast_frontend_amplifies_temporal_b12_absorption() -> None:
    inputs = torch.full((1, 16, 32, 32), 0.3)
    inputs[:, 13:15] = 0.0
    inputs[:, 15] = 0.0
    observable = torch.ones(1, 1, 32, 32)
    baseline = methane_contrast_features(inputs, observable)
    plume = inputs.clone()
    plume[:, 6, 8:24, 8:24] -= 0.003
    plume[:, 0, 8:24, 8:24] = -0.005
    changed = methane_contrast_features(plume, observable)
    assert baseline.shape == (1, len(CONTRAST_CHANNELS), 32, 32)
    b11_b12 = CONTRAST_CHANNELS.index("log_ratio_change_B11_B12")
    assert torch.all(changed[:, b11_b12, 8:24, 8:24] > 0)
    assert torch.all(changed[:, 0, 8:24, 8:24] < 0)
    assert torch.equal(changed[..., :8, :], baseline[..., :8, :])


def test_contrast_vit_preserves_training_and_native_shapes() -> None:
    model = GaussianContrastViTUNet(dimension=64, depth=2, heads=4, reference_grid=13)
    for size in (160, 200):
        inputs = torch.rand(2, 16, size, size)
        inputs[:, 13:15] = 0.0
        inputs[:, 15] = 0.0
        observable = torch.ones(2, 1, size, size)
        output = model(inputs, observable, torch.tensor([0, 1]))
        assert output["segmentation_logits"].shape == (2, 1, size, size)
        assert output["scene_logit"].shape == (2,)
        assert torch.isfinite(output["segmentation_logits"]).all()
        assert torch.isfinite(output["scene_logit"]).all()
