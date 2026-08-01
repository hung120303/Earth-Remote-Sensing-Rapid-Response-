from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_ndmi_patch_mil_fusion import NdmiPatchMilFusionAdapter  # noqa: E402


def test_patch_mil_scene_rule_is_local_and_exact() -> None:
    model = NdmiPatchMilFusionAdapter()
    patches = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    observable = torch.ones(1, 1, 32, 32)
    scene = model._scene_logit(
        torch.randn(1, 192, 2, 2), patches, observable, torch.tensor([0])
    )
    top = torch.arange(56, 64, dtype=torch.float32)
    expected = 0.75 * top.mean() + 0.25 * top.max()
    assert torch.equal(scene, expected.reshape(1))
    assert "scene_head" not in dict(model.named_children())


def test_patch_mil_dense_initialization_remains_exact_identity() -> None:
    torch.manual_seed(11)
    model = NdmiPatchMilFusionAdapter().eval()
    values = torch.rand(2, 16, 32, 32)
    observable = torch.ones(2, 1, 32, 32)
    with torch.no_grad():
        output = model(values, observable, torch.tensor([0, 1]))
    assert torch.equal(output["segmentation_logits"], output["baseline_logits"])
    assert torch.isfinite(output["scene_logit"]).all()
