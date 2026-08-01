from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
TOOLS_ROOT = ROOT / "tools"
for path in (MODEL_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_mars_gaussian_contrast_crossfit import (  # noqa: E402
    TransferGaussianContrastViTUNet,
)


def test_transfer_fusion_is_identity_at_zero_and_bounded() -> None:
    baseline = torch.tensor([0.01, 0.5, 0.99])
    residual = torch.tensor([-100.0, 0.0, 100.0])
    identity = TransferGaussianContrastViTUNet.fuse_scene_score(
        baseline, residual, 0.0
    )
    fused = TransferGaussianContrastViTUNet.fuse_scene_score(
        baseline, residual, 1.0
    )
    assert torch.allclose(identity, baseline, atol=1e-6)
    expected = torch.sigmoid(torch.logit(baseline) + torch.tensor([-2.0, 0.0, 2.0]))
    assert torch.allclose(fused, expected, atol=1e-6)


def test_crossfit_schedule_inherits_selected_full_bank_epoch() -> None:
    protocol = json.loads(
        (ROOT / "configs/mars_gaussian_contrast_crossfit_protocol.json").read_text()
    )
    audit = json.loads(
        (ROOT / "reports/experiments/mars_gaussian_contrast_full_bank.json").read_text()
    )
    assert audit["passed"] is True
    assert audit["selected_epoch"]["epoch"] == 9
    assert protocol["synthetic_bank"]["fixed_selected_epoch"] == 9
    assert protocol["training"]["synthetic_pretrain_epochs"] == 9
    assert protocol["training"]["synthetic_requests_per_epoch"] == 32000
    assert protocol["folds"] == [3, 4]


def test_crossfit_protocol_keeps_confirmation_and_test_closed() -> None:
    protocol = json.loads(
        (ROOT / "configs/mars_gaussian_contrast_crossfit_protocol.json").read_text()
    )
    invariants = " ".join(protocol["invariants"]).lower()
    assert "fold 2" in invariants and "confirmation" in invariants
    assert "official 2024 test" in invariants
    assert protocol["outputs"]["artifact"].endswith(".pt")
