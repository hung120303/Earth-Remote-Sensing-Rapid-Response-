from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_v6_error_correcting_pilot import effective_training_logits


def test_effective_training_logits_adds_unbounded_mars_residual() -> None:
    baseline = torch.tensor([0.2, 0.8])
    residual = torch.tensor([3.0, -3.0])
    protected = torch.tensor([True, True])
    actual = effective_training_logits(
        baseline, residual, protected, residual_scale=1.0
    )
    expected = torch.logit(baseline) + residual
    assert torch.allclose(actual, expected)


def test_effective_training_logits_keeps_external_direct() -> None:
    baseline = torch.tensor([0.01, 0.99])
    residual = torch.tensor([0.7, -0.4])
    protected = torch.tensor([False, False])
    actual = effective_training_logits(
        baseline, residual, protected, residual_scale=1.0
    )
    assert torch.equal(actual, residual)
