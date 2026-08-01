from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_anchored_full_finetune import AnchoredMarsFullFinetune  # noqa: E402
from mars_paper_model import ReleasedMarsUNet  # noqa: E402


def released_like_state() -> dict[str, torch.Tensor]:
    return ReleasedMarsUNet().state_dict()


def test_initial_student_is_exact_teacher_identity() -> None:
    torch.manual_seed(7)
    model = AnchoredMarsFullFinetune(scene_topk_fraction=0.05)
    model.load_released_checkpoint(released_like_state())
    values = torch.randn(2, 16, 32, 32)
    observable = torch.ones(2, 1, 32, 32)
    sensors = torch.tensor([0, 1])
    model.eval()
    output = model(values, observable, sensors)
    assert torch.equal(output["segmentation_logits"], output["baseline_logits"])
    assert torch.count_nonzero(output["correction_logits"]) == 0
    assert torch.count_nonzero(output["scene_delta_logit"]) == 0
    assert model.anchor_penalty().item() == 0.0


def test_training_keeps_batch_norm_state_frozen() -> None:
    torch.manual_seed(11)
    model = AnchoredMarsFullFinetune()
    model.load_released_checkpoint(released_like_state())
    before = {
        name: value.clone()
        for name, value in model.student.state_dict().items()
        if "running_" in name or "num_batches_tracked" in name
    }
    model.train()
    assert all(
        not module.training
        for module in model.student.modules()
        if isinstance(module, nn.BatchNorm2d)
    )
    model(torch.randn(2, 16, 32, 32), torch.ones(2, 1, 32, 32), torch.tensor([0, 1]))
    after = model.student.state_dict()
    assert all(torch.equal(value, after[name]) for name, value in before.items())


def test_gradient_step_moves_student_and_activates_anchor() -> None:
    torch.manual_seed(13)
    model = AnchoredMarsFullFinetune(scene_topk_fraction=0.05)
    model.load_released_checkpoint(released_like_state())
    optimizer = torch.optim.SGD(
        model.parameter_groups(backbone_learning_rate=1e-4, output_learning_rate=1e-3)
    )
    model.train()
    output = model(
        torch.randn(2, 16, 32, 32),
        torch.ones(2, 1, 32, 32),
        torch.tensor([0, 1]),
    )
    loss = output["segmentation_logits"].square().mean()
    loss.backward()
    optimizer.step()
    assert model.anchor_penalty().item() > 0.0
    assert torch.isfinite(model.anchor_penalty())


def test_zero_strength_scene_fusion_is_exact_identity() -> None:
    base = torch.tensor([0.1, 0.8])
    delta = torch.tensor([100.0, -100.0])
    sensor = torch.tensor([0, 1])
    fused = AnchoredMarsFullFinetune.fuse_scene_score(base, delta, sensor, 0.0)
    assert torch.equal(fused, base)
