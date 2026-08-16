from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_prithvi_domain_residual import DomainAdaptiveResidualSceneModel


class _Attention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = _Attention(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.attn.proj(self.attn.qkv(values)[..., : values.shape[-1]])


class _PatchEmbed(nn.Module):
    input_size = (2, 2, 2)
    patch_size = (1, 1, 1)

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(1, width, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        embedded = self.proj(values)
        return embedded.permute(0, 2, 3, 4, 1).reshape(
            values.shape[0], -1, embedded.shape[1]
        )


class _Encoder(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.embed_dim = width
        self.num_frames = 2
        self.temporal_encoding = False
        self.location_encoding = False
        self.patch_embed = _PatchEmbed(width)
        self.pos_embed = nn.Parameter(torch.zeros(1, 9, width))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList([_Block(width), _Block(width)])
        self.norm = nn.LayerNorm(width)

    def interpolate_pos_encoding(
        self, sample_shape: tuple[int, int, int]
    ) -> torch.Tensor:
        del sample_shape
        return self.pos_embed


class _Foundation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()


def _spec() -> dict[str, float | int]:
    return {
        "last_blocks": 1,
        "rank": 2,
        "alpha": 2.0,
        "lora_dropout": 0.0,
        "head_hidden": 8,
        "head_dropout": 0.0,
        "topk_patches": 2,
        "sensor_embedding_dim": 2,
        "correction_bound": 0.5,
    }


def test_new_residual_is_exactly_zero_and_shapes_are_stable() -> None:
    model = DomainAdaptiveResidualSceneModel(_Foundation(), _spec()).eval()
    result = model(
        torch.randn(3, 1, 2, 2, 2),
        torch.zeros(3, 2, 2),
        torch.zeros(3, 2),
        torch.tensor([0, 1, 0]),
    )
    assert result["raw_residual"].shape == (3,)
    assert torch.equal(result["bounded_correction"], torch.zeros(3))
    assert result["patch_logits"].shape == (3, 1, 2, 2)


def test_residual_head_and_lora_receive_finite_gradients() -> None:
    model = DomainAdaptiveResidualSceneModel(_Foundation(), _spec()).train()
    result = model(
        torch.randn(4, 1, 2, 2, 2),
        torch.zeros(4, 2, 2),
        torch.zeros(4, 2),
        torch.tensor([0, 1, 0, 1]),
    )
    loss = result["bounded_correction"].sum() + result["patch_logits"].square().mean()
    loss.backward()
    gradients = [
        value.grad
        for value in model.parameters()
        if value.requires_grad and value.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("foundation.") and ".a." not in name and ".b." not in name
    )
