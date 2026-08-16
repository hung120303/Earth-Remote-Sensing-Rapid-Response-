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

from mars_prithvi_domain_adaptation import (
    configure_extended_pretraining,
    merge_lora_inplace,
    patch_group_normalized_l1_loss,
)
from mars_prithvi_lora_model import LoRALinear


class DummyAttention(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)


class DummyBlock(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.attn = DummyAttention(width)
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )


class DummyEncoder(nn.Module):
    def __init__(self, depth: int = 3, channels: int = 6) -> None:
        super().__init__()
        self.patch_embed = nn.Conv3d(
            channels, 8, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.patch_embed.patch_size = (1, 2, 2)
        self.blocks = nn.ModuleList([DummyBlock() for _ in range(depth)])
        self.norm = nn.LayerNorm(8)


class DummyFoundation(nn.Module):
    def __init__(self, depth: int = 3, channels: int = 6) -> None:
        super().__init__()
        self.encoder = DummyEncoder(depth=depth, channels=channels)
        self.decoder = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))

    def patchify(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = pixel_values.shape
        patch_height = patch_width = 2
        values = pixel_values.reshape(
            batch,
            channels,
            frames,
            height // patch_height,
            patch_height,
            width // patch_width,
            patch_width,
        )
        values = values.permute(0, 2, 3, 5, 4, 6, 1)
        return values.reshape(
            batch, frames * (height // patch_height) * (width // patch_width), -1
        )


def test_configuration_receipt_and_trainability_contract() -> None:
    foundation = DummyFoundation()
    receipt = configure_extended_pretraining(
        foundation, rank=2, alpha=4.0, dropout=0.0, fully_unfrozen_last_blocks=1
    )

    assert len(receipt["adapted_modules"]) == len(foundation.encoder.blocks) * 2
    assert receipt["trainable_total_params"] == (
        receipt["trainable_encoder_params"] + receipt["trainable_decoder_params"]
    )
    assert receipt["trainable_encoder_params"] > 0
    assert receipt["trainable_decoder_params"] == sum(
        parameter.numel() for parameter in foundation.decoder.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in foundation.encoder.patch_embed.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in foundation.encoder.blocks[-1].parameters()
    )
    assert not foundation.encoder.blocks[0].norm.weight.requires_grad
    assert foundation.encoder.blocks[0].attn.qkv.a.weight.requires_grad
    assert not foundation.encoder.blocks[0].attn.qkv.base.weight.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contract")
def test_configuration_places_new_adapters_with_cuda_foundation() -> None:
    foundation = DummyFoundation().cuda()
    configure_extended_pretraining(
        foundation, rank=2, alpha=4.0, dropout=0.0, fully_unfrozen_last_blocks=1
    )
    for module in foundation.modules():
        if isinstance(module, LoRALinear):
            assert module.a.weight.device == module.base.weight.device
            assert module.b.weight.device == module.base.weight.device


def test_merge_is_numerically_equivalent_with_dropout_disabled() -> None:
    foundation = DummyFoundation(depth=2)
    configure_extended_pretraining(foundation, rank=2, alpha=3.0, dropout=0.0)
    foundation.eval()
    with torch.no_grad():
        for block in foundation.encoder.blocks:
            for name in ("qkv", "proj"):
                module = getattr(block.attn, name)
                assert isinstance(module, LoRALinear)
                module.a.weight.normal_(mean=0.0, std=0.2)
                module.b.weight.normal_(mean=0.0, std=0.2)

    inputs = torch.randn(5, 8)
    before = [
        (block.attn.qkv(inputs), block.attn.proj(inputs))
        for block in foundation.encoder.blocks
    ]
    merged = merge_lora_inplace(foundation)
    assert len(merged) == 4
    assert not any(isinstance(module, LoRALinear) for module in foundation.modules())
    after = [
        (block.attn.qkv(inputs), block.attn.proj(inputs))
        for block in foundation.encoder.blocks
    ]
    for (before_qkv, before_proj), (after_qkv, after_proj) in zip(before, after):
        assert torch.allclose(before_qkv, after_qkv, atol=1e-6, rtol=1e-6)
        assert torch.allclose(before_proj, after_proj, atol=1e-6, rtol=1e-6)


def test_group_normalized_loss_is_affine_invariant_and_temporal_zero_for_exact_reconstruction() -> (
    None
):
    foundation = DummyFoundation()
    torch.manual_seed(4)
    target = torch.randn(2, 6, 2, 4, 4)
    prediction = target + 0.15
    mask = torch.zeros(2, 8)
    mask[:, [0, 4, 7]] = 1
    first = patch_group_normalized_l1_loss(
        foundation, target, foundation.patchify(prediction), mask
    )

    scale = torch.tensor([2.0, 2.0, 2.0, 0.5, 3.0, 3.0]).reshape(1, 6, 1, 1, 1)
    offset = torch.tensor([7.0, 7.0, 7.0, -2.0, 11.0, 11.0]).reshape(1, 6, 1, 1, 1)
    transformed_target = target * scale + offset
    transformed_prediction = prediction * scale + offset
    second = patch_group_normalized_l1_loss(
        foundation,
        transformed_target,
        foundation.patchify(transformed_prediction),
        mask,
    )
    assert torch.allclose(first["loss"], second["loss"], atol=1e-5, rtol=1e-5)

    exact = patch_group_normalized_l1_loss(
        foundation,
        target,
        foundation.patchify(target),
        mask,
        temporal_difference_weight=0.2,
    )
    assert exact["reconstruction"].item() == pytest.approx(0.0, abs=1e-6)
    assert exact["temporal_difference"].item() == pytest.approx(0.0, abs=1e-6)


def test_group_normalized_loss_has_finite_gradients_and_validates_mask_and_frames() -> (
    None
):
    foundation = DummyFoundation()
    pixel_values = torch.randn(1, 6, 2, 4, 4)
    prediction = foundation.patchify(pixel_values).clone().requires_grad_(True)
    mask = torch.zeros(1, 8)
    mask[:, 1] = 1
    result = patch_group_normalized_l1_loss(foundation, pixel_values, prediction, mask)
    result["loss"].backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()

    with pytest.raises(ValueError, match="at least one"):
        patch_group_normalized_l1_loss(
            foundation, pixel_values, prediction, torch.zeros_like(mask)
        )
    with pytest.raises(ValueError, match="exactly two"):
        one_frame = torch.randn(1, 6, 1, 4, 4)
        one_frame_prediction = foundation.patchify(one_frame)
        patch_group_normalized_l1_loss(
            foundation,
            one_frame,
            one_frame_prediction,
            torch.ones(1, 4),
        )
