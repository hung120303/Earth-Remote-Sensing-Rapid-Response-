"""Utilities for label-free Prithvi extended pretraining.

The helpers in this module deliberately keep the foundation model's public
interface intact.  They are used by the planned masked-autoencoder adaptation
stage and are also useful for unit-testing the trainability and merge
contracts before any protected MARS-S2L result is opened.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from mars_prithvi_lora_model import LoRALinear, inject_lora
from torch import nn


def _trainable_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def configure_extended_pretraining(
    foundation: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    fully_unfrozen_last_blocks: int = 1,
) -> dict[str, Any]:
    """Configure a Prithvi MAE for parameter-efficient extended pretraining.

    All foundation parameters are frozen first.  LoRA adapters are then added
    to ``qkv`` and ``proj`` in every encoder block.  The patch embed and the
    final ``fully_unfrozen_last_blocks`` transformer blocks are trainable; the
    latter includes their base attention weights in addition to their LoRA
    weights.  The MAE decoder is fully trainable.

    A JSON-friendly receipt is returned so a training run can record exactly
    which modules and parameter counts were adapted.
    """

    if rank <= 0:
        raise ValueError("rank must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if dropout < 0 or dropout >= 1:
        raise ValueError("dropout must be in [0, 1)")

    encoder = getattr(foundation, "encoder", None)
    decoder = getattr(foundation, "decoder", None)
    if encoder is None or decoder is None or not hasattr(encoder, "blocks"):
        raise TypeError("foundation must expose encoder.blocks and decoder")
    depth = len(encoder.blocks)
    if fully_unfrozen_last_blocks < 0 or fully_unfrozen_last_blocks > depth:
        raise ValueError("fully_unfrozen_last_blocks is outside encoder depth")

    for parameter in foundation.parameters():
        parameter.requires_grad_(False)

    adapted_modules = inject_lora(
        foundation,
        last_blocks=depth,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    # ``inject_lora`` constructs new Linear layers on the default CPU.  The
    # foundation may already live on CUDA, so align every adapter with its
    # frozen base immediately instead of relying on a later whole-model move.
    for module in foundation.modules():
        if isinstance(module, LoRALinear):
            device = module.base.weight.device
            dtype = module.base.weight.dtype
            module.a.to(device=device, dtype=dtype)
            module.b.to(device=device, dtype=dtype)

    # Patch embedding adaptation is intentionally small and product-agnostic.
    for parameter in encoder.patch_embed.parameters():
        parameter.requires_grad_(True)

    first_unfrozen = depth - fully_unfrozen_last_blocks
    for block in encoder.blocks[first_unfrozen:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)

    for parameter in decoder.parameters():
        parameter.requires_grad_(True)

    trainable_encoder_params = _trainable_count(encoder)
    trainable_decoder_params = _trainable_count(decoder)
    trainable_total_params = trainable_encoder_params + trainable_decoder_params
    return {
        "adapted_modules": adapted_modules,
        "encoder_depth": depth,
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "fully_unfrozen_last_blocks": int(fully_unfrozen_last_blocks),
        "trainable_encoder_params": trainable_encoder_params,
        "trainable_decoder_params": trainable_decoder_params,
        "trainable_total_params": trainable_total_params,
        # Short aliases keep the receipt convenient for lightweight callers;
        # the explicit *_params keys are the canonical serialized fields.
        "trainable_encoder": trainable_encoder_params,
        "trainable_decoder": trainable_decoder_params,
        "trainable_total": trainable_total_params,
    }


def merge_lora_inplace(foundation: nn.Module) -> list[str]:
    """Fold every encoder attention LoRA adapter into its base linear layer.

    The replacement is performed on the original device and dtype.  Base
    parameter trainability is retained, which makes this safe both before
    evaluation and when a merged model is returned to fine-tuning.
    """

    encoder = getattr(foundation, "encoder", None)
    if encoder is None or not hasattr(encoder, "blocks"):
        raise TypeError("foundation must expose encoder.blocks")

    merged: list[str] = []
    for index, block in enumerate(encoder.blocks):
        attention = getattr(block, "attn", None)
        if attention is None:
            continue
        for name in ("qkv", "proj"):
            module = getattr(attention, name, None)
            if not isinstance(module, LoRALinear):
                continue
            base = module.base
            device = base.weight.device
            dtype = base.weight.dtype
            with torch.no_grad():
                delta = module.b.weight.to(
                    device=device, dtype=dtype
                ) @ module.a.weight.to(device=device, dtype=dtype)
                merged_weight = base.weight.to(device=device, dtype=dtype) + (
                    module.scale * delta
                )

            linear = nn.Linear(
                base.in_features,
                base.out_features,
                bias=base.bias is not None,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                linear.weight.copy_(merged_weight)
                if base.bias is not None:
                    linear.bias.copy_(base.bias.to(device=device, dtype=dtype))
            linear.weight.requires_grad_(base.weight.requires_grad)
            if linear.bias is not None and base.bias is not None:
                linear.bias.requires_grad_(base.bias.requires_grad)
            setattr(attention, name, linear)
            merged.append(f"encoder.blocks.{index}.attn.{name}")

    remaining = [
        name
        for name, module in foundation.named_modules()
        if isinstance(module, LoRALinear)
    ]
    if remaining:
        raise AssertionError(f"LoRALinear modules remain after merge: {remaining}")
    return merged


def _validate_band_groups(
    band_groups: Sequence[Sequence[int]], channels: int
) -> tuple[tuple[int, ...], ...]:
    groups = tuple(tuple(int(channel) for channel in group) for group in band_groups)
    if not groups or any(not group for group in groups):
        raise ValueError("band_groups must contain non-empty groups")
    flattened = [channel for group in groups for channel in group]
    if sorted(flattened) != list(range(channels)) or len(set(flattened)) != channels:
        raise ValueError("band_groups must partition every input channel exactly once")
    return groups


def _group_normalize(
    patch_values: torch.Tensor,
    groups: Iterable[Sequence[int]],
    epsilon: float,
) -> torch.Tensor:
    """Normalize channels in ``patch_values`` per frame, spatial patch, group."""

    # Layout: B, temporal-patches, H-patches, W-patches, patch-time, p, q, C.
    channels = patch_values.shape[-1]
    normalized_channels: list[torch.Tensor | None] = [None] * channels
    for group in groups:
        indices = list(group)
        values = patch_values[..., indices]
        # The Prithvi patch-time dimension is normally one.  Spatial pixels and
        # group bands are the normalization population; each patch-time slice
        # remains independently normalized when a wider temporal patch exists.
        mean = values.mean(dim=(-3, -2, -1), keepdim=True)
        variance = values.var(dim=(-3, -2, -1), unbiased=False, keepdim=True)
        scale = torch.sqrt(variance + epsilon)
        group_normalized = (values - mean) / scale
        for offset, channel in enumerate(indices):
            normalized_channels[channel] = group_normalized[..., offset : offset + 1]
    if any(value is None for value in normalized_channels):
        raise AssertionError(
            "internal band-group normalization did not cover all channels"
        )
    return torch.cat(
        [value for value in normalized_channels if value is not None], dim=-1
    )


def patch_group_normalized_l1_loss(
    foundation: nn.Module,
    pixel_values: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    band_groups: Sequence[Sequence[int]] = ((0, 1, 2), (3,), (4, 5)),
    epsilon: float = 1e-6,
    temporal_difference_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Compute masked group-normalized MAE loss with a temporal difference term.

    ``pixel_values`` follows the Prithvi convention ``(B,C,T,H,W)`` and
    ``pred`` is the corresponding output of ``foundation.decoder`` in its
    ``(B,L,patch_time*patch_height*patch_width*C)`` patchified layout.  Target
    statistics are used to normalize both target and prediction, avoiding a
    train/eval mismatch while making the reconstruction objective invariant to
    per-patch affine radiometric changes within each declared spectral group.
    """

    if pixel_values.ndim != 5:
        raise ValueError("pixel_values must have shape (B,C,T,H,W)")
    if pred.ndim != 3:
        raise ValueError("pred must have shape (B,num_patches,patch_vector)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if temporal_difference_weight < 0:
        raise ValueError("temporal_difference_weight must be non-negative")

    batch, channels, frames, height, width = pixel_values.shape
    groups = _validate_band_groups(band_groups, channels)
    patchified_target = foundation.patchify(pixel_values)
    if patchified_target.ndim != 3 or patchified_target.shape[0] != batch:
        raise ValueError("foundation.patchify returned an invalid target shape")
    if pred.shape != patchified_target.shape:
        raise ValueError(
            "pred must have the same shape as foundation.patchify(pixel_values): "
            f"got {tuple(pred.shape)} vs {tuple(patchified_target.shape)}"
        )

    patch_size = tuple(
        int(value) for value in foundation.encoder.patch_embed.patch_size
    )
    if len(patch_size) != 3:
        raise ValueError("encoder.patch_embed.patch_size must be a 3-tuple")
    patch_frames, patch_height, patch_width = patch_size
    if patch_frames != 1:
        raise ValueError("temporal difference requires a one-frame temporal patch size")
    if frames != 2:
        raise ValueError("temporal difference requires exactly two input frames")
    if height % patch_height or width % patch_width:
        raise ValueError("input spatial dimensions must be divisible by patch size")
    grid_height = height // patch_height
    grid_width = width // patch_width
    expected_patches = frames * grid_height * grid_width
    if patchified_target.shape[1] != expected_patches:
        raise ValueError(
            f"patchify returned {patchified_target.shape[1]} patches; expected {expected_patches}"
        )
    expected_vector = patch_height * patch_width * channels
    if patchified_target.shape[2] != expected_vector:
        raise ValueError(
            f"patchify returned vector width {patchified_target.shape[2]}; expected {expected_vector}"
        )
    if mask.shape != (batch, expected_patches):
        raise ValueError("mask must have shape (B,num_patches)")
    mask_bool = mask.to(device=pred.device).bool()
    if not bool(mask_bool.any().item()):
        raise ValueError("mask must contain at least one masked patch")

    layout = (
        batch,
        frames,
        grid_height,
        grid_width,
        1,
        patch_height,
        patch_width,
        channels,
    )
    target_layout = patchified_target.reshape(layout)
    pred_layout = pred.reshape(layout)
    target_normalized = _group_normalize(target_layout, groups, epsilon)
    pred_normalized = _group_normalize_with_stats(
        pred_layout, target_layout, groups, epsilon
    )

    reconstruction_per_patch = (
        (pred_normalized - target_normalized).abs().mean(dim=(-1, -2, -3, -4))
    )
    reconstruction = reconstruction_per_patch.reshape(-1)[mask_bool.reshape(-1)].mean()

    frame_mask = mask_bool.reshape(batch, frames, grid_height, grid_width)
    temporal_mask = frame_mask[:, 0] | frame_mask[:, 1]
    target_difference = target_normalized[:, 1] - target_normalized[:, 0]
    pred_difference = pred_normalized[:, 1] - pred_normalized[:, 0]
    temporal_per_patch = (
        (pred_difference - target_difference).abs().mean(dim=(-1, -2, -3, -4))
    )
    temporal_difference = temporal_per_patch[temporal_mask].mean()
    loss = reconstruction + float(temporal_difference_weight) * temporal_difference
    return {
        "loss": loss,
        "reconstruction": reconstruction,
        "temporal_difference": temporal_difference,
    }


def _group_normalize_with_stats(
    patch_values: torch.Tensor,
    target_values: torch.Tensor,
    groups: Iterable[Sequence[int]],
    epsilon: float,
) -> torch.Tensor:
    """Normalize predictions using target patch/group statistics."""

    channels = patch_values.shape[-1]
    normalized_channels: list[torch.Tensor | None] = [None] * channels
    for group in groups:
        indices = list(group)
        target_group = target_values[..., indices]
        pred_group = patch_values[..., indices]
        mean = target_group.mean(dim=(-3, -2, -1), keepdim=True)
        variance = target_group.var(dim=(-3, -2, -1), unbiased=False, keepdim=True)
        scale = torch.sqrt(variance + epsilon)
        group_normalized = (pred_group - mean) / scale
        for offset, channel in enumerate(indices):
            normalized_channels[channel] = group_normalized[..., offset : offset + 1]
    if any(value is None for value in normalized_channels):
        raise AssertionError(
            "internal band-group normalization did not cover all channels"
        )
    return torch.cat(
        [value for value in normalized_channels if value is not None], dim=-1
    )
