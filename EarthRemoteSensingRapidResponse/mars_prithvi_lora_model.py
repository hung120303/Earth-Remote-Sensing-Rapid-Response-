"""Parameter-efficient Prithvi scene/patch adaptation for MARS-S2L."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(
        self, base: nn.Linear, *, rank: int, alpha: float, dropout: float
    ) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(alpha) / float(rank)
        nn.init.kaiming_uniform_(self.a.weight, a=5**0.5)
        nn.init.zeros_(self.b.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + self.scale * self.b(self.a(self.dropout(values)))


def inject_lora(
    foundation: nn.Module,
    *,
    last_blocks: int,
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    encoder = foundation.encoder
    if last_blocks <= 0 or last_blocks > len(encoder.blocks):
        raise ValueError("LoRA block count is outside the Prithvi encoder depth")
    adapted: list[str] = []
    for index in range(len(encoder.blocks) - last_blocks, len(encoder.blocks)):
        attention = encoder.blocks[index].attn
        for name in ("qkv", "proj"):
            base = getattr(attention, name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Prithvi block {index} attention {name} is not linear")
            setattr(
                attention,
                name,
                LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout),
            )
            adapted.append(f"encoder.blocks.{index}.attn.{name}")
    return adapted


def encoder_tokens(
    encoder: nn.Module,
    values: torch.Tensor,
    temporal: torch.Tensor,
    location: torch.Tensor,
) -> torch.Tensor:
    """Prithvi forward_features without retaining all 12 intermediate clones."""
    sample_shape = values.shape[-3:]
    tokens = encoder.patch_embed(values)
    position = encoder.interpolate_pos_encoding(sample_shape)
    tokens = tokens + position[:, 1:, :]
    if encoder.temporal_encoding:
        per_frame = tokens.shape[1] // encoder.num_frames
        tokens = tokens + encoder.temporal_embed_enc(temporal, per_frame)
    if encoder.location_encoding:
        tokens = tokens + encoder.location_embed_enc(location)
    cls = (encoder.cls_token + position[:, :1, :]).expand(tokens.shape[0], -1, -1)
    tokens = torch.cat((cls, tokens), dim=1)
    for block in encoder.blocks:
        tokens = block(tokens)
    return encoder.norm(tokens)


class PrithviLoRASceneModel(nn.Module):
    """LoRA-adapted encoder with patch-supervised attention/top-k pooling."""

    def __init__(self, foundation: nn.Module, spec: dict[str, Any]) -> None:
        super().__init__()
        self.foundation = foundation
        for parameter in self.foundation.parameters():
            parameter.requires_grad_(False)
        self.adapted_modules = inject_lora(
            self.foundation,
            last_blocks=int(spec["last_blocks"]),
            rank=int(spec["rank"]),
            alpha=float(spec["alpha"]),
            dropout=float(spec["lora_dropout"]),
        )
        width = int(self.foundation.encoder.embed_dim)
        hidden = int(spec["head_hidden"])
        self.topk = int(spec["topk_patches"])
        self.patch_head = nn.Linear(width, 1)
        self.scene_head = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, hidden),
            nn.GELU(),
            nn.Dropout(float(spec["head_dropout"])),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        values: torch.Tensor,
        temporal: torch.Tensor,
        location: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = encoder_tokens(self.foundation.encoder, values, temporal, location)
        cls = tokens[:, 0]
        patches = tokens[:, 1:]
        if patches.shape[1] % 2 != 0:
            raise ValueError("Prithvi temporal patch count is not divisible by two")
        target = patches.reshape(patches.shape[0], 2, patches.shape[1] // 2, patches.shape[2])[:, 1]
        patch_logits = self.patch_head(target).squeeze(-1)
        attention = torch.softmax(patch_logits, dim=1)
        attended = torch.sum(target * attention.unsqueeze(-1), dim=1)
        top_indices = torch.topk(patch_logits, min(self.topk, patch_logits.shape[1]), dim=1).indices
        selected = torch.gather(
            target,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, target.shape[2]),
        )
        top_pooled = selected.mean(dim=1)
        scene_logit = self.scene_head(torch.cat((cls, attended, top_pooled), dim=1)).squeeze(1)
        grid = int(target.shape[1] ** 0.5)
        if grid * grid != target.shape[1]:
            raise ValueError("Prithvi target tokens do not form a square grid")
        return {
            "scene_logit": scene_logit,
            "patch_logits": patch_logits.reshape(-1, 1, grid, grid),
        }


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    trainable = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(state) != trainable:
        missing = sorted(trainable - set(state))
        extra = sorted(set(state) - trainable)
        raise ValueError(f"LoRA trainable-state schema differs: missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
