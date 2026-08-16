"""Protected scene-residual head for domain-adapted Prithvi features."""

from __future__ import annotations

from typing import Any

import torch
from mars_prithvi_lora_model import encoder_tokens, inject_lora
from torch import nn


class DomainAdaptiveResidualSceneModel(nn.Module):
    """Fuse temporal patch changes and learn a bounded correction to a baseline.

    The caller owns the protected baseline logit.  This module deliberately
    returns only a bounded residual, initialized to exactly zero, so a newly
    constructed model reproduces that baseline before supervised adaptation.
    """

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
        sensor_width = int(spec["sensor_embedding_dim"])
        self.topk = int(spec["topk_patches"])
        self.correction_bound = float(spec["correction_bound"])
        if self.topk <= 0 or self.correction_bound <= 0.0:
            raise ValueError("top-k and correction bound must be positive")
        self.patch_fusion = nn.Sequential(
            nn.LayerNorm(width * 4),
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.patch_head = nn.Linear(width, 1)
        self.sensor_embedding = nn.Embedding(2, sensor_width)
        self.scene_head = nn.Sequential(
            nn.LayerNorm(width * 4 + sensor_width),
            nn.Linear(width * 4 + sensor_width, hidden),
            nn.GELU(),
            nn.Dropout(float(spec["head_dropout"])),
            nn.Linear(hidden, 1),
        )
        final = self.scene_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        values: torch.Tensor,
        temporal: torch.Tensor,
        location: torch.Tensor,
        sensor_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tokens = encoder_tokens(self.foundation.encoder, values, temporal, location)
        cls = tokens[:, 0]
        patches = tokens[:, 1:]
        if patches.shape[1] % 2:
            raise ValueError("Prithvi temporal patch count is not divisible by two")
        frames = patches.reshape(
            patches.shape[0], 2, patches.shape[1] // 2, patches.shape[2]
        )
        reference, target = frames.unbind(dim=1)
        difference = target - reference
        fused = self.patch_fusion(
            torch.cat((reference, target, difference, difference.abs()), dim=-1)
        )
        patch_logits = self.patch_head(fused).squeeze(-1)
        attention = torch.softmax(patch_logits, dim=1)
        attended = torch.sum(fused * attention.unsqueeze(-1), dim=1)
        mean = fused.mean(dim=1)
        count = min(self.topk, patch_logits.shape[1])
        indices = torch.topk(patch_logits, count, dim=1).indices
        selected = torch.gather(
            fused, 1, indices.unsqueeze(-1).expand(-1, -1, fused.shape[-1])
        )
        top = selected.mean(dim=1)
        sensor = sensor_index.to(device=values.device, dtype=torch.long).reshape(-1)
        if sensor.shape != (values.shape[0],) or torch.any((sensor < 0) | (sensor > 1)):
            raise ValueError("sensor index must be a B-vector containing only 0/1")
        pooled = torch.cat(
            (cls, attended, mean, top, self.sensor_embedding(sensor)), dim=1
        )
        raw = self.scene_head(pooled).squeeze(1)
        correction = self.correction_bound * torch.tanh(raw)
        grid = int(fused.shape[1] ** 0.5)
        if grid * grid != fused.shape[1]:
            raise ValueError("Prithvi frame tokens do not form a square grid")
        return {
            "raw_residual": raw,
            "bounded_correction": correction,
            "patch_logits": patch_logits.reshape(-1, 1, grid, grid),
        }
