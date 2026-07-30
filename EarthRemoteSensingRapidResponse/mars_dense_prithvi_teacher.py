"""Dense Prithvi-token fusion around the frozen released MARS-S2L U-Net."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mars_paper_model import ResidualBlock
from mars_physics_guided_teacher import (
    IdentityFeatureGate,
    PhysicsGuidedTeacherAdapter,
)


PRITHVI_BLOCKS = (3, 6, 9, 12)
PRITHVI_EMBED_DIM = 192
PRITHVI_GRID_SIZE = 8
PRITHVI_CHANNELS = len(PRITHVI_BLOCKS) * PRITHVI_EMBED_DIM


class TokenProjection(nn.Module):
    """Normalize one frozen Prithvi depth and retain its spatial geometry."""

    def __init__(self, output_channels: int = 64) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(12, PRITHVI_EMBED_DIM, affine=False),
            nn.Conv2d(PRITHVI_EMBED_DIM, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            ResidualBlock(output_channels, output_channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class DensePrithviTeacherAdapter(PhysicsGuidedTeacherAdapter):
    """Inject frozen foundation tokens into deep teacher features.

    The released segmentation logits and the supplied cross-fitted scene score
    are both reproduced at initialization.  Learned changes are therefore
    residual, bounded, and independently auditable.
    """

    def __init__(self) -> None:
        super().__init__()
        # The parent creates five physics-only gates.  This architecture replaces
        # them with three token/physics fusion gates, so do not retain unused
        # trainable parameters in the optimizer or artifact.
        del self.gates
        self.token_projections = nn.ModuleList(TokenProjection() for _ in PRITHVI_BLOCKS)
        self.deep_fusions = nn.ModuleList(
            (
                ResidualBlock(96 + 64 + 64, 128),
                ResidualBlock(128 + 64 + 64, 160),
                ResidualBlock(160 + 64 + 64, 192),
            )
        )
        self.deep_gates = nn.ModuleList(
            (
                IdentityFeatureGate(128, 256),
                IdentityFeatureGate(160, 512),
                IdentityFeatureGate(192, 512),
            )
        )
        self.patch_head = nn.Conv2d(128, 1, kernel_size=1)
        self.scene_attention = nn.Conv2d(192, 1, kernel_size=1)
        self.scene_fusion = ResidualBlock(512 + 192, 192)
        self.sensor_embedding = nn.Embedding(2, 8)
        self.scene_hidden = nn.Sequential(
            nn.Linear(192 * 3 + 8 + 1, 192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(192, 64),
            nn.GELU(),
        )
        self.scene_output = nn.Linear(64, 1)
        nn.init.zeros_(self.scene_output.weight)
        nn.init.zeros_(self.scene_output.bias)

    @staticmethod
    def _resize(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            values, size=target.shape[-2:], mode="bilinear", align_corners=False
        )

    def _token_maps(self, tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
        expected = (PRITHVI_CHANNELS, PRITHVI_GRID_SIZE, PRITHVI_GRID_SIZE)
        if tokens.ndim != 4 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"Expected Bx{expected[0]}x{expected[1]}x{expected[2]} tokens")
        chunks = tokens.split(PRITHVI_EMBED_DIM, dim=1)
        return tuple(
            projection(chunk)
            for projection, chunk in zip(self.token_projections, chunks)
        )

    def _scene_residual(
        self,
        teacher_level5: torch.Tensor,
        source_level5: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
        base_scene_logit: torch.Tensor,
    ) -> torch.Tensor:
        scene_map = self.scene_fusion(
            torch.cat(
                (
                    teacher_level5,
                    self._resize(source_level5, teacher_level5),
                ),
                dim=1,
            )
        )
        visible = F.interpolate(
            observable.float(),
            size=scene_map.shape[-2:],
            mode="nearest",
        )
        flat = scene_map.flatten(2)
        valid = visible.flatten(2) > 0.5
        weights = valid.to(flat.dtype)
        average = (flat * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        maximum = flat.masked_fill(~valid, -1e4).amax(dim=2)
        attention_logits = self.scene_attention(scene_map).flatten(2)
        attention_logits = attention_logits.masked_fill(~valid, -1e4)
        attention = torch.softmax(attention_logits, dim=2)
        attended = (flat * attention).sum(dim=2)
        features = torch.cat(
            (
                attended,
                average,
                maximum,
                self.sensor_embedding(sensor_index),
                base_scene_logit[:, None],
            ),
            dim=1,
        )
        return 2.0 * torch.tanh(self.scene_output(self.scene_hidden(features)).squeeze(1))

    def forward(
        self,
        values: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
        prithvi_tokens: torch.Tensor,
        base_scene_score: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if base_scene_score.ndim != 1 or base_scene_score.shape[0] != values.shape[0]:
            raise ValueError("Cross-fitted base scene scores must be a length-B vector")
        with torch.no_grad():
            teacher_levels = self._encode(values)
            baseline_logits = self._decode(teacher_levels)
        physics_levels = self.physics(values, baseline_logits.detach(), sensor_index)
        token3, token6, token9, token12 = self._token_maps(prithvi_tokens)
        source3 = self.deep_fusions[0](
            torch.cat(
                (
                    physics_levels[2],
                    self._resize(token3, physics_levels[2]),
                    self._resize(token6, physics_levels[2]),
                ),
                dim=1,
            )
        )
        source4 = self.deep_fusions[1](
            torch.cat(
                (
                    physics_levels[3],
                    self._resize(token6, physics_levels[3]),
                    self._resize(token9, physics_levels[3]),
                ),
                dim=1,
            )
        )
        source5 = self.deep_fusions[2](
            torch.cat(
                (
                    physics_levels[4],
                    self._resize(token9, physics_levels[4]),
                    self._resize(token12, physics_levels[4]),
                ),
                dim=1,
            )
        )
        adapted = (
            teacher_levels[0].detach(),
            teacher_levels[1].detach(),
            self.deep_gates[0](teacher_levels[2].detach(), source3),
            self.deep_gates[1](teacher_levels[3].detach(), source4),
            self.deep_gates[2](teacher_levels[4].detach(), source5),
        )
        logits = self._decode(adapted)
        base_scene_logit = torch.logit(base_scene_score.float().clamp(1e-6, 1.0 - 1e-6))
        scene_delta = self._scene_residual(
            adapted[4],
            source5,
            observable,
            sensor_index,
            base_scene_logit,
        )
        patch_logits = F.adaptive_avg_pool2d(
            self.patch_head(source3), (PRITHVI_GRID_SIZE, PRITHVI_GRID_SIZE)
        )
        return {
            "segmentation_logits": logits,
            "baseline_logits": baseline_logits,
            "correction_logits": logits - baseline_logits,
            "patch_logits": patch_logits,
            "base_scene_logit": base_scene_logit,
            "scene_delta_logit": scene_delta,
            "scene_logit": base_scene_logit + scene_delta,
        }
