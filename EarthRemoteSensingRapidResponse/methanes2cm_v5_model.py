"""Tri-temporal, segmentation-first MethaneS2CM detector for ERSRR v5."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mars_v4_model import DecoderBlock, ResidualBlock
from methanes2cm_adapter import V5_INPUT_CHANNELS

MODEL_NAME = "ersrr_methanes2cm_tri_temporal_v5"
CONTEXT_MODEL_NAME = "ersrr_methanes2cm_tri_temporal_context_v5_1"
MODEL_SCHEMA_VERSION = 1
PHYSICS_SLICE = slice(0, 2)
TARGET_SLICE = slice(2, 8)
REFERENCE90_SLICE = slice(8, 14)
REFERENCE365_SLICE = slice(14, 20)


class SharedTriTemporalEncoder(nn.Module):
    """Encode T, T-90, and T-365 with exactly shared parameters."""

    channels = (24, 48, 96, 160, 256)

    def __init__(self) -> None:
        super().__init__()
        self.stem = ResidualBlock(6, self.channels[0])
        self.stages = nn.ModuleList(
            ResidualBlock(previous, current, stride=2)
            for previous, current in zip(self.channels[:-1], self.channels[1:])
        )

    def forward(self, values: torch.Tensor) -> list[torch.Tensor]:
        features = [self.stem(values)]
        for stage in self.stages:
            features.append(stage(features[-1]))
        return features


class TriTemporalFusion(nn.Module):
    """Fuse absolute spectra and both target-reference changes at one scale."""

    def __init__(self, temporal_channels: int, output_channels: int) -> None:
        super().__init__()
        # T, both references, signed and absolute T-reference changes, plus
        # the two centered MBMP physics channels: 7*C + 2 channels in total.
        self.block = ResidualBlock(temporal_channels * 7 + 2, output_channels)

    def forward(
        self,
        target: torch.Tensor,
        reference90: torch.Tensor,
        reference365: torch.Tensor,
        physics: torch.Tensor,
    ) -> torch.Tensor:
        difference90 = target - reference90
        difference365 = target - reference365
        resized_physics = F.interpolate(
            physics, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.block(
            torch.cat(
                [
                    target,
                    reference90,
                    reference365,
                    difference90,
                    difference365,
                    torch.abs(difference90),
                    torch.abs(difference365),
                    resized_physics,
                ],
                dim=1,
            )
        )


class MethaneS2CMV5Model(nn.Module):
    """Shared tri-temporal U-Net with an optional small context-head ablation.

    The default v5 scene score is mask-derived. Setting a nonzero fixed context
    weight enables v5.1's controlled multitask response to the released
    dataset's unusually broad masks while retaining dense localization.
    """

    fused_channels = (32, 56, 96, 144, 192)

    def __init__(
        self,
        scene_topk_fraction: float = 0.01,
        scene_max_weight: float = 0.15,
        context_scene_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 < scene_topk_fraction <= 1.0:
            raise ValueError("scene_topk_fraction must be in (0, 1]")
        if not 0.0 <= scene_max_weight <= 1.0:
            raise ValueError("scene_max_weight must be in [0, 1]")
        if not 0.0 <= context_scene_weight <= 1.0:
            raise ValueError("context_scene_weight must be in [0, 1]")
        self.scene_topk_fraction = scene_topk_fraction
        self.scene_max_weight = scene_max_weight
        self.context_scene_weight = context_scene_weight
        self.encoder = SharedTriTemporalEncoder()
        self.fusions = nn.ModuleList(
            TriTemporalFusion(temporal, fused)
            for temporal, fused in zip(self.encoder.channels, self.fused_channels)
        )
        self.decoder4 = DecoderBlock(192, 144, 144)
        self.decoder3 = DecoderBlock(144, 96, 96)
        self.decoder2 = DecoderBlock(96, 56, 56)
        self.decoder1 = DecoderBlock(56, 32, 48)
        self.segmentation = nn.Conv2d(48, 1, 1)
        self.physics_prior = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        # Starting at zero makes the learned shortcut opt-in while the two MBMP
        # maps remain available to every temporal fusion scale.
        self.physics_prior_gain = nn.Parameter(torch.zeros(()))
        # v5.1's controlled ablation adds only this 50k-parameter bottleneck
        # context head. The v5 default is exactly unchanged (None, no state).
        context_features = 2 * self.fused_channels[-1] + 3 * 2
        self.context_scene = (
            nn.Sequential(
                nn.LayerNorm(context_features),
                nn.Linear(context_features, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
            if context_scene_weight > 0.0
            else None
        )

    def forward(
        self, inputs: torch.Tensor, observable: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != len(V5_INPUT_CHANNELS):
            raise ValueError(
                f"Expected Bx{len(V5_INPUT_CHANNELS)}xHxW input, got {tuple(inputs.shape)}"
            )
        if observable.shape != inputs[:, :1].shape:
            raise ValueError("Observable mask must be Bx1xHxW and match input dimensions")

        # MBMP is multiplicative and neutral at one. Centering makes zero mean
        # the physical no-change prior; clipping limits corrupt-ratio leverage.
        physics = torch.clamp(inputs[:, PHYSICS_SLICE] - 1.0, min=-1.0, max=4.0)
        target = self.encoder(inputs[:, TARGET_SLICE])
        reference90 = self.encoder(inputs[:, REFERENCE90_SLICE])
        reference365 = self.encoder(inputs[:, REFERENCE365_SLICE])
        fused = [
            fusion(target_value, ref90_value, ref365_value, physics)
            for fusion, target_value, ref90_value, ref365_value in zip(
                self.fusions, target, reference90, reference365
            )
        ]
        decoded = self.decoder4(fused[4], fused[3])
        decoded = self.decoder3(decoded, fused[2])
        decoded = self.decoder2(decoded, fused[1])
        decoded = self.decoder1(decoded, fused[0])
        segmentation_logits = self.segmentation(decoded)
        segmentation_logits = segmentation_logits + self.physics_prior_gain * self.physics_prior(
            physics
        )

        flat = segmentation_logits.flatten(1)
        valid = observable.flatten(1) > 0.5
        masked = flat.masked_fill(~valid, -1e4)
        topk_count = max(1, int(masked.shape[1] * self.scene_topk_fraction))
        topk = torch.topk(masked, k=topk_count, dim=1).values
        mask_scene_logit = (
            (1.0 - self.scene_max_weight) * topk.mean(dim=1)
            + self.scene_max_weight * topk.max(dim=1).values
        )
        result = {
            "segmentation_logits": segmentation_logits,
            "mask_scene_logit": mask_scene_logit,
            "dense_features": decoded,
        }
        if self.context_scene is None:
            result["scene_logit"] = mask_scene_logit
            return result
        bottleneck = fused[-1]
        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(bottleneck, 1).flatten(1),
                F.adaptive_max_pool2d(bottleneck, 1).flatten(1),
                physics.mean(dim=(-2, -1)),
                physics.amax(dim=(-2, -1)),
                physics.amin(dim=(-2, -1)),
            ],
            dim=1,
        )
        context_scene_logit = self.context_scene(pooled)[:, 0]
        result["context_scene_logit"] = context_scene_logit
        result["scene_logit"] = (
            (1.0 - self.context_scene_weight) * mask_scene_logit
            + self.context_scene_weight * context_scene_logit
        )
        return result

    def artifact_metadata(self) -> dict[str, Any]:
        percentage = 100.0 * self.scene_topk_fraction
        mask_score = (
            f"{1.0 - self.scene_max_weight:g} * mean(top {percentage:g}% observable "
            f"mask logits) + {self.scene_max_weight:g} * max(observable mask logits)"
        )
        scene_score = (
            f"sigmoid({mask_score})"
            if self.context_scene is None
            else (
                f"sigmoid({1.0 - self.context_scene_weight:g} * ({mask_score}) + "
                f"{self.context_scene_weight:g} * context logit)"
            )
        )
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_name": CONTEXT_MODEL_NAME if self.context_scene is not None else MODEL_NAME,
            "input_channels": list(V5_INPUT_CHANNELS),
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "frames": ["T", "T-90", "T-365"],
            "temporal_weight_sharing": True,
            "scene_classifier_head": self.context_scene is not None,
            "scene_score": scene_score,
            "scene_topk_fraction": self.scene_topk_fraction,
            "scene_max_weight": self.scene_max_weight,
            "context_scene_weight": self.context_scene_weight,
            "context_scene_features": (
                "mean/max fused bottleneck plus mean/max/min centered MBMP"
                if self.context_scene is not None
                else None
            ),
            "physics_channels": ["MBMP(T,T-90)-1", "MBMP(T,T-365)-1"],
            "initialization": "from_scratch",
        }
