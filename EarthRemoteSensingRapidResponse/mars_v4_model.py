"""Temporal-Siamese, segmentation-first methane detector for ERSRR v4."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAME = "ersrr_temporal_siamese_simulation_v4"
MODEL_SCHEMA_VERSION = 1
INPUT_CHANNELS = (
    "mbmp_release",
    "target_B02",
    "target_B03",
    "target_B04",
    "target_B08",
    "target_B11",
    "target_B12",
    "reference_B02",
    "reference_B03",
    "reference_B04",
    "reference_B08",
    "reference_B11",
    "reference_B12",
    "wind_u_div8",
    "wind_v_div8",
    "cloud_binary",
)
TARGET_SLICE = slice(1, 7)
REFERENCE_SLICE = slice(7, 13)
PHYSICS_INDICES = (0, 13, 14, 15)


def _groups(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.convolution1 = nn.Conv2d(
            input_channels, output_channels, 3, stride=stride, padding=1, bias=False
        )
        self.normalization1 = nn.GroupNorm(_groups(output_channels), output_channels)
        self.convolution2 = nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False)
        self.normalization2 = nn.GroupNorm(_groups(output_channels), output_channels)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        values = F.gelu(self.normalization1(self.convolution1(values)))
        values = self.normalization2(self.convolution2(values))
        return F.gelu(values + residual)


class SharedTemporalEncoder(nn.Module):
    """Encode target and reference with exactly shared weights."""

    channels = (32, 64, 128, 256, 384)

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


class TemporalFusion(nn.Module):
    def __init__(self, temporal_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = ResidualBlock(temporal_channels * 4 + 4, output_channels)

    def forward(
        self,
        target: torch.Tensor,
        reference: torch.Tensor,
        physics: torch.Tensor,
    ) -> torch.Tensor:
        resized_physics = F.interpolate(
            physics, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.block(
            torch.cat(
                [target, reference, target - reference, torch.abs(target - reference), resized_physics],
                dim=1,
            )
        )


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = ResidualBlock(input_channels + skip_channels, output_channels)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(values, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([values, skip], dim=1))


class MarsV4Model(nn.Module):
    """Shared temporal encoder with physics fusion and segmentation-derived presence.

    Unlike v3, v4 has no free high-capacity scene classifier.  Its scene score
    is a deterministic robust top-k aggregation of the segmentation logits, so
    a scene can be positive only when the dense detector produces plume evidence.
    """

    fused_channels = (48, 80, 128, 192, 256)

    def __init__(
        self,
        scene_topk_fraction: float = 0.005,
        scene_max_weight: float = 0.2,
    ) -> None:
        super().__init__()
        if not 0.0 < scene_topk_fraction <= 1.0:
            raise ValueError("scene_topk_fraction must be in (0, 1]")
        if not 0.0 <= scene_max_weight <= 1.0:
            raise ValueError("scene_max_weight must be in [0, 1]")
        self.scene_topk_fraction = scene_topk_fraction
        self.scene_max_weight = scene_max_weight
        self.encoder = SharedTemporalEncoder()
        self.fusions = nn.ModuleList(
            TemporalFusion(temporal, fused)
            for temporal, fused in zip(self.encoder.channels, self.fused_channels)
        )
        self.decoder4 = DecoderBlock(256, 192, 192)
        self.decoder3 = DecoderBlock(192, 128, 128)
        self.decoder2 = DecoderBlock(128, 80, 80)
        self.decoder1 = DecoderBlock(80, 48, 64)
        self.segmentation = nn.Conv2d(64, 1, 1)
        self.physics_prior = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )
        self.physics_prior_gain = nn.Parameter(torch.zeros(()))

    def forward(
        self, inputs: torch.Tensor, observable: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != len(INPUT_CHANNELS):
            raise ValueError(
                f"Expected Bx{len(INPUT_CHANNELS)}xHxW input, got {tuple(inputs.shape)}"
            )
        if observable.shape != inputs[:, :1].shape:
            raise ValueError("Observable mask must be Bx1xHxW and match input dimensions")
        target = self.encoder(inputs[:, TARGET_SLICE])
        reference = self.encoder(inputs[:, REFERENCE_SLICE])
        physics = inputs[:, PHYSICS_INDICES]
        fused = [
            fusion(target_value, reference_value, physics)
            for fusion, target_value, reference_value in zip(
                self.fusions, target, reference
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
        # A small max contribution preserves sensitivity to compact plumes while
        # the robust top-k mean prevents one hot pixel from declaring a scene.
        scene_logit = (
            (1.0 - self.scene_max_weight) * topk.mean(dim=1)
            + self.scene_max_weight * topk.max(dim=1).values
        )
        return {
            "segmentation_logits": segmentation_logits,
            "scene_logit": scene_logit,
            "dense_features": decoded,
        }

    def artifact_metadata(self) -> dict[str, Any]:
        percentage = 100.0 * self.scene_topk_fraction
        mean_weight = 1.0 - self.scene_max_weight
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "input_channels": list(INPUT_CHANNELS),
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "scene_score": (
                f"sigmoid({mean_weight:g} * mean(top {percentage:g}% observable segmentation "
                f"logits) + {self.scene_max_weight:g} * max(observable segmentation logits))"
            ),
            "scene_max_weight": self.scene_max_weight,
            "scene_topk_fraction": self.scene_topk_fraction,
            "temporal_weight_sharing": True,
            "initialization": "from_scratch",
        }
