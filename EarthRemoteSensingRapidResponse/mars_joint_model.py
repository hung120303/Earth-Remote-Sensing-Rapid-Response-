"""Physics-guided dual-temporal MARS-S2L segmentation/presence network."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAME = "ersrr_mars_joint_v1"
MODEL_SCHEMA_VERSION = 1
INPUT_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
        )
        self.projection = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values) + self.projection(values)


class SharedEncoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.stem = ConvBlock(len(INPUT_BANDS), channels[0])
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels[index], channels[index + 1], 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(_groups(channels[index + 1]), channels[index + 1]),
                    nn.GELU(),
                    ConvBlock(channels[index + 1], channels[index + 1]),
                )
                for index in range(len(channels) - 1)
            ]
        )

    def forward(self, values: torch.Tensor) -> list[torch.Tensor]:
        features = [self.stem(values)]
        for layer in self.down:
            features.append(layer(features[-1]))
        return features


class FusionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(4 * channels + 2, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
        )
        self.context = ConvBlock(channels, channels)

    def forward(
        self,
        target: torch.Tensor,
        reference: torch.Tensor,
        mbmp_score: torch.Tensor,
        observable: torch.Tensor,
    ) -> torch.Tensor:
        auxiliary = torch.cat(
            [
                F.interpolate(mbmp_score, size=target.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(observable, size=target.shape[-2:], mode="nearest"),
            ],
            dim=1,
        )
        combined = torch.cat(
            [target, reference, target - reference, torch.abs(target - reference), auxiliary],
            dim=1,
        )
        return self.context(self.reduce(combined))


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(input_channels + skip_channels, skip_channels)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(values, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([values, skip], dim=1))


class MarsJointModel(nn.Module):
    """Shared target/reference encoder with segmentation, presence, and quality heads."""

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.base_channels = base_channels
        self.encoder = SharedEncoder(channels)
        self.fusion = nn.ModuleList([FusionBlock(value) for value in channels])
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(channels[index], channels[index - 1])
                for index in range(len(channels) - 1, 0, -1)
            ]
        )
        self.segmentation = nn.Conv2d(channels[0], 1, 1)
        self.presence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(channels[-1] // 2, 1),
        )
        self.quality = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1] + 1, channels[-1] // 4),
            nn.GELU(),
            nn.Linear(channels[-1] // 4, 1),
        )

    def forward(
        self,
        target: torch.Tensor,
        reference: torch.Tensor,
        mbmp_score: torch.Tensor,
        observable: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_features = self.encoder(target)
        reference_features = self.encoder(reference)
        fused = [
            layer(target_value, reference_value, mbmp_score, observable)
            for layer, target_value, reference_value in zip(
                self.fusion, target_features, reference_features
            )
        ]
        decoded = fused[-1]
        for block, skip in zip(self.decoder, reversed(fused[:-1])):
            decoded = block(decoded, skip)
        deepest = fused[-1]
        observed_fraction = observable.mean(dim=(-2, -1))
        quality_features = torch.cat(
            [F.adaptive_avg_pool2d(deepest, 1).flatten(1), observed_fraction], dim=1
        )
        quality_logit = self.quality[2:](quality_features)
        return {
            "segmentation_logits": self.segmentation(decoded),
            "presence_logit": self.presence(deepest).squeeze(1),
            "quality_logit": quality_logit.squeeze(1),
        }

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "base_channels": self.base_channels,
            "input_bands": list(INPUT_BANDS),
            "heads": ["segmentation", "presence", "quality"],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }
