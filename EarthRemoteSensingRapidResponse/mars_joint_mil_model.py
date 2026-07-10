"""ERSRR MARS joint model v2 with multiple-instance plume presence pooling."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mars_joint_model import (
    INPUT_BANDS,
    MODEL_SCHEMA_VERSION,
    DecoderBlock,
    FusionBlock,
    SharedEncoder,
)

MODEL_NAME = "ersrr_mars_joint_mil_v2"


class MarsJointMILModel(nn.Module):
    """Dual-temporal U-Net with top-k segmentation evidence in the presence head."""

    def __init__(self, base_channels: int = 24, topk_fraction: float = 0.01) -> None:
        super().__init__()
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0,1]")
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.base_channels = base_channels
        self.topk_fraction = topk_fraction
        self.encoder = SharedEncoder(channels)
        self.fusion = nn.ModuleList([FusionBlock(value) for value in channels])
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(channels[index], channels[index - 1])
                for index in range(len(channels) - 1, 0, -1)
            ]
        )
        self.segmentation = nn.Conv2d(channels[0], 1, 1)
        presence_inputs = 2 * channels[-1] + 2
        self.presence = nn.Sequential(
            nn.Linear(presence_inputs, channels[-1]),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.GELU(),
            nn.Linear(channels[-1] // 2, 1),
        )
        self.quality = nn.Sequential(
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
        segmentation_logits = self.segmentation(decoded)
        deepest = fused[-1]
        average_features = F.adaptive_avg_pool2d(deepest, 1).flatten(1)
        maximum_features = F.adaptive_max_pool2d(deepest, 1).flatten(1)
        flat_segmentation = segmentation_logits.flatten(1)
        topk_count = max(1, int(flat_segmentation.shape[1] * self.topk_fraction))
        topk_values = torch.topk(flat_segmentation, k=topk_count, dim=1).values
        segmentation_evidence = torch.stack(
            [topk_values.mean(dim=1), topk_values.max(dim=1).values], dim=1
        )
        presence_features = torch.cat(
            [average_features, maximum_features, segmentation_evidence], dim=1
        )
        observed_fraction = observable.mean(dim=(-2, -1))
        quality_features = torch.cat([average_features, observed_fraction], dim=1)
        return {
            "segmentation_logits": segmentation_logits,
            "presence_logit": self.presence(presence_features).squeeze(1),
            "quality_logit": self.quality(quality_features).squeeze(1),
        }

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "base_channels": self.base_channels,
            "topk_fraction": self.topk_fraction,
            "input_bands": list(INPUT_BANDS),
            "heads": ["segmentation", "presence_mil", "quality"],
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }
