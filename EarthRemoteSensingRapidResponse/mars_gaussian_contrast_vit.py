"""Physics-contrast ViT-U-Net for Gaussian-pretrained MARS plume evidence."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mars_gaussian_vit import ConvBlock, UpBlock
from mars_v4_model import INPUT_CHANNELS


MODEL_NAME = "mars_gaussian_contrast_vit_unet"
MODEL_SCHEMA_VERSION = 1
SPECTRAL_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
RATIO_PAIRS = tuple(
    (left, right)
    for left in range(len(SPECTRAL_BANDS))
    for right in range(left + 1, len(SPECTRAL_BANDS))
)
CONTRAST_CHANNELS = (
    "mbmp_scaled",
    *(f"log_change_{band}" for band in SPECTRAL_BANDS),
    *(f"log_ratio_change_{SPECTRAL_BANDS[left]}_{SPECTRAL_BANDS[right]}" for left, right in RATIO_PAIRS),
    "wind_u",
    "wind_v",
    "cloud",
    "observable",
)


def methane_contrast_features(
    inputs: torch.Tensor,
    observable: torch.Tensor,
    *,
    mbmp_scale: float = 0.02,
    log_change_scale: float = 0.05,
    clip: float = 8.0,
) -> torch.Tensor:
    """Derive per-scene temporal contrasts without cohort/test statistics."""

    if inputs.ndim != 4 or inputs.shape[1] != len(INPUT_CHANNELS):
        raise ValueError(f"Expected Bx{len(INPUT_CHANNELS)}xHxW MARS input")
    if observable.shape != inputs[:, :1].shape:
        raise ValueError("Observable mask does not match the input grid")
    target = inputs[:, 1:7].float().clamp_min(1e-3)
    reference = inputs[:, 7:13].float().clamp_min(1e-3)
    log_target = torch.log(target)
    log_reference = torch.log(reference)
    log_change = (log_target - log_reference) / float(log_change_scale)
    ratio_change = torch.cat(
        [
            (
                (log_target[:, left : left + 1] - log_target[:, right : right + 1])
                - (log_reference[:, left : left + 1] - log_reference[:, right : right + 1])
            )
            / float(log_change_scale)
            for left, right in RATIO_PAIRS
        ],
        dim=1,
    )
    spectral = torch.cat(
        (inputs[:, 0:1].float() / float(mbmp_scale), log_change, ratio_change), dim=1
    ).clamp(-float(clip), float(clip))
    spectral = spectral * observable.float()
    context = torch.cat(
        (
            inputs[:, 13:15].float().clamp(-2.0, 2.0),
            inputs[:, 15:16].float().clamp(0.0, 1.0),
            observable.float(),
        ),
        dim=1,
    )
    return torch.cat((spectral, context), dim=1)


class GaussianContrastViTUNet(nn.Module):
    """Mixed-sensor ViT encoder over normalized temporal physics contrasts."""

    def __init__(
        self,
        *,
        dimension: int = 256,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.05,
        patch_size: int = 16,
        reference_grid: int = 13,
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("Transformer dimension must be divisible by head count")
        self.dimension = int(dimension)
        self.depth = int(depth)
        self.heads = int(heads)
        self.patch_size = int(patch_size)
        self.reference_grid = int(reference_grid)
        combined_channels = len(INPUT_CHANNELS) + len(CONTRAST_CHANNELS)
        self.stem0 = ConvBlock(combined_channels, 48)
        self.stem1 = ConvBlock(48, 72, stride=2)
        self.stem2 = ConvBlock(72, 112, stride=2)
        self.stem3 = ConvBlock(112, 160, stride=2)
        self.patch_embedding = nn.Conv2d(
            len(CONTRAST_CHANNELS), dimension, patch_size, stride=patch_size
        )
        self.position = nn.Parameter(torch.zeros(1, dimension, reference_grid, reference_grid))
        self.sensor_embedding = nn.Embedding(2, dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=dimension * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, norm=nn.LayerNorm(dimension)
        )
        self.up1 = UpBlock(dimension, 160, 160)
        self.up2 = UpBlock(160, 112, 112)
        self.up3 = UpBlock(112, 72, 72)
        self.up4 = UpBlock(72, 48, 48)
        self.segmentation_head = nn.Sequential(ConvBlock(48, 48), nn.Conv2d(48, 1, 1))
        self.scene_head = nn.Sequential(
            nn.LayerNorm(dimension * 2 + 2),
            nn.Linear(dimension * 2 + 2, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 1),
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.sensor_embedding.weight, std=0.02)

    def _pad(self, values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = values.shape[-2:]
        pad_height = (-height) % self.patch_size
        pad_width = (-width) % self.patch_size
        return F.pad(values, (0, pad_width, 0, pad_height), mode="reflect"), (height, width)

    @staticmethod
    def _top_fraction(values: torch.Tensor, observable: torch.Tensor, fraction: float = 0.01) -> torch.Tensor:
        flattened = values.flatten(1)
        valid = observable.flatten(1) > 0.5
        flattened = flattened.masked_fill(~valid, torch.finfo(flattened.dtype).min)
        count = max(1, int(math.ceil(flattened.shape[1] * fraction)))
        selected = torch.topk(flattened, k=count, dim=1).values
        return torch.where(torch.isfinite(selected), selected, torch.zeros_like(selected)).mean(dim=1)

    def forward(
        self,
        inputs: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        contrast = methane_contrast_features(inputs, observable)
        combined = torch.cat((inputs.float(), contrast), dim=1)
        padded, original_shape = self._pad(combined)
        padded_contrast = F.pad(
            contrast,
            (0, padded.shape[-1] - original_shape[1], 0, padded.shape[-2] - original_shape[0]),
            mode="reflect",
        )
        stem0 = self.stem0(padded)
        stem1 = self.stem1(stem0)
        stem2 = self.stem2(stem1)
        stem3 = self.stem3(stem2)
        embedded = self.patch_embedding(padded_contrast)
        grid = embedded.shape[-2:]
        embedded = embedded + F.interpolate(
            self.position, size=grid, mode="bicubic", align_corners=False
        )
        tokens = embedded.flatten(2).transpose(1, 2)
        tokens = tokens + self.sensor_embedding(sensor_index)[:, None]
        tokens = self.transformer(tokens)
        encoded = tokens.transpose(1, 2).reshape(inputs.shape[0], self.dimension, *grid)
        decoded = self.up1(encoded, stem3)
        decoded = self.up2(decoded, stem2)
        decoded = self.up3(decoded, stem1)
        decoded = self.up4(decoded, stem0)
        logits = self.segmentation_head(decoded)[..., : original_shape[0], : original_shape[1]]
        mean_token = tokens.mean(dim=1)
        max_token = tokens.max(dim=1).values
        top_evidence = self._top_fraction(logits * observable, observable)
        valid_fraction = observable.flatten(1).mean(dim=1)
        scene_logit = self.scene_head(
            torch.cat((mean_token, max_token, top_evidence[:, None], valid_fraction[:, None]), dim=1)
        )[:, 0]
        return {
            "segmentation_logits": logits,
            "scene_logit": scene_logit,
            "top_evidence": top_evidence,
            "contrast_features": contrast,
        }

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "name": MODEL_NAME,
            "schema_version": MODEL_SCHEMA_VERSION,
            "input_channels": list(INPUT_CHANNELS),
            "derived_contrast_channels": list(CONTRAST_CHANNELS),
            "patch_size": self.patch_size,
            "dimension": self.dimension,
            "depth": self.depth,
            "heads": self.heads,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "normalization": "per-pixel fixed physical scales; no cohort/test statistics",
        }
