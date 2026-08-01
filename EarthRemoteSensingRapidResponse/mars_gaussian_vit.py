"""Full-scene ViT-U-Net for Gaussian-pretrained mixed-sensor plume evidence."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mars_v4_model import INPUT_CHANNELS


MODEL_NAME = "mars_gaussian_pretrained_vit_unet"
MODEL_SCHEMA_VERSION = 1


def _groups(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values) + self.skip(values)


class UpBlock(nn.Module):
    def __init__(
        self, input_channels: int, skip_channels: int, output_channels: int
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            input_channels, output_channels, kernel_size=2, stride=2
        )
        self.fuse = ConvBlock(output_channels + skip_channels, output_channels)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = self.up(values)
        if values.shape[-2:] != skip.shape[-2:]:
            values = F.interpolate(
                values, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.fuse(torch.cat((values, skip), dim=1))


class GaussianPretrainedViTUNet(nn.Module):
    """ViT-B-inspired temporal encoder with convolutional dense decoder.

    The released 16-channel MARS contract is kept intact. Inputs are padded to
    a multiple of the 16-pixel patch size and cropped back exactly, allowing
    efficient 160-pixel training crops and native 200-pixel inference.
    """

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
        self.stem0 = ConvBlock(len(INPUT_CHANNELS), 48)
        self.stem1 = ConvBlock(48, 72, stride=2)
        self.stem2 = ConvBlock(72, 112, stride=2)
        self.stem3 = ConvBlock(112, 160, stride=2)
        self.patch_embedding = nn.Conv2d(
            len(INPUT_CHANNELS), dimension, patch_size, stride=patch_size
        )
        self.position = nn.Parameter(
            torch.zeros(1, dimension, reference_grid, reference_grid)
        )
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
        self.segmentation_head = nn.Sequential(
            ConvBlock(48, 48),
            nn.Conv2d(48, 1, 1),
        )
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
    def _top_fraction(
        values: torch.Tensor, observable: torch.Tensor, fraction: float = 0.01
    ) -> torch.Tensor:
        flattened = values.flatten(1)
        valid = observable.flatten(1) > 0.5
        minimum = torch.finfo(flattened.dtype).min
        flattened = flattened.masked_fill(~valid, minimum)
        count = max(1, int(math.ceil(flattened.shape[1] * fraction)))
        selected = torch.topk(flattened, k=count, dim=1).values
        selected = torch.where(torch.isfinite(selected), selected, torch.zeros_like(selected))
        return selected.mean(dim=1)

    def forward(
        self,
        inputs: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != len(INPUT_CHANNELS):
            raise ValueError(
                f"Expected Bx{len(INPUT_CHANNELS)}xHxW, got {tuple(inputs.shape)}"
            )
        if observable.shape != inputs[:, :1].shape:
            raise ValueError("Observable mask does not match input grid")
        padded, original_shape = self._pad(inputs)
        padded_observable = F.pad(
            observable,
            (0, padded.shape[-1] - original_shape[1], 0, padded.shape[-2] - original_shape[0]),
            value=0.0,
        )
        stem0 = self.stem0(padded)
        stem1 = self.stem1(stem0)
        stem2 = self.stem2(stem1)
        stem3 = self.stem3(stem2)
        embedded = self.patch_embedding(padded)
        grid = embedded.shape[-2:]
        position = F.interpolate(
            self.position, size=grid, mode="bicubic", align_corners=False
        )
        embedded = embedded + position
        tokens = embedded.flatten(2).transpose(1, 2)
        tokens = tokens + self.sensor_embedding(sensor_index)[:, None]
        tokens = self.transformer(tokens)
        encoded = tokens.transpose(1, 2).reshape(
            inputs.shape[0], self.dimension, *grid
        )
        decoded = self.up1(encoded, stem3)
        decoded = self.up2(decoded, stem2)
        decoded = self.up3(decoded, stem1)
        decoded = self.up4(decoded, stem0)
        logits = self.segmentation_head(decoded)
        logits = logits[..., : original_shape[0], : original_shape[1]]
        masked_logits = logits * observable
        mean_token = tokens.mean(dim=1)
        max_token = tokens.max(dim=1).values
        top_evidence = self._top_fraction(masked_logits, observable)
        valid_fraction = observable.flatten(1).mean(dim=1)
        scene_logit = self.scene_head(
            torch.cat((mean_token, max_token, top_evidence[:, None], valid_fraction[:, None]), dim=1)
        )[:, 0]
        return {
            "segmentation_logits": logits,
            "scene_logit": scene_logit,
            "top_evidence": top_evidence,
            "padded_observable": padded_observable,
        }

    @staticmethod
    def fuse_scene_score(
        baseline_score: torch.Tensor,
        scene_logit: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        baseline = torch.logit(baseline_score.clamp(1e-6, 1 - 1e-6))
        correction = 2.0 * torch.tanh(scene_logit / 2.0)
        return torch.sigmoid(baseline + float(strength) * correction)

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "name": MODEL_NAME,
            "schema_version": MODEL_SCHEMA_VERSION,
            "input_channels": list(INPUT_CHANNELS),
            "patch_size": self.patch_size,
            "dimension": self.dimension,
            "depth": self.depth,
            "heads": self.heads,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "training_grid": "160x160 at 10 m",
            "inference_grid": "native 200x200 at 10 m",
            "scene_evidence": "global mean/max ViT tokens plus top-1%-pixel evidence",
        }
