"""Standalone sensor-aware ordinal U-Net for the preregistered MARS pilot."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAME = "mars_sensor_aware_ordinal_unet"
MODEL_SCHEMA_VERSION = 1
INPUT_CHANNELS = (
    "target_B02", "target_B03", "target_B04", "target_B08", "target_B11", "target_B12",
    "reference_B02", "reference_B03", "reference_B04", "reference_B08", "reference_B11", "reference_B12",
    "radiometric_valid_mask", "cloud_indicator",
)
SENSORS = ("Sentinel-2", "Landsat")


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )


class SensorStem(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(14, 24, 3, padding=1, bias=False),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1, bias=False),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
        )


def monotone_cumulative_logits(raw: torch.Tensor) -> torch.Tensor:
    """Convert (a1,d2,d3,d4) to decreasing cumulative ordinal logits."""
    if raw.ndim != 4 or raw.shape[1] != 4:
        raise ValueError("Ordinal raw output must have shape Bx4xHxW")
    first = raw[:, :1]
    decrements = F.softplus(raw[:, 1:])
    return torch.cat((first, first - torch.cumsum(decrements, dim=1)), dim=1)


def _weighted_quantile(values: torch.Tensor, weights: torch.Tensor, quantile: float) -> torch.Tensor:
    """Batch weighted quantile with an all-zero branch mapped exactly to zero."""
    if values.shape != weights.shape or values.ndim != 3:
        raise ValueError("Weighted quantile fields must be matching BxHxW tensors")
    flat_values = values.flatten(1)
    flat_weights = weights.flatten(1)
    sorted_values, order = torch.sort(flat_values, dim=1)
    sorted_weights = torch.gather(flat_weights, 1, order)
    total = sorted_weights.sum(dim=1)
    target = total * float(quantile)
    cumulative = torch.cumsum(sorted_weights, dim=1)
    index = torch.searchsorted(cumulative.contiguous(), target[:, None].contiguous(), right=False)
    index = index.clamp_max(flat_values.shape[1] - 1)
    selected = torch.gather(sorted_values, 1, index).squeeze(1)
    return torch.where(total > 0, selected, torch.zeros_like(selected))


def pooled_field(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.shape != weights.shape or values.ndim != 3:
        raise ValueError("Pool fields must be matching BxHxW tensors")
    total = weights.flatten(1).sum(dim=1)
    mean = (values * weights).flatten(1).sum(dim=1) / total.clamp_min(1e-12)
    mean = torch.where(total > 0, mean, torch.zeros_like(mean))
    quantiles = [_weighted_quantile(values, weights, q) for q in (0.50, 0.75, 0.90, 0.97)]
    return torch.stack((mean, *quantiles), dim=1)


class MarsSensorOrdinalUNet(nn.Module):
    """Two sensor stems, one compact U-Net, monotone ordinal and isolated scene heads."""
    def __init__(self) -> None:
        super().__init__()
        self.sensor_stems = nn.ModuleList((SensorStem(), SensorStem()))
        self.enc1 = ConvBlock(24, 32, 8)
        self.enc2 = ConvBlock(32, 48, 8)
        self.enc3 = ConvBlock(48, 64, 8)
        self.bottleneck = ConvBlock(64, 96, 8)
        self.dec3 = ConvBlock(160, 64, 8)
        self.dec2 = ConvBlock(112, 48, 8)
        self.dec1 = ConvBlock(80, 32, 8)
        self.binary_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False), nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv2d(16, 1, 1)
        )
        self.ordinal_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False), nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv2d(16, 4, 1)
        )
        self.scene_projection = nn.Sequential(
            nn.Conv2d(96, 16, 1, bias=False), nn.GroupNorm(4, 16), nn.SiLU()
        )
        self.scene_mlp = nn.Sequential(nn.Linear(26, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1))

    @staticmethod
    def _upsample(values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(values, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def pixel_parameters(self) -> list[nn.Parameter]:
        scene_ids = {id(p) for module in (self.scene_projection, self.scene_mlp) for p in module.parameters()}
        return [p for p in self.parameters() if id(p) not in scene_ids]

    def scene_parameters(self) -> list[nn.Parameter]:
        return [p for module in (self.scene_projection, self.scene_mlp) for p in module.parameters()]

    def _stem(self, inputs: torch.Tensor, sensor_index: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 14:
            raise ValueError("Expected exact Bx14xHxW model input")
        if sensor_index.shape != (inputs.shape[0],) or not torch.all((sensor_index == 0) | (sensor_index == 1)):
            raise ValueError("Sensor indices must be a B-vector containing only 0/1")
        # Evaluate only the selected independent stem for each row; index_copy keeps gradients local.
        output = torch.empty((inputs.shape[0], 24, *inputs.shape[-2:]), device=inputs.device, dtype=inputs.dtype)
        for sensor, stem in enumerate(self.sensor_stems):
            rows = torch.nonzero(sensor_index == sensor, as_tuple=False).flatten()
            if rows.numel():
                output = output.index_copy(0, rows, stem(inputs.index_select(0, rows)))
        return output

    def forward(self, inputs: torch.Tensor, sensor_index: torch.Tensor, observable: torch.Tensor) -> dict[str, torch.Tensor]:
        if observable.shape != inputs[:, :1].shape:
            raise ValueError("Observable mask does not match model grid")
        original = inputs.shape[-2:]
        pad_h, pad_w = (-original[0]) % 8, (-original[1]) % 8
        inputs = F.pad(inputs, (0, pad_w, 0, pad_h))
        observable = F.pad(observable.float(), (0, pad_w, 0, pad_h))
        stem = self._stem(inputs, sensor_index)
        e1 = self.enc1(stem)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        bottleneck = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat((self._upsample(bottleneck, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self._upsample(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self._upsample(d2, e1), e1), dim=1))
        binary = self.binary_head(d1)[..., : original[0], : original[1]]
        ordinal = monotone_cumulative_logits(self.ordinal_head(d1))[..., : original[0], : original[1]]
        valid = observable[..., : original[0], : original[1]].squeeze(1)
        dense_probability = torch.sigmoid(binary).squeeze(1).detach()
        ordinal_probability = torch.sigmoid(ordinal).detach()
        dense_pool = pooled_field(dense_probability, valid * dense_probability)
        ordinal_mean = ordinal_probability.mean(dim=1)
        ordinal_pool = pooled_field(ordinal_mean, valid * ordinal_probability[:, 0])
        projected = self.scene_projection(bottleneck.detach())
        bottleneck_valid = F.interpolate(observable, size=projected.shape[-2:], mode="nearest")
        global_feature = (projected * bottleneck_valid).flatten(2).sum(dim=2) / bottleneck_valid.flatten(2).sum(dim=2).clamp_min(1.0)
        descriptor = torch.cat((dense_pool, ordinal_pool, global_feature), dim=1)
        scene_logit = self.scene_mlp(descriptor).squeeze(1)
        return {
            "binary_logit": binary,
            "ordinal_logits": ordinal,
            "scene_logit": scene_logit,
            "scene_descriptor": descriptor,
            "bottleneck": bottleneck,
        }

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "name": MODEL_NAME,
            "schema_version": MODEL_SCHEMA_VERSION,
            "input_channels": list(INPUT_CHANNELS),
            "sensors": list(SENSORS),
            "parameters": sum(p.numel() for p in self.parameters()),
            "pretrained": False,
            "scene_gradient_isolation": True,
        }


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def pixel_loss(output: dict[str, torch.Tensor], plume: torch.Tensor, observable: torch.Tensor, ordinal_level: torch.Tensor, ordinal_support: torch.Tensor) -> dict[str, torch.Tensor]:
    binary = output["binary_logit"].squeeze(1)
    target = plume.float()
    valid = observable.bool()
    bce = masked_mean(F.binary_cross_entropy_with_logits(binary, target, reduction="none"), valid)
    probability = torch.sigmoid(binary)
    intersection = (probability * target * valid).sum()
    denominator = ((probability + target) * valid).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    cumulative_target = torch.stack([(ordinal_level >= level).float() for level in range(1, 5)], dim=1)
    support = ordinal_support[:, None].expand_as(cumulative_target)
    ordinal = masked_mean(F.binary_cross_entropy_with_logits(output["ordinal_logits"], cumulative_target, reduction="none"), support)
    binary_loss = 0.7 * bce + 0.3 * dice
    total = binary_loss + 0.5 * ordinal
    return {"loss": total, "binary": binary_loss, "bce": bce, "dice": dice, "ordinal": ordinal}
