"""Full-resolution ERSRR v3 methane segmentation and proposal-presence model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAME = "ersrr_mars_full_unet_proposal_v3"
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
WIND_U_INDEX = 13
WIND_V_INDEX = 14
CLOUD_INDEX = 15


def _groups(channels: int) -> int:
    for value in (16, 8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class DoubleConv(nn.Module):
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

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class Down(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(input_channels, output_channels))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class Up(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = DoubleConv(input_channels, output_channels)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(values, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.block(torch.cat([skip, values], dim=1))


def soft_plume_geometry(
    segmentation_logits: torch.Tensor,
    observable: torch.Tensor,
    wind: torch.Tensor,
) -> torch.Tensor:
    """Differentiable area, shape, and wind-alignment descriptors.

    Image rows increase southward, so the normalized y coordinate is reversed to
    align the descriptor with north-positive wind-v metadata.
    """
    probability = torch.sigmoid(segmentation_logits) * observable
    batch, _, height, width = probability.shape
    dtype = probability.dtype
    device = probability.device
    x_axis = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
    y_axis = torch.linspace(1.0, -1.0, height, dtype=dtype, device=device)
    y_grid, x_grid = torch.meshgrid(y_axis, x_axis, indexing="ij")
    x_grid = x_grid.view(1, 1, height, width)
    y_grid = y_grid.view(1, 1, height, width)
    mass = probability.sum(dim=(-2, -1)).clamp_min(1e-6)
    observable_area = observable.sum(dim=(-2, -1)).clamp_min(1.0)
    centroid_x = (probability * x_grid).sum(dim=(-2, -1)) / mass
    centroid_y = (probability * y_grid).sum(dim=(-2, -1)) / mass
    delta_x = x_grid - centroid_x[:, :, None, None]
    delta_y = y_grid - centroid_y[:, :, None, None]
    variance_x = (probability * delta_x.square()).sum(dim=(-2, -1)) / mass
    variance_y = (probability * delta_y.square()).sum(dim=(-2, -1)) / mass
    covariance_xy = (probability * delta_x * delta_y).sum(dim=(-2, -1)) / mass

    wind_norm = wind.norm(dim=1, keepdim=True).clamp_min(1e-4)
    unit_x = wind[:, :1] / wind_norm
    unit_y = wind[:, 1:] / wind_norm
    parallel_variance = (
        unit_x.square() * variance_x
        + 2.0 * unit_x * unit_y * covariance_xy
        + unit_y.square() * variance_y
    )
    perpendicular_variance = (
        unit_y.square() * variance_x
        - 2.0 * unit_x * unit_y * covariance_xy
        + unit_x.square() * variance_y
    )
    wind_elongation = torch.log(
        (parallel_variance + 1e-4) / (perpendicular_variance + 1e-4)
    )
    total_variation = 0.5 * (
        torch.abs(probability[:, :, 1:] - probability[:, :, :-1]).mean(dim=(-2, -1))
        + torch.abs(probability[:, :, :, 1:] - probability[:, :, :, :-1]).mean(
            dim=(-2, -1)
        )
    )
    compactness_proxy = (mass / observable_area) / (
        torch.sqrt(variance_x + variance_y + 1e-4)
    )
    return torch.cat(
        [
            mass / observable_area,
            centroid_x,
            centroid_y,
            variance_x,
            variance_y,
            covariance_xy,
            parallel_variance,
            perpendicular_variance,
            wind_elongation,
            total_variation,
            compactness_proxy,
            wind_norm,
        ],
        dim=1,
    ).reshape(batch, 12)


class MarsV3Model(nn.Module):
    """Full U-Net with segmentation, proposal-aware presence, and quality heads.

    This model must be trained from scratch on the frozen ERSRR fit split. The
    released MARS-S2L weights are a baseline only because those weights already
    saw the official-train samples used by ERSRR internal validation.
    """

    def __init__(self, topk_fraction: float = 0.01) -> None:
        super().__init__()
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0,1]")
        self.topk_fraction = topk_fraction
        self.inc = DoubleConv(len(INPUT_CHANNELS), 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 128)
        self.segmentation = nn.Conv2d(128, 1, kernel_size=1)
        self.component_embedding = nn.Sequential(nn.Conv2d(128, 16, 1), nn.GELU())

        # Deep global context (1024), high-evidence component context (32),
        # segmentation top-k evidence (2), and soft geometry/wind (12).
        descriptor_channels = 1024 + 32 + 2 + 12
        self.presence = nn.Sequential(
            nn.LayerNorm(descriptor_channels),
            nn.Linear(descriptor_channels, 512),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )
        self.quality = nn.Sequential(
            nn.LayerNorm(514),
            nn.Linear(514, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        observable: torch.Tensor,
        *,
        return_dense_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != len(INPUT_CHANNELS):
            raise ValueError(
                f"Expected Bx{len(INPUT_CHANNELS)}xHxW input, got {tuple(inputs.shape)}"
            )
        if observable.shape != inputs[:, :1].shape:
            raise ValueError("Observable mask must be Bx1xHxW and match input spatial dimensions")
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        deepest = self.down4(x4)
        decoded = self.up1(deepest, x4)
        decoded = self.up2(decoded, x3)
        decoded = self.up3(decoded, x2)
        decoded = self.up4(decoded, x1)
        segmentation_logits = self.segmentation(decoded)
        component_features = self.component_embedding(decoded)

        flat_logits = segmentation_logits.flatten(1)
        flat_observable = observable.flatten(1) > 0.5
        masked_logits = flat_logits.masked_fill(~flat_observable, -1e4)
        topk_count = max(1, int(masked_logits.shape[1] * self.topk_fraction))
        topk = torch.topk(masked_logits, k=topk_count, dim=1)
        flat_components = component_features.flatten(2)
        gather_index = topk.indices[:, None, :].expand(-1, flat_components.shape[1], -1)
        proposal_values = torch.gather(flat_components, 2, gather_index)
        proposal_context = torch.cat(
            [proposal_values.mean(dim=2), proposal_values.max(dim=2).values], dim=1
        )
        global_context = torch.cat(
            [
                F.adaptive_avg_pool2d(deepest, 1).flatten(1),
                F.adaptive_max_pool2d(deepest, 1).flatten(1),
            ],
            dim=1,
        )
        wind = torch.stack(
            [inputs[:, WIND_U_INDEX].mean(dim=(-2, -1)), inputs[:, WIND_V_INDEX].mean(dim=(-2, -1))],
            dim=1,
        )
        geometry = soft_plume_geometry(segmentation_logits, observable, wind)
        segmentation_evidence = torch.stack(
            [topk.values.mean(dim=1), topk.values.max(dim=1).values], dim=1
        )
        proposal_descriptor = torch.cat(
            [global_context, proposal_context, segmentation_evidence, geometry], dim=1
        )
        observed_fraction = observable.mean(dim=(-2, -1))
        cloud_fraction = inputs[:, CLOUD_INDEX : CLOUD_INDEX + 1].mean(dim=(-2, -1))
        quality_descriptor = torch.cat(
            [F.adaptive_avg_pool2d(deepest, 1).flatten(1), observed_fraction, cloud_fraction],
            dim=1,
        )
        result = {
            "segmentation_logits": segmentation_logits,
            "presence_logit": self.presence(proposal_descriptor).squeeze(1),
            "quality_logit": self.quality(quality_descriptor).squeeze(1),
            "proposal_descriptor": proposal_descriptor,
            "soft_geometry": geometry,
        }
        if return_dense_features:
            result["component_features"] = component_features
            result["deepest_features"] = deepest
        return result

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "input_channels": list(INPUT_CHANNELS),
            "topk_fraction": self.topk_fraction,
            "heads": ["segmentation", "proposal_presence", "quality"],
            "normalization": "GroupNorm",
            "initialization": "from_scratch_required_for_primary_experiment",
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }
