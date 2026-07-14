"""Baseline-preserving mixed-sensor successor to the released MARS-S2L U-Net."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mars_v4_model import INPUT_CHANNELS, REFERENCE_SLICE, TARGET_SLICE

MODEL_NAME = "ersrr_mars_paper_residual_v1"
MODEL_SCHEMA_VERSION = 1
RELEASED_CHECKPOINT_SHA256 = (
    "be634fb9e24dc4877f44c1ff9f69972e6f0453e30d70c0dc03677876340ef246"
)
SENSOR_NAMES = ("Sentinel-2", "Landsat")


def _groups(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def double_conv(input_channels: int, output_channels: int) -> nn.Sequential:
    """Exact convolution block used by upstream ``UnetOriginal``."""
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.GELU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.GELU(),
    )


class ReleasedUp(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = double_conv(input_channels, output_channels)

    def forward(self, deep: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        deep = self.up(deep)
        difference_y = skip.size(2) - deep.size(2)
        difference_x = skip.size(3) - deep.size(3)
        deep = F.pad(
            deep,
            [
                difference_x // 2,
                difference_x - difference_x // 2,
                difference_y // 2,
                difference_y - difference_y // 2,
            ],
        )
        return self.conv(torch.cat([skip, deep], dim=1))


class ReleasedMarsUNet(nn.Module):
    """Forward-equivalent form of the paper's released ``UnetOriginal``."""

    def __init__(self, input_channels: int = len(INPUT_CHANNELS)) -> None:
        super().__init__()
        self.inc = double_conv(input_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), double_conv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), double_conv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), double_conv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), double_conv(512, 512))
        self.up1 = ReleasedUp(1024, 256)
        self.up2 = ReleasedUp(512, 128)
        self.up3 = ReleasedUp(256, 64)
        self.up4 = ReleasedUp(128, 128)
        self.out = nn.Conv2d(128, 1, kernel_size=1, stride=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        level1 = self.inc(values)
        level2 = self.down1(level1)
        level3 = self.down2(level2)
        level4 = self.down3(level3)
        level5 = self.down4(level4)
        decoded = self.up1(level5, level4)
        decoded = self.up2(decoded, level3)
        decoded = self.up3(decoded, level2)
        decoded = self.up4(decoded, level1)
        return self.out(decoded)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.block(values) + self.skip(values))


class ResidualDecoder(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = ResidualBlock(input_channels + skip_channels, output_channels)

    def forward(self, deep: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        deep = F.interpolate(
            deep, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.block(torch.cat([deep, skip], dim=1))


class TemporalPhysicsCorrection(nn.Module):
    """Learn a sensor-aware logit correction from explicit temporal physics."""

    channels = (32, 64, 112, 160)

    def __init__(self) -> None:
        super().__init__()
        # MBMP, six raw differences, six normalized differences, six log
        # ratios, wind u/v, cloud, two sensor indicators, and baseline logit.
        self.input_channels = 25
        self.level1 = ResidualBlock(self.input_channels, self.channels[0])
        self.level2 = ResidualBlock(self.channels[0], self.channels[1], stride=2)
        self.level3 = ResidualBlock(self.channels[1], self.channels[2], stride=2)
        self.level4 = ResidualBlock(self.channels[2], self.channels[3], stride=2)
        self.decode3 = ResidualDecoder(self.channels[3], self.channels[2], 112)
        self.decode2 = ResidualDecoder(112, self.channels[1], 64)
        self.decode1 = ResidualDecoder(64, self.channels[0], 32)
        self.output = nn.Conv2d(32, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def features(
        self,
        inputs: torch.Tensor,
        baseline_logits: torch.Tensor,
        sensor_index: torch.Tensor,
    ) -> torch.Tensor:
        target = inputs[:, TARGET_SLICE]
        reference = inputs[:, REFERENCE_SLICE]
        difference = target - reference
        normalized_difference = difference / (target.abs() + reference.abs() + 0.02)
        log_ratio = torch.log(target.clamp_min(0.0) + 0.01) - torch.log(
            reference.clamp_min(0.0) + 0.01
        )
        sensor = F.one_hot(sensor_index, num_classes=len(SENSOR_NAMES)).to(
            dtype=inputs.dtype
        )
        sensor = sensor[:, :, None, None].expand(
            -1, -1, inputs.shape[-2], inputs.shape[-1]
        )
        return torch.cat(
            [
                inputs[:, 0:1] - 1.0,
                difference,
                normalized_difference,
                log_ratio,
                inputs[:, 13:16],
                sensor,
                baseline_logits,
            ],
            dim=1,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        baseline_logits: torch.Tensor,
        sensor_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.features(inputs, baseline_logits, sensor_index)
        level1 = self.level1(values)
        level2 = self.level2(level1)
        level3 = self.level3(level2)
        level4 = self.level4(level3)
        decoded = self.decode3(level4, level3)
        decoded = self.decode2(decoded, level2)
        decoded = self.decode1(decoded, level1)
        return self.output(decoded), decoded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def released_state(path: Path) -> dict[str, torch.Tensor]:
    if _sha256(path) != RELEASED_CHECKPOINT_SHA256:
        raise ValueError("Released MARS-S2L checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    original = payload["model_state_dict"]
    return {
        key.removeprefix("_orig_mod.module.").removeprefix("module."): value
        for key, value in original.items()
    }


class MarsPaperResidualModel(nn.Module):
    """Released MARS-S2L plus a zero-initialized temporal-physics correction."""

    def __init__(self, *, scene_topk_fraction: float = 0.01) -> None:
        super().__init__()
        if not 0.0 < scene_topk_fraction <= 1.0:
            raise ValueError("scene_topk_fraction must be in (0,1]")
        self.scene_topk_fraction = scene_topk_fraction
        self.backbone = ReleasedMarsUNet()
        self.correction = TemporalPhysicsCorrection()
        self.sensor_log_scale = nn.Parameter(torch.zeros(len(SENSOR_NAMES)))
        self.sensor_bias = nn.Parameter(torch.zeros(len(SENSOR_NAMES)))
        self.backbone_trainable = False
        self.set_backbone_trainable(False)

    def load_released_checkpoint(self, path: Path) -> None:
        incompatible = self.backbone.load_state_dict(released_state(path), strict=False)
        if incompatible.missing_keys:
            raise ValueError(f"Released checkpoint is missing keys: {incompatible.missing_keys}")
        if any(not key.startswith("out_mlp.") for key in incompatible.unexpected_keys):
            raise ValueError(
                f"Released checkpoint has unknown surplus keys: {incompatible.unexpected_keys}"
            )

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone_trainable = bool(trainable)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = self.backbone_trainable
        if not self.backbone_trainable:
            self.backbone.eval()

    def train(self, mode: bool = True) -> "MarsPaperResidualModel":
        super().train(mode)
        if not self.backbone_trainable:
            self.backbone.eval()
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != len(INPUT_CHANNELS):
            raise ValueError(
                f"Expected Bx{len(INPUT_CHANNELS)}xHxW input, got {tuple(inputs.shape)}"
            )
        if observable.shape != inputs[:, :1].shape:
            raise ValueError("Observable mask must be Bx1xHxW")
        if sensor_index.shape != (inputs.shape[0],):
            raise ValueError("Sensor index must have shape B")
        if torch.any((sensor_index < 0) | (sensor_index >= len(SENSOR_NAMES))):
            raise ValueError("Sensor index is outside the frozen mixed-sensor contract")

        baseline_logits = self.backbone(inputs)
        # Keep the calibration path in the released logit's dtype.  Under CUDA
        # autocast the backbone emits float16 logits while these parameters are
        # stored as float32.  Allowing ordinary type promotion here would make
        # the mathematically identity initialization numerically different from
        # the released model after sigmoid/thresholding.
        scale = self.sensor_log_scale[sensor_index].exp().to(
            baseline_logits.dtype
        )[:, None, None, None]
        bias = self.sensor_bias[sensor_index].to(baseline_logits.dtype)[
            :, None, None, None
        ]
        correction, dense = self.correction(
            inputs, baseline_logits.detach(), sensor_index
        )
        logits = baseline_logits * scale + bias + correction

        flat = logits.flatten(1)
        valid = observable.flatten(1) > 0.5
        masked = flat.masked_fill(~valid, -1e4)
        topk_count = max(1, int(masked.shape[1] * self.scene_topk_fraction))
        scene_logit = torch.topk(masked, k=topk_count, dim=1).values.mean(dim=1)
        return {
            "segmentation_logits": logits,
            "baseline_logits": baseline_logits,
            "correction_logits": correction,
            "scene_logit": scene_logit,
            "dense_features": dense,
        }

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "input_channels": list(INPUT_CHANNELS),
            "sensor_names": list(SENSOR_NAMES),
            "released_checkpoint_sha256": RELEASED_CHECKPOINT_SHA256,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameter_count_correction_only": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "initial_equivalence": "exact released U-Net logits before correction training",
            "physics_features": [
                "centered MBMP",
                "target-reference reflectance",
                "normalized temporal difference",
                "log temporal ratio",
                "wind",
                "cloud",
                "sensor identity",
                "released baseline logit",
            ],
            "scene_topk_fraction": self.scene_topk_fraction,
        }
