"""Physics-guided, identity-initialized adapters for the released MARS-S2L U-Net."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mars_paper_model import ReleasedMarsUNet, ResidualBlock, SENSOR_NAMES
from mars_v4_model import INPUT_CHANNELS, REFERENCE_SLICE, TARGET_SLICE


class PhysicsPyramid(nn.Module):
    channels = (32, 64, 96, 128, 160)

    def __init__(self) -> None:
        super().__init__()
        self.level1 = ResidualBlock(25, self.channels[0])
        self.level2 = ResidualBlock(self.channels[0], self.channels[1], stride=2)
        self.level3 = ResidualBlock(self.channels[1], self.channels[2], stride=2)
        self.level4 = ResidualBlock(self.channels[2], self.channels[3], stride=2)
        self.level5 = ResidualBlock(self.channels[3], self.channels[4], stride=2)

    @staticmethod
    def inputs(
        values: torch.Tensor, baseline_logits: torch.Tensor, sensor_index: torch.Tensor
    ) -> torch.Tensor:
        target = values[:, TARGET_SLICE]
        reference = values[:, REFERENCE_SLICE]
        difference = target - reference
        normalized = difference / (target.abs() + reference.abs() + 0.02)
        log_ratio = torch.log(target.clamp_min(0.0) + 0.01) - torch.log(
            reference.clamp_min(0.0) + 0.01
        )
        sensor = F.one_hot(sensor_index, num_classes=len(SENSOR_NAMES)).to(values.dtype)
        sensor = sensor[:, :, None, None].expand(-1, -1, values.shape[-2], values.shape[-1])
        return torch.cat(
            (
                values[:, 0:1] - 1.0,
                difference,
                normalized,
                log_ratio,
                values[:, 13:16],
                sensor,
                baseline_logits,
            ),
            dim=1,
        )

    def forward(
        self, values: torch.Tensor, baseline_logits: torch.Tensor, sensor_index: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        level1 = self.level1(self.inputs(values, baseline_logits, sensor_index))
        level2 = self.level2(level1)
        level3 = self.level3(level2)
        level4 = self.level4(level3)
        level5 = self.level5(level4)
        return level1, level2, level3, level4, level5


class IdentityFeatureGate(nn.Module):
    """Bounded feature modulation that is exactly identity at initialization."""

    def __init__(self, physics_channels: int, teacher_channels: int) -> None:
        super().__init__()
        self.scale = nn.Conv2d(physics_channels, teacher_channels, kernel_size=1)
        self.shift = nn.Conv2d(physics_channels, teacher_channels, kernel_size=1)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, teacher: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        if physics.shape[-2:] != teacher.shape[-2:]:
            physics = F.interpolate(physics, teacher.shape[-2:], mode="bilinear", align_corners=False)
        rms = teacher.detach().square().mean(dim=(-2, -1), keepdim=True).add(1e-6).sqrt()
        scale = 0.25 * torch.tanh(self.scale(physics))
        shift = 0.25 * rms * torch.tanh(self.shift(physics))
        return teacher * (1.0 + scale) + shift


class PhysicsGuidedTeacherAdapter(nn.Module):
    """Frozen released U-Net with trainable multi-scale methane-guided feature gates."""

    def __init__(self) -> None:
        super().__init__()
        self.teacher = ReleasedMarsUNet()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.physics = PhysicsPyramid()
        teacher_channels = (64, 128, 256, 512, 512)
        self.gates = nn.ModuleList(
            IdentityFeatureGate(source, target)
            for source, target in zip(self.physics.channels, teacher_channels)
        )
        self.backbone_trainable = False

    def load_released_checkpoint(self, state: dict[str, torch.Tensor]) -> None:
        incompatible = self.teacher.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            raise ValueError(f"Released checkpoint is missing keys: {incompatible.missing_keys}")
        if any(not key.startswith("out_mlp.") for key in incompatible.unexpected_keys):
            raise ValueError(f"Released checkpoint has unexpected keys: {incompatible.unexpected_keys}")

    def train(self, mode: bool = True) -> "PhysicsGuidedTeacherAdapter":
        super().train(mode)
        self.teacher.eval()
        return self

    def _encode(self, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        level1 = self.teacher.inc(values)
        level2 = self.teacher.down1(level1)
        level3 = self.teacher.down2(level2)
        level4 = self.teacher.down3(level3)
        level5 = self.teacher.down4(level4)
        return level1, level2, level3, level4, level5

    def _decode(self, levels: tuple[torch.Tensor, ...]) -> torch.Tensor:
        level1, level2, level3, level4, level5 = levels
        decoded = self.teacher.up1(level5, level4)
        decoded = self.teacher.up2(decoded, level3)
        decoded = self.teacher.up3(decoded, level2)
        decoded = self.teacher.up4(decoded, level1)
        return self.teacher.out(decoded)

    @staticmethod
    def scene_surrogate(logits: torch.Tensor, observable: torch.Tensor) -> torch.Tensor:
        flat = logits.flatten(1)
        valid = observable.flatten(1) > 0.5
        masked = flat.masked_fill(~valid, -20.0)
        k = min(100, masked.shape[1])
        global_top100 = torch.topk(masked, k=k, dim=1).values.mean(dim=1)
        local_values = logits.masked_fill(observable <= 0.5, -20.0)
        local_mean = F.avg_pool2d(local_values, kernel_size=10, stride=2)
        local_valid = F.avg_pool2d(observable, kernel_size=10, stride=2)
        local_mean = local_mean.masked_fill(local_valid < 0.9, -20.0)
        local100 = local_mean.flatten(1).amax(dim=1)
        return 0.5 * global_top100 + 0.5 * local100

    def forward(
        self, values: torch.Tensor, observable: torch.Tensor, sensor_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if values.ndim != 4 or values.shape[1] != len(INPUT_CHANNELS):
            raise ValueError(f"Expected Bx{len(INPUT_CHANNELS)}xHxW input")
        with torch.no_grad():
            teacher_levels = self._encode(values)
            baseline_logits = self._decode(teacher_levels)
        physics_levels = self.physics(values, baseline_logits.detach(), sensor_index)
        adapted = tuple(
            gate(teacher.detach(), physics)
            for gate, teacher, physics in zip(self.gates, teacher_levels, physics_levels)
        )
        logits = self._decode(adapted)
        return {
            "segmentation_logits": logits,
            "baseline_logits": baseline_logits,
            "correction_logits": logits - baseline_logits,
            "scene_logit": self.scene_surrogate(logits, observable),
        }

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("teacher.")
        }

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
