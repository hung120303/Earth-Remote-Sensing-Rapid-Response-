"""Physics-guided bi-temporal fusion around the released MARS-S2L U-Net.

The trainable branch uses one shared spectral encoder for the target and
reference dates, then fuses their signed/absolute changes with an explicit
methane guide.  The guide includes NDMI, MBMP, normalized band changes, and
log-ratios.  Zero-initialized feature gates make the dense output exactly equal
to the released model before training, while an independent patch/MIL head can
learn scene ranking without forcing scene truth through a single mask summary.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mars_paper_model import ReleasedMarsUNet, ResidualBlock, SENSOR_NAMES
from mars_physics_guided_teacher import IdentityFeatureGate
from mars_v4_model import INPUT_CHANNELS, REFERENCE_SLICE, TARGET_SLICE


MODEL_NAME = "ersrr_ndmi_bitemporal_fusion_v1"
MODEL_SCHEMA_VERSION = 1
PATCH_GRID_SIZE = 8
GUIDE_CHANNELS = 23
SCENE_EVIDENCE_BOUND = 2.0


class SharedSpectralEncoder(nn.Module):
    """Encode both dates with exactly the same parameters."""

    channels = (32, 48, 64, 96, 128)

    def __init__(self) -> None:
        super().__init__()
        self.stem = ResidualBlock(6, self.channels[0])
        self.stages = nn.ModuleList(
            ResidualBlock(source, target, stride=2)
            for source, target in zip(self.channels[:-1], self.channels[1:])
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        levels = [self.stem(values)]
        for stage in self.stages:
            levels.append(stage(levels[-1]))
        return tuple(levels)


class MethaneChangeGuide(nn.Module):
    """Build a multi-scale guide from interpretable spectral-change maps."""

    channels = (24, 32, 48, 64, 96)

    def __init__(self) -> None:
        super().__init__()
        self.level1 = ResidualBlock(GUIDE_CHANNELS, self.channels[0])
        self.stages = nn.ModuleList(
            ResidualBlock(source, target, stride=2)
            for source, target in zip(self.channels[:-1], self.channels[1:])
        )

    @staticmethod
    def inputs(
        values: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
    ) -> torch.Tensor:
        target = values[:, TARGET_SLICE]
        reference = values[:, REFERENCE_SLICE]
        difference = target - reference
        normalized = difference / (target.abs() + reference.abs() + 0.02)
        log_ratio = torch.log(target.clamp_min(0.0) + 0.01) - torch.log(
            reference.clamp_min(0.0) + 0.01
        )

        # Within each six-band date, B11 and B12 are positions four and five.
        # This follows the N-BPMSNet sign convention (B11-B12)/(B11+B12).
        target_ndmi = (target[:, 4:5] - target[:, 5:6]) / (
            target[:, 4:5].abs() + target[:, 5:6].abs() + 0.02
        )
        reference_ndmi = (reference[:, 4:5] - reference[:, 5:6]) / (
            reference[:, 4:5].abs() + reference[:, 5:6].abs() + 0.02
        )
        ndmi_change = target_ndmi - reference_ndmi
        sensor = F.one_hot(sensor_index, num_classes=len(SENSOR_NAMES)).to(values.dtype)
        sensor = sensor[:, :, None, None].expand(
            -1, -1, values.shape[-2], values.shape[-1]
        )
        return torch.cat(
            (
                torch.clamp(values[:, 0:1] - 1.0, min=-1.0, max=4.0),
                target_ndmi,
                reference_ndmi,
                ndmi_change,
                ndmi_change.abs(),
                normalized,
                log_ratio,
                values[:, 13:15],
                values[:, 15:16],
                observable,
                sensor,
            ),
            dim=1,
        )

    def forward(
        self,
        values: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        levels = [self.level1(self.inputs(values, observable, sensor_index))]
        for stage in self.stages:
            levels.append(stage(levels[-1]))
        return tuple(levels)


class CrossChannelTemporalFusion(nn.Module):
    """Fuse absolute spectra and target/reference changes at one scale."""

    def __init__(
        self, temporal_channels: int, guide_channels: int, output_channels: int
    ) -> None:
        super().__init__()
        self.fusion = ResidualBlock(4 * temporal_channels + guide_channels, output_channels)
        self.guide_scale = nn.Conv2d(guide_channels, output_channels, kernel_size=1)
        self.guide_shift = nn.Conv2d(guide_channels, output_channels, kernel_size=1)

    def forward(
        self,
        target: torch.Tensor,
        reference: torch.Tensor,
        guide: torch.Tensor,
    ) -> torch.Tensor:
        difference = target - reference
        fused = self.fusion(
            torch.cat((target, reference, difference, difference.abs(), guide), dim=1)
        )
        # Feature-wise change guidance is bounded so a few extreme spectral
        # ratios cannot dominate the entire representation.
        scale = 0.5 * torch.tanh(self.guide_scale(guide))
        shift = 0.5 * torch.tanh(self.guide_shift(guide))
        return fused * (1.0 + scale) + shift


class NdmiBitemporalFusionAdapter(nn.Module):
    """Identity-safe dense adapter with an independent patch/MIL scene head."""

    fused_channels = (40, 64, 96, 144, 192)
    teacher_channels = (64, 128, 256, 512, 512)

    def __init__(self) -> None:
        super().__init__()
        self.teacher = ReleasedMarsUNet()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.encoder = SharedSpectralEncoder()
        self.guide = MethaneChangeGuide()
        self.fusions = nn.ModuleList(
            CrossChannelTemporalFusion(temporal, guide, fused)
            for temporal, guide, fused in zip(
                self.encoder.channels, self.guide.channels, self.fused_channels
            )
        )
        self.teacher_gates = nn.ModuleList(
            IdentityFeatureGate(source, target)
            for source, target in zip(self.fused_channels, self.teacher_channels)
        )

        self.patch_head = nn.Sequential(
            ResidualBlock(self.fused_channels[2] + self.guide.channels[2], 96),
            nn.Conv2d(96, 1, kernel_size=1),
        )
        self.scene_attention = nn.Conv2d(self.fused_channels[-1], 1, kernel_size=1)
        self.sensor_embedding = nn.Embedding(len(SENSOR_NAMES), 8)
        scene_features = 3 * self.fused_channels[-1] + 2 + 8
        self.scene_head = nn.Sequential(
            nn.LayerNorm(scene_features),
            nn.Linear(scene_features, 192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def load_released_checkpoint(self, state: dict[str, torch.Tensor]) -> None:
        incompatible = self.teacher.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            raise ValueError(f"Released checkpoint is missing keys: {incompatible.missing_keys}")
        if any(not key.startswith("out_mlp.") for key in incompatible.unexpected_keys):
            raise ValueError(
                f"Released checkpoint has unexpected keys: {incompatible.unexpected_keys}"
            )

    def train(self, mode: bool = True) -> "NdmiBitemporalFusionAdapter":
        super().train(mode)
        self.teacher.eval()
        return self

    def _encode_teacher(self, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        level1 = self.teacher.inc(values)
        level2 = self.teacher.down1(level1)
        level3 = self.teacher.down2(level2)
        level4 = self.teacher.down3(level3)
        level5 = self.teacher.down4(level4)
        return level1, level2, level3, level4, level5

    def _decode_teacher(self, levels: tuple[torch.Tensor, ...]) -> torch.Tensor:
        level1, level2, level3, level4, level5 = levels
        decoded = self.teacher.up1(level5, level4)
        decoded = self.teacher.up2(decoded, level3)
        decoded = self.teacher.up3(decoded, level2)
        decoded = self.teacher.up4(decoded, level1)
        return self.teacher.out(decoded)

    @staticmethod
    def bounded_scene_evidence(scene_logit: torch.Tensor) -> torch.Tensor:
        return SCENE_EVIDENCE_BOUND * torch.tanh(
            scene_logit.float() / SCENE_EVIDENCE_BOUND
        )

    @staticmethod
    def fuse_scene_score(
        base_score: torch.Tensor,
        scene_logit: torch.Tensor,
        sensor_index: torch.Tensor,
        strength: float,
        *,
        sentinel_only: bool = True,
    ) -> torch.Tensor:
        """Add bounded learned evidence to an existing calibrated scene score."""

        if float(strength) == 0.0:
            return base_score.float()
        base_logit = torch.logit(base_score.float().clamp(1e-6, 1.0 - 1e-6))
        evidence = NdmiBitemporalFusionAdapter.bounded_scene_evidence(scene_logit)
        candidate = torch.sigmoid(base_logit + float(strength) * evidence)
        if sentinel_only:
            candidate = torch.where(sensor_index == 0, candidate, base_score.float())
        return candidate

    def _scene_logit(
        self,
        deepest: torch.Tensor,
        patch_logits: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
    ) -> torch.Tensor:
        visible = F.interpolate(observable.float(), deepest.shape[-2:], mode="nearest") > 0.5
        flat = deepest.flatten(2)
        valid = visible.flatten(2)
        weights = valid.to(flat.dtype)
        average = (flat * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        maximum = flat.masked_fill(~valid, -1e4).amax(dim=2)
        attention = self.scene_attention(deepest).flatten(2).masked_fill(~valid, -1e4)
        attended = (flat * torch.softmax(attention, dim=2)).sum(dim=2)
        patch_flat = patch_logits.flatten(1)
        top_count = min(8, patch_flat.shape[1])
        top = torch.topk(patch_flat, k=top_count, dim=1).values
        features = torch.cat(
            (
                attended,
                average,
                maximum,
                top.mean(dim=1, keepdim=True),
                top.max(dim=1, keepdim=True).values,
                self.sensor_embedding(sensor_index),
            ),
            dim=1,
        )
        return self.scene_head(features).squeeze(1)

    def forward(
        self,
        values: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if values.ndim != 4 or values.shape[1] != len(INPUT_CHANNELS):
            raise ValueError(f"Expected Bx{len(INPUT_CHANNELS)}xHxW MARS input")
        if observable.shape != values[:, :1].shape:
            raise ValueError("Observable mask must be Bx1xHxW")
        if sensor_index.shape != (values.shape[0],):
            raise ValueError("Sensor index must have shape B")

        with torch.no_grad():
            teacher_levels = self._encode_teacher(values)
            baseline_logits = self._decode_teacher(teacher_levels)
        target_levels = self.encoder(values[:, TARGET_SLICE])
        reference_levels = self.encoder(values[:, REFERENCE_SLICE])
        guide_levels = self.guide(values, observable, sensor_index)
        fused_levels = tuple(
            fusion(target, reference, guide)
            for fusion, target, reference, guide in zip(
                self.fusions, target_levels, reference_levels, guide_levels
            )
        )
        adapted_levels = tuple(
            gate(teacher.detach(), fused)
            for gate, teacher, fused in zip(
                self.teacher_gates, teacher_levels, fused_levels
            )
        )
        segmentation_logits = self._decode_teacher(adapted_levels)
        patch_features = torch.cat((fused_levels[2], guide_levels[2]), dim=1)
        patch_logits = F.adaptive_avg_pool2d(
            self.patch_head(patch_features), (PATCH_GRID_SIZE, PATCH_GRID_SIZE)
        )
        scene_logit = self._scene_logit(
            fused_levels[-1], patch_logits, observable, sensor_index
        )
        return {
            "segmentation_logits": segmentation_logits,
            "baseline_logits": baseline_logits,
            "correction_logits": segmentation_logits - baseline_logits,
            "patch_logits": patch_logits,
            "scene_logit": scene_logit,
            "scene_delta_logit": scene_logit,
        }

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("teacher.")
        }

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "input_channels": list(INPUT_CHANNELS),
            "sensor_names": list(SENSOR_NAMES),
            "temporal_weight_sharing": True,
            "patch_grid_size": PATCH_GRID_SIZE,
            "scene_evidence_bound": SCENE_EVIDENCE_BOUND,
            "initial_dense_equivalence": "exact released MARS-S2L logits",
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "physics_guidance": [
                "MBMP",
                "target/reference/difference NDMI",
                "six normalized temporal band differences",
                "six temporal log-ratios",
                "wind",
                "cloud",
                "observability",
                "sensor identity",
            ],
            "scene_head": "independently supervised patch logits plus attentive/mean/max deep MIL",
        }
