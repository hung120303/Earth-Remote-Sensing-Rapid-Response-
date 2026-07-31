"""DINOv3 semantic-change fusion around the frozen released MARS-S2L U-Net."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from mars_paper_model import ResidualBlock
from mars_physics_guided_teacher import (
    IdentityFeatureGate,
    PhysicsGuidedTeacherAdapter,
)
from mars_v4_model import INPUT_CHANNELS


DINO_MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
DINO_BLOCKS = (2, 5, 8, 11)
DINO_EMBED_DIM = 384
DINO_GRID_SIZE = 16
COUNTERFACTUAL_CHANNELS = 28
PATCH_GRID_SIZE = 8
SCENE_PROTECTION_GATE = 0.5
RGB_TARGET_INDICES = (3, 2, 1)
RGB_REFERENCE_INDICES = (9, 8, 7)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoTemporalProjection(nn.Module):
    """Project signed, absolute, and contextual temporal DINO features."""

    def __init__(self, output_channels: int = 64) -> None:
        super().__init__()
        input_channels = 3 * DINO_EMBED_DIM
        self.block = nn.Sequential(
            nn.GroupNorm(24, input_channels, affine=False),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            ResidualBlock(output_channels, output_channels),
        )

    def forward(self, target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        difference = target - reference
        context = 0.5 * (target + reference)
        return self.block(torch.cat((difference, difference.abs(), context), dim=1))


class FrozenDinoTemporalEncoder(nn.Module):
    """Extract four frozen DINOv3 target/reference feature pairs on demand."""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self.backbone = timm.create_model(DINO_MODEL_NAME, pretrained=False)
        incompatible = self.backbone.load_state_dict(
            load_file(checkpoint.as_posix(), device="cpu"), strict=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("DINOv3 checkpoint does not match the frozen architecture")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD, dtype=torch.float32)[None, :, None, None]
        )
        self.backbone.eval()

    @staticmethod
    def _shared_contrast_pair(
        target: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a sample-local shared 2/98 percent stretch to both dates."""

        combined = torch.cat((target, reference), dim=1).float()
        flattened = combined.flatten(1)
        low = torch.quantile(flattened, 0.02, dim=1, keepdim=True)
        high = torch.quantile(flattened, 0.98, dim=1, keepdim=True)
        scale = (high - low).clamp_min(1e-4)
        low = low[:, :, None, None]
        scale = scale[:, :, None, None]
        return (
            ((target.float() - low) / scale).clamp(0.0, 1.0),
            ((reference.float() - low) / scale).clamp(0.0, 1.0),
        )

    def _rgb_pair(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The MARS adapter divides the producer's 0..10000 reflectance scale by
        # 5000. The semantic stream is contrast-stretched jointly, while the
        # physics stream retains the calibrated values unchanged.
        target = values[:, RGB_TARGET_INDICES]
        reference = values[:, RGB_REFERENCE_INDICES]
        target, reference = self._shared_contrast_pair(target, reference)
        pair = torch.cat((target, reference), dim=0)
        pair = F.interpolate(
            pair,
            size=(256, 256),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
        pair = (pair - self.mean) / self.std
        return pair[: target.shape[0]], pair[target.shape[0] :]

    def train(self, mode: bool = True) -> "FrozenDinoTemporalEncoder":
        super().train(False)
        self.backbone.eval()
        return self

    def forward(
        self, values: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if values.ndim != 4 or values.shape[1] != len(INPUT_CHANNELS):
            raise ValueError(f"Expected Bx{len(INPUT_CHANNELS)}xHxW MARS input")
        target, reference = self._rgb_pair(values)
        batch_size = target.shape[0]
        with torch.no_grad():
            outputs = self.backbone.forward_intermediates(
                torch.cat((target, reference), dim=0),
                indices=list(DINO_BLOCKS),
                norm=True,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        pairs = tuple((value[:batch_size], value[batch_size:]) for value in outputs)
        expected = (DINO_EMBED_DIM, DINO_GRID_SIZE, DINO_GRID_SIZE)
        if len(pairs) != len(DINO_BLOCKS) or any(
            tuple(target_map.shape[1:]) != expected
            or tuple(reference_map.shape[1:]) != expected
            for target_map, reference_map in pairs
        ):
            raise RuntimeError("DINOv3 intermediate geometry differs from the frozen contract")
        return pairs


class CounterfactualMethaneEncoder(nn.Module):
    """Encode label-free MARS counterfactual and spectral-change maps."""

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = nn.GroupNorm(4, COUNTERFACTUAL_CHANNELS, affine=False)
        self.level1 = ResidualBlock(COUNTERFACTUAL_CHANNELS, 64, stride=2)
        self.level2 = ResidualBlock(64, 96, stride=2)
        self.level3 = ResidualBlock(96, 96)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1:] != (
            COUNTERFACTUAL_CHANNELS,
            64,
            64,
        ):
            raise ValueError("Expected Bx28x64x64 counterfactual methane maps")
        return self.level3(self.level2(self.level1(self.input_norm(values.float()))))


class DinoMethaneFusionAdapter(PhysicsGuidedTeacherAdapter):
    """Identity-safe spatial fusion of DINO semantics and methane physics."""

    def __init__(self, dino_checkpoint: Path) -> None:
        super().__init__()
        del self.gates
        self.dino_encoder = FrozenDinoTemporalEncoder(dino_checkpoint)
        self.temporal_projections = nn.ModuleList(
            DinoTemporalProjection() for _ in DINO_BLOCKS
        )
        self.semantic_fusion = ResidualBlock(64 * len(DINO_BLOCKS), 192)
        self.methane_encoder = CounterfactualMethaneEncoder()
        self.methane_gate = nn.Conv2d(96, 192, kernel_size=1)
        self.deep_fusions = nn.ModuleList(
            (
                ResidualBlock(96 + 96 + 192, 160),
                ResidualBlock(128 + 96 + 192, 192),
                ResidualBlock(160 + 96 + 192, 224),
            )
        )
        self.deep_gates = nn.ModuleList(
            (
                IdentityFeatureGate(160, 256),
                IdentityFeatureGate(192, 512),
                IdentityFeatureGate(224, 512),
            )
        )
        self.patch_head = nn.Conv2d(192 + 96, 1, kernel_size=1)
        self.scene_attention = nn.Conv2d(192, 1, kernel_size=1)
        self.scene_fusion = ResidualBlock(512 + 224, 192)
        self.sensor_embedding = nn.Embedding(2, 8)
        self.scene_hidden = nn.Sequential(
            nn.Linear(192 * 3 + 8 + 1, 192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(192, 64),
            nn.GELU(),
        )
        self.scene_output = nn.Linear(64, 1)
        nn.init.zeros_(self.scene_output.weight)
        nn.init.zeros_(self.scene_output.bias)

    def train(self, mode: bool = True) -> "DinoMethaneFusionAdapter":
        super().train(mode)
        self.teacher.eval()
        self.dino_encoder.eval()
        return self

    @staticmethod
    def _resize(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            values, size=target.shape[-2:], mode="bilinear", align_corners=False
        )

    @staticmethod
    def protected_scene_score(
        base_score: torch.Tensor, delta_logit: torch.Tensor, strength: float = 1.0
    ) -> torch.Tensor:
        """Preserve every score below the operating region exactly.

        Scores above the fixed gate are mapped into local [0, 1] coordinates,
        adjusted in log-odds space, and mapped back above the same gate. This
        prevents the scene branch from perturbing low-FPR boundary examples.
        """

        gate = SCENE_PROTECTION_GATE
        local = ((base_score.float() - gate) / (1.0 - gate)).clamp(1e-6, 1.0 - 1e-6)
        local_logit = torch.logit(local)
        adjustment = (1.0 - gate) * (
            torch.sigmoid(local_logit + float(strength) * delta_logit.float())
            - torch.sigmoid(local_logit)
        )
        adjusted = base_score.float() + adjustment
        return torch.where(base_score.float() < gate, base_score.float(), adjusted)

    def _semantic_change(self, values: torch.Tensor) -> torch.Tensor:
        pairs = self.dino_encoder(values)
        projected = tuple(
            projection(target, reference)
            for projection, (target, reference) in zip(self.temporal_projections, pairs)
        )
        return self.semantic_fusion(torch.cat(projected, dim=1))

    def _scene_residual(
        self,
        teacher_level5: torch.Tensor,
        source_level5: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
        base_scene_logit: torch.Tensor,
    ) -> torch.Tensor:
        scene_map = self.scene_fusion(
            torch.cat((teacher_level5, self._resize(source_level5, teacher_level5)), dim=1)
        )
        visible = F.interpolate(observable.float(), scene_map.shape[-2:], mode="nearest")
        flat = scene_map.flatten(2)
        valid = visible.flatten(2) > 0.5
        weights = valid.to(flat.dtype)
        average = (flat * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        maximum = flat.masked_fill(~valid, -1e4).amax(dim=2)
        attention_logits = self.scene_attention(scene_map).flatten(2)
        attention_logits = attention_logits.masked_fill(~valid, -1e4)
        attended = (flat * torch.softmax(attention_logits, dim=2)).sum(dim=2)
        features = torch.cat(
            (
                attended,
                average,
                maximum,
                self.sensor_embedding(sensor_index),
                base_scene_logit[:, None],
            ),
            dim=1,
        )
        return 2.0 * torch.tanh(self.scene_output(self.scene_hidden(features)).squeeze(1))

    def forward(
        self,
        values: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
        counterfactual_maps: torch.Tensor,
        base_scene_score: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if base_scene_score.ndim != 1 or base_scene_score.shape[0] != values.shape[0]:
            raise ValueError("Cross-fitted base scene scores must be a length-B vector")
        with torch.no_grad():
            teacher_levels = self._encode(values)
            baseline_logits = self._decode(teacher_levels)
        physics_levels = self.physics(values, baseline_logits.detach(), sensor_index)
        methane = self.methane_encoder(counterfactual_maps)
        semantic = self._semantic_change(values)
        semantic = semantic * (0.5 + torch.sigmoid(self.methane_gate(methane)))
        source3 = self.deep_fusions[0](
            torch.cat(
                (
                    physics_levels[2],
                    self._resize(methane, physics_levels[2]),
                    self._resize(semantic, physics_levels[2]),
                ),
                dim=1,
            )
        )
        source4 = self.deep_fusions[1](
            torch.cat(
                (
                    physics_levels[3],
                    self._resize(methane, physics_levels[3]),
                    self._resize(semantic, physics_levels[3]),
                ),
                dim=1,
            )
        )
        source5 = self.deep_fusions[2](
            torch.cat(
                (
                    physics_levels[4],
                    self._resize(methane, physics_levels[4]),
                    self._resize(semantic, physics_levels[4]),
                ),
                dim=1,
            )
        )
        adapted = (
            teacher_levels[0].detach(),
            teacher_levels[1].detach(),
            self.deep_gates[0](teacher_levels[2].detach(), source3),
            self.deep_gates[1](teacher_levels[3].detach(), source4),
            self.deep_gates[2](teacher_levels[4].detach(), source5),
        )
        logits = self._decode(adapted)
        base_scene_logit = torch.logit(base_scene_score.float().clamp(1e-6, 1.0 - 1e-6))
        scene_delta = self._scene_residual(
            adapted[4], source5, observable, sensor_index, base_scene_logit
        )
        scene_score = self.protected_scene_score(base_scene_score, scene_delta)
        patch_logits = F.adaptive_avg_pool2d(
            self.patch_head(torch.cat((semantic, methane), dim=1)),
            (PATCH_GRID_SIZE, PATCH_GRID_SIZE),
        )
        return {
            "segmentation_logits": logits,
            "baseline_logits": baseline_logits,
            "correction_logits": logits - baseline_logits,
            "patch_logits": patch_logits,
            "base_scene_logit": base_scene_logit,
            "scene_delta_logit": scene_delta,
            "scene_score": scene_score,
            "scene_logit": torch.logit(scene_score.clamp(1e-6, 1.0 - 1e-6)),
        }

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith(("teacher.", "dino_encoder.backbone."))
        }
