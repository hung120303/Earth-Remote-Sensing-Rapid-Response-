"""Instance-aware, physics-guided adapters for the released MARS-S2L U-Net."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mars_paper_model import ResidualBlock
from mars_physics_guided_teacher import PhysicsGuidedTeacherAdapter


class InstanceGuidedTeacherAdapter(PhysicsGuidedTeacherAdapter):
    """Add plume-object supervision while preserving the released model at init."""

    def __init__(self) -> None:
        super().__init__()
        self.object_fusion = ResidualBlock(64 + 128, 64)
        self.object_head = nn.Conv2d(64, 2, kernel_size=1)
        self.pixel_fusion = ResidualBlock(32 + 64 + 64, 48)
        self.instance_delta = nn.Conv2d(48, 1, kernel_size=1)
        nn.init.zeros_(self.instance_delta.weight)
        nn.init.zeros_(self.instance_delta.bias)

    @staticmethod
    def proposal_scene_surrogate(
        object_logits: torch.Tensor, observable: torch.Tensor
    ) -> torch.Tensor:
        occupancy = object_logits[:, 0:1]
        center = object_logits[:, 1:2]
        coarse_valid = F.avg_pool2d(observable, kernel_size=2, stride=2) >= 0.9
        masked_occupancy = occupancy.masked_fill(~coarse_valid, -20.0).flatten(1)
        k = min(25, masked_occupancy.shape[1])
        occupancy_score = torch.topk(masked_occupancy, k=k, dim=1).values.mean(dim=1)
        center_score = center.masked_fill(~coarse_valid, -20.0).flatten(1).amax(dim=1)
        return 0.8 * occupancy_score + 0.2 * center_score

    def forward(
        self, values: torch.Tensor, observable: torch.Tensor, sensor_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            teacher_levels = self._encode(values)
            baseline_logits = self._decode(teacher_levels)
        physics_levels = self.physics(values, baseline_logits.detach(), sensor_index)
        adapted = tuple(
            gate(teacher.detach(), physics)
            for gate, teacher, physics in zip(self.gates, teacher_levels, physics_levels)
        )
        adapted_logits = self._decode(adapted)

        object_features = self.object_fusion(
            torch.cat((physics_levels[1], adapted[1]), dim=1)
        )
        object_logits = self.object_head(object_features)
        object_gate = torch.sigmoid(
            F.interpolate(
                object_logits[:, 0:1],
                size=baseline_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        full_object_features = F.interpolate(
            object_features,
            size=baseline_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pixel_features = self.pixel_fusion(
            torch.cat((physics_levels[0], adapted[0], full_object_features), dim=1)
        )
        bounded_instance_delta = 2.0 * torch.tanh(self.instance_delta(pixel_features))
        raw_correction = adapted_logits - baseline_logits + bounded_instance_delta
        correction = object_gate * raw_correction
        logits = baseline_logits + correction
        pixel_scene = self.scene_surrogate(logits, observable)
        proposal_scene = self.proposal_scene_surrogate(object_logits, observable)
        return {
            "segmentation_logits": logits,
            "baseline_logits": baseline_logits,
            "correction_logits": correction,
            "raw_correction_logits": raw_correction,
            "object_logits": object_logits,
            "object_gate": object_gate,
            "scene_logit": 0.75 * pixel_scene + 0.25 * proposal_scene,
        }

