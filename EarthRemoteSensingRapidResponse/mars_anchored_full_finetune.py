"""Weight-space anchored adaptation of the exact released MARS-S2L U-Net."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mars_paper_model import ReleasedMarsUNet, SENSOR_NAMES
from mars_v4_model import INPUT_CHANNELS

MODEL_NAME = "ersrr_mars_anchored_full_finetune_v1"
MODEL_SCHEMA_VERSION = 1
PATCH_GRID_SIZE = 8
SCENE_EVIDENCE_BOUND = 2.0


class AnchoredMarsFullFinetune(nn.Module):
    """Fine-tune released filters while retaining an immutable released teacher."""

    def __init__(self, *, scene_topk_fraction: float = 0.01) -> None:
        super().__init__()
        if not 0.0 < scene_topk_fraction <= 1.0:
            raise ValueError("scene_topk_fraction must be in (0, 1]")
        self.scene_topk_fraction = float(scene_topk_fraction)
        self.teacher = ReleasedMarsUNet()
        self.student = ReleasedMarsUNet()
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        # The released BN population statistics define part of the radiometric
        # contract. Freeze both their affine terms and their running state.
        for module in self.student.modules():
            if isinstance(module, nn.BatchNorm2d):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        self._anchor: dict[str, torch.Tensor] = {}

    def load_released_checkpoint(self, state: dict[str, torch.Tensor]) -> None:
        for model in (self.teacher, self.student):
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.missing_keys:
                raise ValueError(
                    f"Released checkpoint is missing keys: {incompatible.missing_keys}"
                )
            if any(not key.startswith("out_mlp.") for key in incompatible.unexpected_keys):
                raise ValueError(
                    f"Released checkpoint has unexpected keys: {incompatible.unexpected_keys}"
                )
        self._anchor = {
            name: parameter.detach().clone()
            for name, parameter in self.student.named_parameters()
            if parameter.requires_grad
        }

    def train(self, mode: bool = True) -> "AnchoredMarsFullFinetune":
        super().train(mode)
        self.teacher.eval()
        # Keep student BN in released inference mode even while convolutions train.
        for module in self.student.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def parameter_groups(
        self, *, backbone_learning_rate: float, output_learning_rate: float
    ) -> list[dict[str, Any]]:
        body: list[nn.Parameter] = []
        output: list[nn.Parameter] = []
        for name, parameter in self.student.named_parameters():
            if not parameter.requires_grad:
                continue
            (output if name.startswith("out.") else body).append(parameter)
        if not body or not output:
            raise ValueError("Expected trainable body and output-head parameters")
        return [
            {"params": body, "lr": float(backbone_learning_rate)},
            {"params": output, "lr": float(output_learning_rate)},
        ]

    def anchor_penalty(self) -> torch.Tensor:
        if not self._anchor:
            raise RuntimeError("Released checkpoint must be loaded before training")
        total: torch.Tensor | None = None
        count = 0
        for name, parameter in self.student.named_parameters():
            if not parameter.requires_grad:
                continue
            anchor = self._anchor[name]
            scale = anchor.square().mean().clamp_min(1e-8)
            value = (parameter - anchor).square().mean() / scale
            total = value if total is None else total + value
            count += 1
        if total is None or count == 0:
            raise RuntimeError("No trainable anchored parameters")
        return total / count

    @staticmethod
    def bounded_scene_evidence(scene_delta_logit: torch.Tensor) -> torch.Tensor:
        return SCENE_EVIDENCE_BOUND * torch.tanh(
            scene_delta_logit.float() / SCENE_EVIDENCE_BOUND
        )

    @staticmethod
    def fuse_scene_score(
        base_score: torch.Tensor,
        scene_logit: torch.Tensor,
        sensor_index: torch.Tensor,
        strength: float,
        *,
        sentinel_only: bool = False,
    ) -> torch.Tensor:
        if float(strength) == 0.0:
            return base_score.float()
        base_logit = torch.logit(base_score.float().clamp(1e-6, 1.0 - 1e-6))
        evidence = AnchoredMarsFullFinetune.bounded_scene_evidence(scene_logit)
        candidate = torch.sigmoid(base_logit + float(strength) * evidence)
        if sentinel_only:
            candidate = torch.where(sensor_index == 0, candidate, base_score.float())
        return candidate

    def _scene_delta(
        self, correction: torch.Tensor, observable: torch.Tensor
    ) -> torch.Tensor:
        flat = correction.flatten(1)
        visible = observable.flatten(1) > 0.5
        masked = flat.masked_fill(~visible, -1e4)
        count = max(1, int(flat.shape[1] * self.scene_topk_fraction))
        return torch.topk(masked, k=count, dim=1).values.mean(dim=1)

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
        if torch.any((sensor_index < 0) | (sensor_index >= len(SENSOR_NAMES))):
            raise ValueError("Sensor index is outside the frozen mixed-sensor contract")

        with torch.no_grad():
            baseline_logits = self.teacher(values)
        segmentation_logits = self.student(values)
        correction = segmentation_logits - baseline_logits
        scene_delta = self._scene_delta(correction, observable)
        # Directly supervise the student at coarse plume-support scale; the
        # complementary scene head remains the signed student-teacher change.
        patch_logits = F.adaptive_max_pool2d(
            segmentation_logits, (PATCH_GRID_SIZE, PATCH_GRID_SIZE)
        )
        return {
            "segmentation_logits": segmentation_logits,
            "baseline_logits": baseline_logits,
            "correction_logits": correction,
            "patch_logits": patch_logits,
            "scene_logit": scene_delta,
            "scene_delta_logit": scene_delta,
        }

    def trainable_state(self) -> dict[str, torch.Tensor]:
        # Persist the complete student, including immutable BN buffers, so a
        # promoted endpoint has no dependency on implicit reconstruction.
        return {
            name: value.detach().cpu()
            for name, value in self.student.state_dict().items()
        }

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "input_channels": list(INPUT_CHANNELS),
            "sensor_names": list(SENSOR_NAMES),
            "scene_topk_fraction": self.scene_topk_fraction,
            "scene_evidence_bound": SCENE_EVIDENCE_BOUND,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self.student.parameters()
                if parameter.requires_grad
            ),
            "frozen_batch_normalization": True,
            "initial_equivalence": "exact released MARS-S2L logits",
            "adaptation": "full convolutional student with normalized L2-SP anchoring",
        }
