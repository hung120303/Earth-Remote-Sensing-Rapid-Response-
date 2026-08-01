"""Recall-protected bi-sensor fusion for anchored MARS full fine-tuning."""

from __future__ import annotations

from typing import Any

import torch

from mars_anchored_full_finetune import AnchoredMarsFullFinetune

MODEL_NAME = "ersrr_mars_protected_bisensor_finetune_v1"
PROTECTION_GATE = 0.25


class ProtectedBisensorFinetune(AnchoredMarsFullFinetune):
    """Rerank only above a fixed gate and apply evidence to both sensors."""

    @staticmethod
    def fuse_scene_score(
        base_score: torch.Tensor,
        scene_logit: torch.Tensor,
        sensor_index: torch.Tensor,
        strength: float,
        *,
        sentinel_only: bool = False,
    ) -> torch.Tensor:
        del sentinel_only
        if float(strength) == 0.0:
            return base_score.float()
        base = base_score.float()
        if not bool(torch.isfinite(base).all()) or bool(((base < 0) | (base > 1)).any()):
            raise ValueError("Base scores must be finite probabilities")
        if sensor_index.shape != base.shape:
            raise ValueError("Sensor indices must align with base scores")
        high = base >= PROTECTION_GATE
        local = ((base - PROTECTION_GATE) / (1.0 - PROTECTION_GATE)).clamp(
            1e-6, 1.0 - 1e-6
        )
        evidence = ProtectedBisensorFinetune.bounded_scene_evidence(scene_logit)
        reranked = PROTECTION_GATE + (1.0 - PROTECTION_GATE) * torch.sigmoid(
            torch.logit(local) + float(strength) * evidence
        )
        return torch.where(high, reranked, base)

    def artifact_metadata(self) -> dict[str, Any]:
        metadata = super().artifact_metadata()
        metadata.update(
            {
                "model_name": MODEL_NAME,
                "scene_routing": "both Sentinel-2 and Landsat",
                "protection_gate": PROTECTION_GATE,
                "operating_region": "scores below the gate are exact identity",
            }
        )
        return metadata
