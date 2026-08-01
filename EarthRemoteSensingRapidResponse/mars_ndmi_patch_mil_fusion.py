"""Patch-local scene variant of the NDMI-guided bi-temporal fusion model."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from mars_ndmi_bitemporal_fusion import NdmiBitemporalFusionAdapter


MODEL_NAME = "ersrr_ndmi_bitemporal_patch_mil_v1"


class NdmiPatchMilFusionAdapter(NdmiBitemporalFusionAdapter):
    """Derive scene evidence only from directly mask-supervised patch logits."""

    def __init__(self) -> None:
        super().__init__()
        # The rejected parent pilot used these global context modules. Removing
        # them makes scene evidence local, spatially accountable, and unable to
        # classify a source merely from a pooled landscape representation.
        del self.scene_attention
        del self.sensor_embedding
        del self.scene_head

    def _scene_logit(
        self,
        deepest: torch.Tensor,
        patch_logits: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
    ) -> torch.Tensor:
        del deepest, sensor_index
        visible = (
            F.adaptive_avg_pool2d(observable.float(), patch_logits.shape[-2:]) >= 0.9
        )
        flattened = patch_logits.masked_fill(~visible, -20.0).flatten(1)
        top_count = min(8, flattened.shape[1])
        top = torch.topk(flattened, k=top_count, dim=1).values
        return 0.75 * top.mean(dim=1) + 0.25 * top.max(dim=1).values

    def artifact_metadata(self) -> dict[str, object]:
        metadata = super().artifact_metadata()
        metadata.update(
            {
                "model_name": MODEL_NAME,
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                ),
                "scene_head": (
                    "deterministic 0.75*mean(top-8)+0.25*max of directly "
                    "mask-supervised 8x8 patch logits"
                ),
                "global_context_scene_features": False,
            }
        )
        return metadata
