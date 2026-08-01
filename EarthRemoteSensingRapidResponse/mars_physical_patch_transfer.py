"""Physical-scale cross-domain methane patch detector for the MARS successor.

The model deliberately works on a 640 m field of view.  MethaneS2CM supplies
native 32x32 crops at 20 m while MARS supplies 64x64 crops at 10 m that are
area-pooled to the same 32x32 grid.  Both sources are represented by the same
tri-temporal channel contract before entering a shared encoder.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

from mars_paper_model import SENSOR_NAMES
from mars_v4_model import DecoderBlock, INPUT_CHANNELS, ResidualBlock
from methanes2cm_adapter import V5_INPUT_CHANNELS


MODEL_NAME = "ersrr_mars_physical_patch_transfer_v1"
MODEL_SCHEMA_VERSION = 1
PATCH_PIXELS = 32
MARS_TILE_PIXELS = 64
MARS_TILE_STRIDE = 32
SCENE_EVIDENCE_BOUND = 2.0
TARGET_SLICE = slice(2, 8)
REFERENCE90_SLICE = slice(8, 14)
REFERENCE365_SLICE = slice(14, 20)
GUIDE_CHANNELS = 8


def _normalized_ratio_torch(
    b11: torch.Tensor,
    b12: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Torch equivalent of the released validity-aware MBMP ratio."""

    if b11.shape != b12.shape or b11.shape != valid.shape:
        raise ValueError("MBMP ratio tensors must have matching BxHxW shapes")
    result = torch.ones_like(b11)
    for index in range(b11.shape[0]):
        usable = valid[index] & torch.isfinite(b11[index]) & torch.isfinite(b12[index])
        usable = usable & (b11[index] != 0)
        if not torch.any(usable):
            continue
        ratio = b12[index][usable] / b11[index][usable]
        median = torch.median(ratio)
        if not torch.isfinite(median) or torch.abs(median) < 1e-8:
            median = torch.ones((), dtype=ratio.dtype, device=ratio.device)
        normalized = torch.clamp(ratio / median, min=0.0, max=10.0)
        result[index][usable] = normalized
    return result


def compute_mbmp_torch(
    target: torch.Tensor,
    reference: torch.Tensor,
    observable: torch.Tensor,
) -> torch.Tensor:
    """Recompute the released MBMP after spatial area pooling."""

    if target.ndim != 4 or target.shape[1] != 6 or target.shape != reference.shape:
        raise ValueError("Expected matching Bx6xHxW target and reference tensors")
    if observable.shape != target[:, :1].shape:
        raise ValueError("Observable mask must be Bx1xHxW")
    valid = observable[:, 0] > 0.5
    target_ratio = _normalized_ratio_torch(target[:, 4], target[:, 5], valid)
    reference_ratio = _normalized_ratio_torch(reference[:, 4], reference[:, 5], valid)
    mbmp = torch.ones_like(target_ratio)
    usable = valid & (reference_ratio != 0)
    mbmp[usable] = target_ratio[usable] / reference_ratio[usable]
    mbmp = torch.nan_to_num(mbmp, nan=1.0, posinf=1.0, neginf=1.0)
    mbmp = torch.clamp(mbmp, min=0.0, max=10.0)
    return torch.where(valid, mbmp, torch.ones_like(mbmp))[:, None]


def mars_tile_to_canonical(
    values: torch.Tensor,
    observable: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Area-pool one even-sized MARS tile and emit the shared 20-channel form.

    Returns ``canonical, auxiliary, pooled_observable``.  The two-reference
    contract duplicates MARS's single historical reference; auxiliary contains
    wind-u, wind-v, and conservative max-pooled cloud occupancy.
    """

    if values.ndim != 4 or values.shape[1] != len(INPUT_CHANNELS):
        raise ValueError(f"Expected Bx{len(INPUT_CHANNELS)}xHxW MARS input")
    if observable.shape != values[:, :1].shape:
        raise ValueError("Observable mask must match the MARS tile")
    if values.shape[-2] % 2 or values.shape[-1] % 2:
        raise ValueError("MARS physical tiles must have even spatial dimensions")
    spectral = F.avg_pool2d(values[:, 1:13].float(), kernel_size=2, stride=2)
    target = spectral[:, :6]
    reference = spectral[:, 6:]
    pooled_observable = (
        F.avg_pool2d(observable.float(), kernel_size=2, stride=2) >= 1.0 - 1e-6
    ).to(values.dtype)
    target = target * pooled_observable
    reference = reference * pooled_observable
    mbmp = compute_mbmp_torch(target, reference, pooled_observable)
    canonical = torch.cat((mbmp, mbmp, target, reference, reference), dim=1)
    wind = F.avg_pool2d(values[:, 13:15].float(), kernel_size=2, stride=2)
    cloud = F.max_pool2d(values[:, 15:16].float(), kernel_size=2, stride=2)
    auxiliary = torch.cat((wind, cloud), dim=1)
    canonical = canonical.to(values.dtype)
    auxiliary = auxiliary.to(values.dtype)
    canonical[:, :2] = torch.where(
        pooled_observable > 0.5,
        canonical[:, :2],
        torch.ones_like(canonical[:, :2]),
    )
    canonical[:, 2:] *= pooled_observable
    return canonical, auxiliary, pooled_observable


def physical_tile_starts(
    scene_pixels: int,
    tile_pixels: int = MARS_TILE_PIXELS,
    stride: int = MARS_TILE_STRIDE,
) -> tuple[int, ...]:
    """Return deterministic starts with an edge-aligned final tile."""

    if scene_pixels < tile_pixels or tile_pixels <= 0 or stride <= 0:
        raise ValueError("Invalid physical tiling dimensions")
    starts = list(range(0, scene_pixels - tile_pixels + 1, stride))
    final = scene_pixels - tile_pixels
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


def gradient_reverse(values: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReverse.apply(values, float(strength))


class SharedTriTemporalEncoder(nn.Module):
    channels = (24, 48, 96, 160, 256)

    def __init__(self) -> None:
        super().__init__()
        self.stem = ResidualBlock(6, self.channels[0])
        self.stages = nn.ModuleList(
            ResidualBlock(previous, current, stride=2)
            for previous, current in zip(self.channels[:-1], self.channels[1:])
        )

    def forward(self, values: torch.Tensor) -> list[torch.Tensor]:
        levels = [self.stem(values)]
        for stage in self.stages:
            levels.append(stage(levels[-1]))
        return levels


class GuidedTriTemporalFusion(nn.Module):
    """Fuse absolute spectra, two changes, and operational metadata."""

    def __init__(self, temporal_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = ResidualBlock(7 * temporal_channels + GUIDE_CHANNELS, output_channels)
        self.guide_scale = nn.Conv2d(GUIDE_CHANNELS, output_channels, kernel_size=1)
        self.guide_shift = nn.Conv2d(GUIDE_CHANNELS, output_channels, kernel_size=1)

    def forward(
        self,
        target: torch.Tensor,
        reference90: torch.Tensor,
        reference365: torch.Tensor,
        guide: torch.Tensor,
    ) -> torch.Tensor:
        difference90 = target - reference90
        difference365 = target - reference365
        guide = F.interpolate(guide, target.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.block(
            torch.cat(
                (
                    target,
                    reference90,
                    reference365,
                    difference90,
                    difference365,
                    difference90.abs(),
                    difference365.abs(),
                    guide,
                ),
                dim=1,
            )
        )
        scale = 0.25 * torch.tanh(self.guide_scale(guide))
        shift = 0.25 * torch.tanh(self.guide_shift(guide))
        return fused * (1.0 + scale) + shift


class PhysicalPatchTransferDetector(nn.Module):
    """Shared 640 m plume detector with a Sentinel-2 source adversary."""

    fused_channels = (32, 56, 96, 144, 192)

    def __init__(self, context_scene_weight: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= context_scene_weight <= 1.0:
            raise ValueError("context_scene_weight must be in [0, 1]")
        self.context_scene_weight = float(context_scene_weight)
        self.encoder = SharedTriTemporalEncoder()
        self.fusions = nn.ModuleList(
            GuidedTriTemporalFusion(temporal, fused)
            for temporal, fused in zip(self.encoder.channels, self.fused_channels)
        )
        self.decoder4 = DecoderBlock(192, 144, 144)
        self.decoder3 = DecoderBlock(144, 96, 96)
        self.decoder2 = DecoderBlock(96, 56, 56)
        self.decoder1 = DecoderBlock(56, 32, 48)
        self.segmentation = nn.Conv2d(48, 1, kernel_size=1)
        representation_channels = 2 * self.fused_channels[-1]
        context_channels = representation_channels + 3 * GUIDE_CHANNELS
        self.context_scene = nn.Sequential(
            nn.LayerNorm(context_channels),
            nn.Linear(context_channels, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.domain_head = nn.Sequential(
            nn.LayerNorm(representation_channels),
            nn.Linear(representation_channels, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _guide(
        inputs: torch.Tensor,
        auxiliary: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
    ) -> torch.Tensor:
        sensor = F.one_hot(sensor_index, num_classes=len(SENSOR_NAMES)).to(inputs.dtype)
        sensor = sensor[:, :, None, None].expand(-1, -1, inputs.shape[-2], inputs.shape[-1])
        return torch.cat(
            (
                torch.clamp(inputs[:, :2] - 1.0, min=-1.0, max=4.0),
                auxiliary,
                observable,
                sensor,
            ),
            dim=1,
        )

    @staticmethod
    def bounded_scene_evidence(scene_logit: torch.Tensor) -> torch.Tensor:
        return SCENE_EVIDENCE_BOUND * torch.tanh(scene_logit.float() / SCENE_EVIDENCE_BOUND)

    @staticmethod
    def fuse_scene_score(
        base_score: torch.Tensor,
        scene_logit: torch.Tensor,
        sensor_index: torch.Tensor,
        strength: float,
        *,
        sentinel_only: bool = True,
    ) -> torch.Tensor:
        if float(strength) == 0.0:
            return base_score.float()
        base_logit = torch.logit(base_score.float().clamp(1e-6, 1.0 - 1e-6))
        candidate = torch.sigmoid(
            base_logit
            + float(strength) * PhysicalPatchTransferDetector.bounded_scene_evidence(scene_logit)
        )
        if sentinel_only:
            candidate = torch.where(sensor_index == 0, candidate, base_score.float())
        return candidate

    @staticmethod
    def aggregate_tile_scene_logits(tile_logits: torch.Tensor) -> torch.Tensor:
        """Aggregate BxT local logits with the frozen top-four rule."""

        if tile_logits.ndim != 2:
            raise ValueError("Tile scene logits must have shape BxT")
        top_count = min(4, tile_logits.shape[1])
        top = torch.topk(tile_logits, k=top_count, dim=1).values
        return 0.75 * top.mean(dim=1) + 0.25 * top.max(dim=1).values

    def forward(
        self,
        inputs: torch.Tensor,
        auxiliary: torch.Tensor,
        observable: torch.Tensor,
        sensor_index: torch.Tensor,
        *,
        grl_strength: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != len(V5_INPUT_CHANNELS):
            raise ValueError(f"Expected Bx{len(V5_INPUT_CHANNELS)}xHxW canonical input")
        if inputs.shape[-2:] != (PATCH_PIXELS, PATCH_PIXELS):
            raise ValueError(f"Expected {PATCH_PIXELS}x{PATCH_PIXELS} physical patches")
        if auxiliary.shape != inputs[:, :3].shape:
            raise ValueError("Auxiliary wind/cloud tensor must be Bx3xHxW")
        if observable.shape != inputs[:, :1].shape:
            raise ValueError("Observable tensor must be Bx1xHxW")
        if sensor_index.shape != (inputs.shape[0],):
            raise ValueError("Sensor index must have shape B")

        guide = self._guide(inputs, auxiliary, observable, sensor_index)
        target = self.encoder(inputs[:, TARGET_SLICE])
        reference90 = self.encoder(inputs[:, REFERENCE90_SLICE])
        reference365 = self.encoder(inputs[:, REFERENCE365_SLICE])
        fused = [
            fusion(target_value, ref90_value, ref365_value, guide)
            for fusion, target_value, ref90_value, ref365_value in zip(
                self.fusions, target, reference90, reference365
            )
        ]
        decoded = self.decoder4(fused[4], fused[3])
        decoded = self.decoder3(decoded, fused[2])
        decoded = self.decoder2(decoded, fused[1])
        decoded = self.decoder1(decoded, fused[0])
        segmentation_logits = self.segmentation(decoded)

        flat = segmentation_logits.flatten(1)
        valid = observable.flatten(1) > 0.5
        masked = flat.masked_fill(~valid, -20.0)
        top_count = max(1, int(masked.shape[1] * 0.01))
        top = torch.topk(masked, k=top_count, dim=1).values
        mask_scene_logit = 0.85 * top.mean(dim=1) + 0.15 * top.max(dim=1).values

        bottleneck = fused[-1]
        representation = torch.cat(
            (
                F.adaptive_avg_pool2d(bottleneck, 1).flatten(1),
                F.adaptive_max_pool2d(bottleneck, 1).flatten(1),
            ),
            dim=1,
        )
        context = torch.cat(
            (
                representation,
                guide.mean(dim=(-2, -1)),
                guide.amax(dim=(-2, -1)),
                guide.amin(dim=(-2, -1)),
            ),
            dim=1,
        )
        context_scene_logit = self.context_scene(context)[:, 0]
        scene_logit = (
            (1.0 - self.context_scene_weight) * mask_scene_logit
            + self.context_scene_weight * context_scene_logit
        )
        domain_logit = self.domain_head(gradient_reverse(representation, grl_strength))[:, 0]
        return {
            "segmentation_logits": segmentation_logits,
            "mask_scene_logit": mask_scene_logit,
            "context_scene_logit": context_scene_logit,
            "scene_logit": scene_logit,
            "domain_logit": domain_logit,
            "representation": representation,
        }

    def artifact_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "canonical_input_channels": list(V5_INPUT_CHANNELS),
            "mars_input_channels": list(INPUT_CHANNELS),
            "patch_pixels": PATCH_PIXELS,
            "physical_field_of_view_m": 640,
            "mars_source_resolution_m": 10,
            "methanes2cm_source_resolution_m": 20,
            "temporal_weight_sharing": True,
            "context_scene_weight": self.context_scene_weight,
            "scene_evidence_bound": SCENE_EVIDENCE_BOUND,
            "domain_adversary": "binary MARS-vs-MethaneS2CM on Sentinel-2 representations",
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }
