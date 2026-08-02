"""Product-aware, multi-cohort Prithvi/UNet architecture for ERSRR v6.

The fixed input adapter converts every source to physical reflectance before a
learned, near-identity product correction.  Scene and dense paths deliberately
use separate encoders, LoRA adapters, product corrections, and embeddings so
their objectives can be optimized in disjoint phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from mars_prithvi_lora_model import encoder_tokens, inject_lora


CANONICAL_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
FRAME_NAMES = ("reference90", "reference365", "target")
PRODUCT_NAMES = (
    "sentinel2_l1c_mars",
    "landsat_l1_mars",
    "sentinel2_l2a_methanes2cm",
)
SENSOR_NAMES = ("Sentinel-2", "Landsat")
MARS_INPUT_CHANNELS = 16
METHANES2CM_V5_INPUT_CHANNELS = 20
MARS_PHYSICAL_SCALE = 0.5  # MARS loader stores raw DN / 5,000; physical is DN / 10,000.
PRITHVI_DN_SCALE = 10_000.0
PRITHVI_INPUT_SIZE = 128
DENSE_PHYSICS_CHANNELS = 47


@dataclass(frozen=True)
class CanonicalProductBatch:
    """Fixed physical contract shared by MARS-S2L and MethaneS2CM.

    ``frames`` is B x 3 x 6 x H x W in reference90, reference365,
    target order. ``reference_available`` is B x 2 and applies to the two
    reference slots. Missing references initially contain the target frame;
    the learned product harmonizer supplies a bounded imputation residual.
    """

    frames: torch.Tensor
    observable: torch.Tensor
    reference_available: torch.Tensor
    product_index: torch.Tensor
    sensor_index: torch.Tensor

    def validate(self) -> None:
        if self.frames.ndim != 5 or self.frames.shape[1:3] != (3, 6):
            raise ValueError("Canonical frames must have shape Bx3x6xHxW")
        batch, _, _, height, width = self.frames.shape
        if self.observable.shape != (batch, 1, height, width):
            raise ValueError("Observable mask does not match canonical frames")
        if self.reference_available.shape != (batch, 2):
            raise ValueError("Reference availability must have shape Bx2")
        if self.product_index.shape != (batch,) or self.sensor_index.shape != (batch,):
            raise ValueError("Product and sensor indices must have shape B")
        if torch.any((self.product_index < 0) | (self.product_index >= len(PRODUCT_NAMES))):
            raise ValueError("Product index is outside the v6 contract")
        if torch.any((self.sensor_index < 0) | (self.sensor_index >= len(SENSOR_NAMES))):
            raise ValueError("Sensor index is outside the v6 contract")
        if not torch.isfinite(self.frames).all():
            raise ValueError("Canonical frames contain non-finite values")


def _availability(
    values: torch.Tensor | None, batch: int, device: torch.device
) -> torch.Tensor:
    if values is None:
        return torch.ones(batch, dtype=torch.float32, device=device)
    result = values.to(device=device, dtype=torch.float32).reshape(-1)
    if result.shape != (batch,) or torch.any((result < 0.0) | (result > 1.0)):
        raise ValueError("Reference availability must be a B-vector in [0, 1]")
    return result


def canonicalize_mars(
    inputs: torch.Tensor,
    observable: torch.Tensor,
    sensor_index: torch.Tensor,
    *,
    reference90_available: torch.Tensor | None = None,
) -> CanonicalProductBatch:
    """Convert the released 16-channel MARS tensor to physical reflectance."""

    if inputs.ndim != 4 or inputs.shape[1] != MARS_INPUT_CHANNELS:
        raise ValueError("Expected Bx16xHxW MARS input")
    batch, _, height, width = inputs.shape
    if observable.shape != (batch, 1, height, width):
        raise ValueError("MARS observable mask does not match input")
    sensor = sensor_index.to(device=inputs.device, dtype=torch.long).reshape(-1)
    if sensor.shape != (batch,):
        raise ValueError("MARS sensor index must have shape B")
    available90 = _availability(reference90_available, batch, inputs.device)
    target = inputs[:, 1:7].float() * MARS_PHYSICAL_SCALE
    reference90 = inputs[:, 7:13].float() * MARS_PHYSICAL_SCALE
    available_map = available90[:, None, None, None]
    reference90 = available_map * reference90 + (1.0 - available_map) * target
    reference365 = target.clone()
    frames = torch.stack((reference90, reference365, target), dim=1)
    valid = (observable > 0.5).to(frames.dtype)
    frames = frames * valid[:, None]
    product = torch.where(sensor == 0, torch.zeros_like(sensor), torch.ones_like(sensor))
    batch_value = CanonicalProductBatch(
        frames=frames,
        observable=valid,
        reference_available=torch.stack((available90, torch.zeros_like(available90)), dim=1),
        product_index=product,
        sensor_index=sensor,
    )
    batch_value.validate()
    return batch_value


def canonicalize_methanes2cm(
    inputs: torch.Tensor,
    observable: torch.Tensor,
    *,
    reference90_available: torch.Tensor | None = None,
    reference365_available: torch.Tensor | None = None,
) -> CanonicalProductBatch:
    """Convert the 20-channel MethaneS2CM v5 tensor to the common contract."""

    if inputs.ndim != 4 or inputs.shape[1] != METHANES2CM_V5_INPUT_CHANNELS:
        raise ValueError("Expected Bx20xHxW MethaneS2CM v5 input")
    batch, _, height, width = inputs.shape
    if observable.shape != (batch, 1, height, width):
        raise ValueError("MethaneS2CM observable mask does not match input")
    target = inputs[:, 2:8].float()
    reference90 = inputs[:, 8:14].float()
    reference365 = inputs[:, 14:20].float()
    available90 = _availability(reference90_available, batch, inputs.device)
    available365 = _availability(reference365_available, batch, inputs.device)
    reference90 = available90[:, None, None, None] * reference90 + (
        1.0 - available90[:, None, None, None]
    ) * target
    reference365 = available365[:, None, None, None] * reference365 + (
        1.0 - available365[:, None, None, None]
    ) * target
    frames = torch.stack((reference90, reference365, target), dim=1)
    valid = (observable > 0.5).to(frames.dtype)
    frames = frames * valid[:, None]
    batch_value = CanonicalProductBatch(
        frames=frames,
        observable=valid,
        reference_available=torch.stack((available90, available365), dim=1),
        product_index=torch.full(
            (batch,), 2, dtype=torch.long, device=inputs.device
        ),
        sensor_index=torch.zeros(batch, dtype=torch.long, device=inputs.device),
    )
    batch_value.validate()
    return batch_value


class ProductAffineHarmonizer(nn.Module):
    """Near-identity radiometric correction plus learned missing-frame imputation."""

    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(len(PRODUCT_NAMES), len(CANONICAL_BANDS)))
        self.raw_bias = nn.Parameter(torch.zeros(len(PRODUCT_NAMES), len(CANONICAL_BANDS)))
        self.raw_missing_delta = nn.Parameter(
            torch.zeros(len(PRODUCT_NAMES), 2, len(CANONICAL_BANDS))
        )

    def forward(self, batch: CanonicalProductBatch) -> torch.Tensor:
        batch.validate()
        index = batch.product_index
        scale = torch.exp(0.15 * torch.tanh(self.log_scale[index]))[:, None, :, None, None]
        bias = 0.05 * torch.tanh(self.raw_bias[index])[:, None, :, None, None]
        frames = batch.frames * scale + bias
        target = frames[:, 2]
        references = frames[:, :2]
        delta = 0.10 * torch.tanh(self.raw_missing_delta[index])[:, :, :, None, None]
        availability = batch.reference_available[:, :, None, None, None]
        references = availability * references + (1.0 - availability) * (
            target[:, None] + delta
        )
        result = torch.cat((references, target[:, None]), dim=1)
        result = torch.clamp(result, min=0.0, max=2.0)
        return result * batch.observable[:, None]


def _normalized_ratio(values: torch.Tensor, observable: torch.Tensor) -> torch.Tensor:
    result = torch.ones_like(values[:, 0])
    valid = observable[:, 0] > 0.5
    for row in range(values.shape[0]):
        usable = valid[row] & (values[row, 4] > 1e-8) & torch.isfinite(values[row, 5])
        if not torch.any(usable):
            continue
        ratio = values[row, 5][usable] / values[row, 4][usable]
        median = torch.median(ratio)
        median = torch.where(torch.abs(median) >= 1e-8, median, torch.ones_like(median))
        normalized = torch.clamp(ratio / median, min=0.0, max=10.0)
        result[row][usable] = normalized
    return result


def mbmp_evidence(
    target: torch.Tensor, reference: torch.Tensor, observable: torch.Tensor
) -> torch.Tensor:
    target_ratio = _normalized_ratio(target, observable)
    reference_ratio = _normalized_ratio(reference, observable)
    valid = (observable[:, 0] > 0.5) & (reference_ratio > 1e-8)
    result = torch.ones_like(target_ratio)
    result[valid] = target_ratio[valid] / reference_ratio[valid]
    return torch.clamp(torch.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 10.0)[:, None]


def dense_physics_features(
    frames: torch.Tensor,
    observable: torch.Tensor,
    reference_available: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the explicit 47-channel dense contract and two MBMP guide maps."""

    reference90, reference365, target = frames.unbind(dim=1)
    difference90 = target - reference90
    difference365 = target - reference365
    mbmp90 = mbmp_evidence(target, reference90, observable)
    mbmp365 = mbmp_evidence(target, reference365, observable)
    height, width = target.shape[-2:]
    availability = reference_available[:, :, None, None].expand(-1, -1, height, width)
    features = torch.cat(
        (
            target,
            reference90,
            reference365,
            difference90,
            difference365,
            difference90.abs(),
            difference365.abs(),
            mbmp90 - 1.0,
            mbmp365 - 1.0,
            observable,
            availability,
        ),
        dim=1,
    )
    if features.shape[1] != DENSE_PHYSICS_CHANNELS:
        raise RuntimeError("Dense v6 channel contract changed unexpectedly")
    return features, torch.cat((mbmp90 - 1.0, mbmp365 - 1.0), dim=1)


class PrithviPairEncoder(nn.Module):
    """LoRA-adapted Prithvi encoder returning target-grid and CLS features."""

    def __init__(self, foundation: nn.Module, spec: dict[str, Any]) -> None:
        super().__init__()
        for parameter in foundation.parameters():
            parameter.requires_grad_(False)
        self.adapted_modules = inject_lora(
            foundation,
            last_blocks=int(spec["last_blocks"]),
            rank=int(spec["rank"]),
            alpha=float(spec["alpha"]),
            dropout=float(spec["lora_dropout"]),
        )
        # The MAE decoder is not used downstream. Retaining only the encoder
        # avoids placing roughly 27M permanently frozen parameters per branch
        # on the training device.
        self.encoder = foundation.encoder
        self.width = int(self.encoder.embed_dim)

    def forward(
        self, values: torch.Tensor, temporal: torch.Tensor, location: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        tokens = encoder_tokens(self.encoder, values, temporal, location)
        patches = tokens[:, 1:]
        if patches.shape[1] % 2:
            raise ValueError("Prithvi pair tokens are not divisible into two frames")
        target = patches.reshape(
            patches.shape[0], 2, patches.shape[1] // 2, patches.shape[2]
        )[:, 1]
        grid = int(target.shape[1] ** 0.5)
        if grid * grid != target.shape[1]:
            raise ValueError("Prithvi target tokens do not form a square grid")
        return {
            "cls": tokens[:, 0],
            "target_grid": target.transpose(1, 2).reshape(-1, self.width, grid, grid),
        }


class ConvGNBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class PhysicsGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.residual = nn.Conv2d(2, channels, 1)
        self.gate = nn.Conv2d(2, channels, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, values: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        local = F.interpolate(evidence, values.shape[-2:], mode="bilinear", align_corners=False)
        return values + torch.sigmoid(self.gate(local)) * self.residual(local)


class DecoderStage(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = ConvGNBlock(input_channels + skip_channels, output_channels)
        self.physics = PhysicsGate(output_channels)

    def forward(
        self, values: torch.Tensor, skip: torch.Tensor, evidence: torch.Tensor
    ) -> torch.Tensor:
        values = F.interpolate(values, skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.physics(self.block(torch.cat((values, skip), dim=1)), evidence)


class ProductHarmonizedMultiCohortV6(nn.Module):
    """Dual-adapter Prithvi v6 with decoupled scene and dense optimization."""

    def __init__(
        self,
        scene_encoder: nn.Module,
        dense_encoder: nn.Module,
        prithvi_mean: torch.Tensor,
        prithvi_std: torch.Tensor,
        *,
        product_embedding_dim: int = 8,
        sensor_embedding_dim: int = 4,
        scene_hidden: int = 256,
        scene_topk: int = 4,
    ) -> None:
        super().__init__()
        if not hasattr(scene_encoder, "width") or not hasattr(dense_encoder, "width"):
            raise TypeError("Pair encoders must expose a width attribute")
        self.scene_encoder = scene_encoder
        self.dense_encoder = dense_encoder
        self.scene_width = int(scene_encoder.width)
        self.dense_width = int(dense_encoder.width)
        self.scene_harmonizer = ProductAffineHarmonizer()
        self.dense_harmonizer = ProductAffineHarmonizer()
        self.scene_product_embedding = nn.Embedding(len(PRODUCT_NAMES), product_embedding_dim)
        self.scene_sensor_embedding = nn.Embedding(len(SENSOR_NAMES), sensor_embedding_dim)
        self.dense_product_embedding = nn.Embedding(len(PRODUCT_NAMES), product_embedding_dim)
        self.dense_sensor_embedding = nn.Embedding(len(SENSOR_NAMES), sensor_embedding_dim)
        mean = torch.as_tensor(prithvi_mean, dtype=torch.float32).reshape(1, 6, 1, 1, 1)
        std = torch.as_tensor(prithvi_std, dtype=torch.float32).reshape(1, 6, 1, 1, 1)
        if torch.any(std <= 0):
            raise ValueError("Prithvi standard deviations must be positive")
        self.register_buffer("prithvi_mean", mean)
        self.register_buffer("prithvi_std", std)
        dense_input = DENSE_PHYSICS_CHANNELS + product_embedding_dim + sensor_embedding_dim
        self.stem = ConvGNBlock(dense_input, 32)
        self.down1 = ConvGNBlock(32, 48, stride=2)
        self.down2 = ConvGNBlock(48, 80, stride=2)
        self.down3 = ConvGNBlock(80, 128, stride=2)
        self.down4 = ConvGNBlock(128, 160, stride=2)
        self.token_projection = nn.Conv2d(self.dense_width * 3, 128, 1)
        self.bottleneck = ConvGNBlock(288, 192)
        self.up4 = DecoderStage(192, 128, 128)
        self.up3 = DecoderStage(128, 80, 80)
        self.up2 = DecoderStage(80, 48, 48)
        self.up1 = DecoderStage(48, 32, 32)
        self.dense_head = nn.Conv2d(32, 1, 1)
        self.scene_patch_head = nn.Conv2d(self.scene_width, 1, 1)
        self.scene_topk = int(scene_topk)
        scene_input = self.scene_width * 6 + product_embedding_dim + sensor_embedding_dim + 2
        self.scene_head = nn.Sequential(
            nn.LayerNorm(scene_input),
            nn.Linear(scene_input, scene_hidden),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(scene_hidden, 1),
        )
        # Preserve the initialization-time trainability contract. Prithvi base
        # weights are already frozen; only LoRA residuals belong to a phase.
        initially_trainable = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        self._scene_phase_names = frozenset(
            name for name in initially_trainable if name.startswith("scene_")
        )
        self._dense_phase_names = frozenset(
            name
            for name in initially_trainable
            if name.startswith("dense_")
            or name.startswith(("stem", "down", "token_projection", "bottleneck", "up"))
        )

    def _pair_input(
        self, reference: torch.Tensor, target: torch.Tensor, observable: torch.Tensor
    ) -> torch.Tensor:
        pair = torch.stack((reference, target), dim=2)
        batch = pair.shape[0]
        resized = F.interpolate(
            pair.permute(0, 2, 1, 3, 4).flatten(0, 1),
            size=(PRITHVI_INPUT_SIZE, PRITHVI_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, 2, 6, PRITHVI_INPUT_SIZE, PRITHVI_INPUT_SIZE).permute(0, 2, 1, 3, 4)
        valid = F.interpolate(
            observable.float(),
            size=(PRITHVI_INPUT_SIZE, PRITHVI_INPUT_SIZE),
            mode="nearest",
        )[:, :, None]
        normalized = (resized * PRITHVI_DN_SCALE - self.prithvi_mean) / self.prithvi_std
        return normalized.masked_fill(valid <= 0.5, 0.0)

    def _encode_pairs(
        self,
        encoder: nn.Module,
        frames: torch.Tensor,
        observable: torch.Tensor,
        temporal: torch.Tensor,
        location: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if temporal.ndim != 3 or temporal.shape[1:] != (3, 2):
            raise ValueError("Temporal coordinates must have shape Bx3x2")
        target = frames[:, 2]
        pair90 = self._pair_input(frames[:, 0], target, observable)
        pair365 = self._pair_input(frames[:, 1], target, observable)
        encoded90 = encoder(pair90, temporal[:, (0, 2)], location)
        encoded365 = encoder(pair365, temporal[:, (1, 2)], location)
        return encoded90, encoded365

    def _scene_pool(self, encoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        grid = encoded["target_grid"]
        logits = self.scene_patch_head(grid).flatten(1)
        patches = grid.flatten(2).transpose(1, 2)
        top = torch.topk(logits, min(self.scene_topk, logits.shape[1]), dim=1).indices
        selected = torch.gather(
            patches, 1, top[:, :, None].expand(-1, -1, patches.shape[-1])
        ).mean(dim=1)
        return patches.mean(dim=1), selected

    def _scene_forward(
        self,
        batch: CanonicalProductBatch,
        temporal: torch.Tensor,
        location: torch.Tensor,
    ) -> torch.Tensor:
        frames = self.scene_harmonizer(batch)
        encoded90, encoded365 = self._encode_pairs(
            self.scene_encoder, frames, batch.observable, temporal, location
        )
        mean90, top90 = self._scene_pool(encoded90)
        mean365, top365 = self._scene_pool(encoded365)
        features = torch.cat(
            (
                encoded90["cls"],
                encoded365["cls"],
                mean90,
                mean365,
                top90,
                top365,
                self.scene_product_embedding(batch.product_index),
                self.scene_sensor_embedding(batch.sensor_index),
                batch.reference_available,
            ),
            dim=1,
        )
        return self.scene_head(features).squeeze(1)

    def _dense_forward(
        self,
        batch: CanonicalProductBatch,
        temporal: torch.Tensor,
        location: torch.Tensor,
    ) -> torch.Tensor:
        frames = self.dense_harmonizer(batch)
        physics, evidence = dense_physics_features(
            frames, batch.observable, batch.reference_available
        )
        height, width = physics.shape[-2:]
        product = self.dense_product_embedding(batch.product_index)[:, :, None, None].expand(
            -1, -1, height, width
        )
        sensor = self.dense_sensor_embedding(batch.sensor_index)[:, :, None, None].expand(
            -1, -1, height, width
        )
        stem = self.stem(torch.cat((physics, product, sensor), dim=1))
        level1 = self.down1(stem)
        level2 = self.down2(level1)
        level3 = self.down3(level2)
        level4 = self.down4(level3)
        encoded90, encoded365 = self._encode_pairs(
            self.dense_encoder, frames, batch.observable, temporal, location
        )
        grid90 = encoded90["target_grid"]
        grid365 = encoded365["target_grid"]
        tokens = self.token_projection(
            torch.cat((grid90, grid365, (grid90 - grid365).abs()), dim=1)
        )
        tokens = F.interpolate(tokens, level4.shape[-2:], mode="bilinear", align_corners=False)
        values = self.bottleneck(torch.cat((level4, tokens), dim=1))
        values = self.up4(values, level3, evidence)
        values = self.up3(values, level2, evidence)
        values = self.up2(values, level1, evidence)
        values = self.up1(values, stem, evidence)
        return self.dense_head(values)

    def forward(
        self,
        batch: CanonicalProductBatch,
        temporal: torch.Tensor,
        location: torch.Tensor,
        *,
        branch: Literal["scene", "dense", "both"] = "both",
    ) -> dict[str, torch.Tensor]:
        batch.validate()
        result: dict[str, torch.Tensor] = {}
        if branch in {"scene", "both"}:
            result["scene_logit"] = self._scene_forward(batch, temporal, location)
        if branch in {"dense", "both"}:
            result["dense_logits"] = self._dense_forward(batch, temporal, location)
        if not result:
            raise ValueError(f"Unsupported v6 branch: {branch}")
        return result

    @staticmethod
    def protected_scene_score(
        baseline_score: torch.Tensor,
        residual_logit: torch.Tensor,
        *,
        strength: float,
        protection_gate: float,
    ) -> torch.Tensor:
        gate = float(protection_gate)
        if not 0.0 <= gate < 1.0:
            raise ValueError("Protection gate must lie in [0, 1)")
        eligible = baseline_score >= gate
        local = ((baseline_score - gate) / (1.0 - gate)).clamp(1e-6, 1.0 - 1e-6)
        corrected = torch.sigmoid(
            torch.logit(local) + float(strength) * 2.0 * torch.tanh(residual_logit / 2.0)
        )
        candidate = gate + (1.0 - gate) * corrected
        return torch.where(eligible, candidate, baseline_score)

    def set_trainable_phase(self, phase: Literal["scene", "dense", "all", "frozen"]) -> None:
        if phase not in {"scene", "dense", "all", "frozen"}:
            raise ValueError(f"Unsupported trainable phase: {phase}")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                (phase in {"scene", "all"} and name in self._scene_phase_names)
                or (phase in {"dense", "all"} and name in self._dense_phase_names)
            )
