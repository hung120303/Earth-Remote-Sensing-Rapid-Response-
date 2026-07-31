from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_dofa_v2_base import CHECKPOINT_SHA256  # noqa: E402
from acquire_mars_metadata import sha256  # noqa: E402
from dofa_v2_backbone import vit_base_patch14  # noqa: E402
from extract_mars_dofa_v2_scene_features import (  # noqa: E402
    DOFA_MEAN,
    DOFA_STD,
    FEATURE_WIDTH,
    LANDSAT_WAVELENGTHS,
    MARS_TO_DOFA_MULTIPLIER,
    S2_WAVELENGTHS,
    build_dofa_frames,
    sensor_wavelengths,
    temporal_scene_features,
)


def test_dofa_frames_restore_physical_scale_and_order() -> None:
    inputs = torch.zeros((1, 16, 200, 200), dtype=torch.float32)
    for channel in range(6):
        inputs[:, 1 + channel] = 0.1 * (channel + 1)
        inputs[:, 7 + channel] = 0.7 + 0.1 * channel
    observable = torch.ones((1, 1, 200, 200), dtype=torch.float32)
    observable[:, :, 100:] = 0.0
    frames = build_dofa_frames({"inputs": inputs, "observable": observable})

    reference = torch.tensor([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    target = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    mean = torch.tensor(DOFA_MEAN)
    std = torch.tensor(DOFA_STD)
    assert frames.shape == (1, 2, 6, 224, 224)
    assert torch.allclose(
        frames[0, 0, :, 16, 16], (reference * MARS_TO_DOFA_MULTIPLIER - mean) / std
    )
    assert torch.allclose(
        frames[0, 1, :, 16, 16], (target * MARS_TO_DOFA_MULTIPLIER - mean) / std
    )
    assert torch.count_nonzero(frames[:, :, :, 160:]) == 0


def test_sensor_wavelength_contract_is_distinct() -> None:
    assert sensor_wavelengths(0) == S2_WAVELENGTHS
    assert sensor_wavelengths(1) == LANDSAT_WAVELENGTHS
    assert S2_WAVELENGTHS != LANDSAT_WAVELENGTHS


def test_temporal_scene_feature_schema_is_finite() -> None:
    reference = [torch.zeros((2, 768, 4, 4)) for _ in range(4)]
    target = [torch.randn((2, 768, 4, 4)) for _ in range(4)]
    features = temporal_scene_features(reference, target)
    assert features.shape == (2, FEATURE_WIDTH)
    assert torch.isfinite(features).all()


@pytest.mark.skipif(
    not (ROOT / ".research/foundation_models/dofa/checkpoints/dofav2_vit_base_e150.pth").exists(),
    reason="ignored DOFA-v2 checkpoint is not acquired",
)
def test_dofa_v2_checkpoint_loads_strictly() -> None:
    checkpoint = ROOT / ".research/foundation_models/dofa/checkpoints/dofav2_vit_base_e150.pth"
    assert sha256(checkpoint) == CHECKPOINT_SHA256
    model = vit_base_patch14()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    assert sum(parameter.numel() for parameter in model.parameters()) == 105_433_856
