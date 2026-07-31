from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from extract_mars_prithvi_physical_scene_features import (  # noqa: E402
    MARS_TO_PRITHVI_MULTIPLIER,
    build_physical_input,
)
from train_mars_prithvi_physical_scene_probe import select_features  # noqa: E402


def test_physical_input_restores_raw_dn_and_chronological_order() -> None:
    inputs = torch.zeros((1, 16, 200, 200), dtype=torch.float32)
    for channel in range(6):
        inputs[:, 1 + channel] = 0.1 * (channel + 1)  # target
        inputs[:, 7 + channel] = 0.7 + 0.1 * channel  # reference
    observable = torch.ones((1, 1, 200, 200), dtype=torch.float32)
    observable[:, :, 100:] = 0.0
    mean = torch.zeros((1, 6, 1, 1, 1), dtype=torch.float32)
    std = torch.ones_like(mean)

    output = build_physical_input(
        {"inputs": inputs, "observable": observable}, mean, std
    )

    assert MARS_TO_PRITHVI_MULTIPLIER == 5_000.0
    assert output.shape == (1, 6, 2, 128, 128)
    expected_reference = torch.tensor([0.7, 0.8, 0.9, 1.0, 1.1, 1.2]) * 5_000
    expected_target = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) * 5_000
    assert torch.allclose(output[0, :, 0, 16, 16], expected_reference)
    assert torch.allclose(output[0, :, 1, 16, 16], expected_target)
    assert torch.count_nonzero(output[:, :, :, 96:]) == 0


def test_physical_probe_feature_slices_match_frozen_schema() -> None:
    encoded = np.zeros((2, 3072), dtype=np.float32)
    names = np.asarray([f"feature_{index}" for index in range(3072)])

    cls, cls_names = select_features(encoded, names, "cls")
    change, change_names = select_features(encoded, names, "temporal_change")
    all_features, all_names = select_features(encoded, names, "all")

    assert cls.shape == (2, 768) and cls_names.shape == (768,)
    assert change.shape == (2, 1152) and change_names.shape == (1152,)
    assert all_features.shape == (2, 3072) and all_names.shape == (3072,)
