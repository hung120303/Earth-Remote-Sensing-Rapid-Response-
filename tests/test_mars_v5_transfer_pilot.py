from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_mars_v5_transfer_pilot import INPUT_SIZE, resize_batch  # noqa: E402


def test_resize_batch_duplicates_single_mars_reference() -> None:
    inputs = torch.zeros(2, 16, 32, 32)
    for channel in range(16):
        inputs[:, channel] = float(channel)
    batch = {
        "inputs": inputs,
        "observable": torch.ones(2, 1, 32, 32),
        "mask": torch.zeros(2, 1, 32, 32),
    }
    result = resize_batch(batch)
    assert result["inputs"].shape == (2, 20, INPUT_SIZE, INPUT_SIZE)
    assert torch.equal(result["inputs"][:, 0], result["inputs"][:, 1])
    assert torch.equal(result["inputs"][:, 8:14], result["inputs"][:, 14:20])
    assert result["observable"].shape[-2:] == (INPUT_SIZE, INPUT_SIZE)
    assert result["mask"].shape[-2:] == (INPUT_SIZE, INPUT_SIZE)
