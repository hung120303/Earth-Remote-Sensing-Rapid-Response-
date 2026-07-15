from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from extract_mars_spatial_scene_inputs import (  # noqa: E402
    CHANNEL_NAMES,
    spatial_scene_channels,
)


class MarsSpatialSceneInputTests(unittest.TestCase):
    def test_spatial_scene_channels_preserve_frozen_order(self) -> None:
        inputs = torch.zeros((1, 16, 4, 4))
        inputs[:, 0] = 1.25
        inputs[:, 5] = 0.4
        inputs[:, 11] = 0.1
        inputs[:, 6] = 0.5
        inputs[:, 12] = 0.2
        logits = torch.zeros((1, 1, 4, 4))
        observable = torch.ones((1, 1, 4, 4))
        actual = spatial_scene_channels(inputs, logits, observable, output_size=2)
        self.assertEqual(actual.shape, (1, len(CHANNEL_NAMES), 2, 2))
        torch.testing.assert_close(actual[:, 0], torch.full((1, 2, 2), 0.5))
        torch.testing.assert_close(actual[:, 1], torch.full((1, 2, 2), 0.5))
        torch.testing.assert_close(actual[:, 2], torch.full((1, 2, 2), 0.25))
        torch.testing.assert_close(actual[:, 3], torch.full((1, 2, 2), 0.3))
        torch.testing.assert_close(actual[:, 4], torch.full((1, 2, 2), 0.3))

    def test_misaligned_logits_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            spatial_scene_channels(
                torch.zeros((1, 16, 4, 4)),
                torch.zeros((1, 1, 2, 2)),
                torch.ones((1, 1, 4, 4)),
            )


if __name__ == "__main__":
    unittest.main()
