from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_mars_spatial_scene_classifier import (  # noqa: E402
    SpatialSceneClassifier,
    augment_batch,
    channel_indices,
)


class MarsSpatialSceneClassifierTests(unittest.TestCase):
    def test_classifier_forward_shape(self) -> None:
        model = SpatialSceneClassifier(9, dropout=0.0)
        output = model(torch.zeros((2, 9, 64, 64)), torch.tensor([0, 1]))
        self.assertEqual(tuple(output.shape), (2,))

    def test_probability_channel_subset_is_frozen(self) -> None:
        self.assertEqual(channel_indices("probability_spatial"), (0, 1, 7, 8))

    def test_augmentation_preserves_shape(self) -> None:
        generator = torch.Generator().manual_seed(1)
        values = torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
        self.assertEqual(augment_batch(values, generator).shape, values.shape)


if __name__ == "__main__":
    unittest.main()
