from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from extract_mars_scene_features import (  # noqa: E402
    pooled_scene_features,
    tensor_feature_names,
)


class SceneFeatureTests(unittest.TestCase):
    def test_schema_width_and_finiteness(self) -> None:
        torch.manual_seed(9)
        inputs = torch.rand(2, 16, 64, 64)
        primary = torch.randn(2, 1, 64, 64)
        released = torch.randn(2, 1, 64, 64)
        clear = torch.ones(2, 1, 64, 64)
        observable = torch.ones_like(clear)
        observable[0, :, :8, :] = 0
        values = pooled_scene_features(inputs, primary, released, clear, observable)
        self.assertEqual(values.shape, (2, len(tensor_feature_names())))
        self.assertTrue(torch.isfinite(values).all())

    def test_all_invalid_scene_remains_finite(self) -> None:
        inputs = torch.zeros(1, 16, 64, 64)
        logits = torch.zeros(1, 1, 64, 64)
        invalid = torch.zeros(1, 1, 64, 64)
        values = pooled_scene_features(inputs, logits, logits, invalid, invalid)
        self.assertTrue(torch.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
