from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from extract_mars_encoder_scene_features import masked_channel_moments  # noqa: E402


class MarsEncoderSceneFeatureTests(unittest.TestCase):
    def test_masked_channel_moments_ignore_unobservable_pixels(self) -> None:
        values = torch.tensor([[[[1.0, 3.0], [100.0, 100.0]]]])
        observable = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        actual = masked_channel_moments(values, observable)
        torch.testing.assert_close(actual, torch.tensor([[2.0, 1.0, 3.0]]))

    def test_empty_observable_mask_is_finite(self) -> None:
        values = torch.ones((1, 2, 2, 2))
        observable = torch.zeros((1, 1, 2, 2))
        actual = masked_channel_moments(values, observable)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, torch.zeros_like(actual))


if __name__ == "__main__":
    unittest.main()
