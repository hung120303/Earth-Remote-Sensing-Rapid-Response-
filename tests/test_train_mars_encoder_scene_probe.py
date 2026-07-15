from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_mars_encoder_scene_probe import SceneProbe, select_features  # noqa: E402


class MarsEncoderSceneProbeTests(unittest.TestCase):
    def test_probe_forward_shape(self) -> None:
        model = SceneProbe(4, 3, 0.0)
        self.assertEqual(tuple(model(torch.ones((2, 4))).shape), (2,))

    def test_level5_feature_selection_appends_base(self) -> None:
        encoder = np.zeros((2, 1536), dtype=np.float16)
        base = np.ones((2, 2), dtype=np.float32)
        encoder_names = np.asarray([f"level5_{index}" for index in range(1536)])
        base_names = np.asarray(["base_a", "base_b"])
        values, names = select_features(
            encoder, base, encoder_names, base_names, "level5_plus_base"
        )
        self.assertEqual(values.shape, (2, 1538))
        self.assertEqual(names[-2:], ["base_a", "base_b"])


if __name__ == "__main__":
    unittest.main()
