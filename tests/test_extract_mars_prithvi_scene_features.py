from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from extract_mars_prithvi_scene_features import (  # noqa: E402
    build_features,
    build_input,
    date_coordinate,
    feature_names,
    reference_date_coordinate,
    scene_date_coordinate,
)
from train_mars_prithvi_scene_probe import (  # noqa: E402
    candidate_specs,
    select_features,
    spec_key,
)


class PrithviFeatureExtractionTests(unittest.TestCase):
    def test_temporal_coordinates_and_missing_reference_policy(self) -> None:
        self.assertEqual(date_coordinate("2024-02-29T12:00:00+00:00"), (2024, 60))
        self.assertEqual(
            scene_date_coordinate("S2A_MSIL1C_20220714T100041_example"), (2022, 195)
        )
        record = {
            "reference_scene_id": "",
            "target_datetime": "2023-10-03 16:34:58.952000+00:00",
        }
        self.assertEqual(reference_date_coordinate(record), (2023, 276))
        with self.assertRaisesRegex(ValueError, "Cannot parse reference date"):
            scene_date_coordinate("missing-date")

    def test_build_input_uses_chronological_six_band_frames(self) -> None:
        inputs = torch.zeros((1, 16, 4, 4), dtype=torch.float32)
        for channel in range(16):
            inputs[:, channel] = float(channel)
        batch = {
            "inputs": inputs,
            "observable": torch.ones((1, 1, 4, 4), dtype=torch.float32),
        }
        mean = torch.zeros((1, 6, 1, 1, 1), dtype=torch.float32)
        std = torch.full((1, 6, 1, 1, 1), 10_000.0, dtype=torch.float32)
        result = build_input(batch, mean, std)
        self.assertEqual(result.shape, (1, 6, 2, 128, 128))
        self.assertTrue(torch.allclose(result[0, :, 0, 0, 0], torch.arange(7, 13).float()))
        self.assertTrue(torch.allclose(result[0, :, 1, 0, 0], torch.arange(1, 7).float()))

    def test_build_features_has_frozen_3072_value_schema(self) -> None:
        outputs = [torch.zeros((1, 5, 192), dtype=torch.float32) for _ in range(12)]
        for block in (3, 6, 9, 12):
            outputs[block - 1][:, 0] = float(block)
        outputs[-1][0, 1:, 0] = torch.tensor([1.0, 3.0, 5.0, 9.0])
        values = build_features(outputs)
        names = feature_names()
        self.assertEqual(values.shape, (1, 3072))
        self.assertEqual(len(names), 3072)
        self.assertEqual(names[0], "prithvi_block3_cls_0")
        self.assertEqual(names[767], "prithvi_block12_cls_191")
        self.assertAlmostEqual(values[0, 0].item(), 3.0)
        reference = 768
        target = reference + 3 * 192
        difference = target + 3 * 192
        absolute = difference + 3 * 192
        self.assertAlmostEqual(values[0, reference].item(), 2.0)
        self.assertAlmostEqual(values[0, reference + 192].item(), 1.0)
        self.assertAlmostEqual(values[0, reference + 384].item(), 3.0)
        self.assertAlmostEqual(values[0, target].item(), 7.0)
        self.assertAlmostEqual(values[0, target + 192].item(), 2.0)
        self.assertAlmostEqual(values[0, difference].item(), 5.0)
        self.assertAlmostEqual(values[0, difference + 384].item(), 6.0)
        self.assertAlmostEqual(values[0, absolute].item(), 5.0)

    def test_temporal_probe_slice_and_candidate_grid_are_frozen(self) -> None:
        encoder = np.zeros((2, 3072), dtype=np.float16)
        base = np.zeros((2, 3), dtype=np.float32)
        values, names = select_features(
            encoder,
            base,
            np.asarray(feature_names()),
            np.asarray(["base_a", "base_b", "base_c"]),
            "temporal_change_plus_base",
        )
        self.assertEqual(values.shape, (2, 1155))
        self.assertEqual(names[0], "prithvi_target_minus_reference_mean_0")
        self.assertEqual(names[-4], "prithvi_absolute_difference_max_191")
        self.assertEqual(names[-3:], ["base_a", "base_b", "base_c"])
        specs = candidate_specs()
        self.assertEqual(len(specs), 14)
        self.assertEqual(len({spec_key(spec) for spec in specs}), 14)


if __name__ == "__main__":
    unittest.main()
