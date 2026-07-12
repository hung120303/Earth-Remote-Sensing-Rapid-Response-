from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from external_emit_adapter import build_external_inputs
from mars_v3_model import INPUT_CHANNELS


class ExternalEmitAdapterTests(unittest.TestCase):
    def test_build_external_inputs_matches_v3_scaling_and_channels(self) -> None:
        target = np.full((6, 4, 5), 5000, dtype=np.uint16)
        reference = np.full((6, 4, 5), 2500, dtype=np.uint16)
        cloud = np.zeros((4, 5), dtype=np.uint8)
        cloud[0, 0] = 2
        inputs, observable = build_external_inputs(target, reference, cloud, (8.0, -4.0))
        self.assertEqual(inputs.shape, (len(INPUT_CHANNELS), 4, 5))
        self.assertTrue(np.allclose(inputs[1:7], 1.0))
        self.assertTrue(np.allclose(inputs[7:13], 0.5))
        self.assertTrue(np.allclose(inputs[13], 1.0))
        self.assertTrue(np.allclose(inputs[14], -0.5))
        self.assertEqual(inputs[15, 0, 0], 1.0)
        self.assertFalse(observable[0, 0])
        self.assertTrue(observable[1, 1])

    def test_build_external_inputs_marks_radiometric_nodata_unobservable(self) -> None:
        target = np.full((6, 3, 3), 1000, dtype=np.uint16)
        reference = target.copy()
        reference[4, 1, 2] = 0
        inputs, observable = build_external_inputs(
            target, reference, np.zeros((3, 3), dtype=np.uint8), (0.0, 0.0)
        )
        self.assertEqual(inputs.shape[0], len(INPUT_CHANNELS))
        self.assertFalse(observable[1, 2])

    def test_build_external_inputs_rejects_unknown_cloud_class(self) -> None:
        values = np.ones((6, 2, 2), dtype=np.uint16)
        with self.assertRaises(ValueError):
            build_external_inputs(values, values, np.full((2, 2), 9), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
