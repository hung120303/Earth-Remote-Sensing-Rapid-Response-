from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_v4_model import INPUT_CHANNELS, MarsV4Model  # noqa: E402
from mars_v4_simulation import (  # noqa: E402
    MarsPlumeSimulator,
    MarsTransmittanceLut,
    injection_slices,
)

LUT = ROOT / "configs" / "mars_s2_integrated_transmittances.json"


class MarsV4ModelTests(unittest.TestCase):
    def test_temporal_siamese_forward_and_backward_contract(self) -> None:
        torch.manual_seed(7)
        model = MarsV4Model()
        inputs = torch.randn(2, len(INPUT_CHANNELS), 32, 32, requires_grad=True)
        observable = torch.ones(2, 1, 32, 32)
        output = model(inputs, observable)
        self.assertEqual(output["segmentation_logits"].shape, (2, 1, 32, 32))
        self.assertEqual(output["scene_logit"].shape, (2,))
        self.assertEqual(output["dense_features"].shape, (2, 64, 32, 32))
        (output["segmentation_logits"].mean() + output["scene_logit"].mean()).backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        metadata = model.artifact_metadata()
        self.assertTrue(metadata["temporal_weight_sharing"])
        self.assertGreater(metadata["parameter_count"], 5_000_000)

    def test_scene_score_ignores_unobservable_hot_pixels(self) -> None:
        model = MarsV4Model()
        inputs = torch.zeros(1, len(INPUT_CHANNELS), 32, 32)
        observable = torch.ones(1, 1, 32, 32)
        observable[:, :, :16] = 0
        output = model(inputs, observable)
        self.assertTrue(torch.isfinite(output["scene_logit"]).all())

    def test_released_lut_matches_upstream_reference_values(self) -> None:
        lut = MarsTransmittanceLut(LUT)
        enhancement = np.asarray([[0.0, 500.0, 2000.0]], dtype=np.float32)
        b12, b11 = lut.transmittance("S2B", 30.0, 5.0, enhancement)
        np.testing.assert_allclose(
            b12,
            [[0.9999999991914, 0.9909898492251972, 0.967069603697185]],
            rtol=0,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            b11,
            [[0.9999999950285126, 0.9982352115841717, 0.9935992330512269]],
            rtol=0,
            atol=2e-7,
        )

    def test_simulation_attenuates_b12_more_than_b11(self) -> None:
        simulator = MarsPlumeSimulator(LUT, padding=4)
        target = np.full((6, 32, 32), 5000, dtype=np.uint16)
        ch4 = np.zeros((12, 12), dtype=np.float32)
        mask = np.zeros_like(ch4, dtype=bool)
        mask[4:8, 2:10] = True
        ch4[mask] = 2000.0
        result = simulator.simulate(
            target,
            ch4,
            mask,
            source_wind=(2.0, 0.0),
            target_wind=(2.0, 0.0),
            satellite="S2A",
            solar_zenith_degrees=30.0,
            view_zenith_degrees=5.0,
            rng=np.random.default_rng(17),
        )
        self.assertTrue(np.any(result.mask))
        self.assertLess(float(np.mean(result.target[5][result.mask])), 5000.0)
        self.assertLess(
            float(np.mean(result.target[5][result.mask])),
            float(np.mean(result.target[4][result.mask])),
        )
        np.testing.assert_array_equal(result.target[:4], target[:4])

    def test_injection_slices_preserve_matching_shapes(self) -> None:
        image, plume = injection_slices((10, 10), (8, 8), (-3, 7))
        self.assertEqual(image[0].stop - image[0].start, plume[0].stop - plume[0].start)
        self.assertEqual(image[1].stop - image[1].start, plume[1].stop - plume[1].start)


if __name__ == "__main__":
    unittest.main()
