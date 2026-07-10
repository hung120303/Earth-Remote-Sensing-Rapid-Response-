from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_v3_model import INPUT_CHANNELS, MarsV3Model, soft_plume_geometry  # noqa: E402


class MarsV3ModelTests(unittest.TestCase):
    def test_forward_contract_and_backward(self) -> None:
        torch.manual_seed(7)
        model = MarsV3Model(topk_fraction=0.02)
        inputs = torch.randn(2, len(INPUT_CHANNELS), 32, 32, requires_grad=True)
        observable = torch.ones(2, 1, 32, 32)
        output = model(inputs, observable)
        self.assertEqual(output["segmentation_logits"].shape, (2, 1, 32, 32))
        self.assertEqual(output["presence_logit"].shape, (2,))
        self.assertEqual(output["quality_logit"].shape, (2,))
        self.assertEqual(output["proposal_descriptor"].shape, (2, 1294))
        self.assertEqual(output["soft_geometry"].shape, (2, 12))
        loss = (
            output["segmentation_logits"].mean()
            + output["presence_logit"].mean()
            + output["quality_logit"].mean()
        )
        loss.backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        metadata = model.artifact_metadata()
        self.assertEqual(metadata["initialization"], "from_scratch_required_for_primary_experiment")
        self.assertGreater(metadata["parameter_count"], 13_000_000)

    def test_geometry_responds_to_wind_alignment(self) -> None:
        logits = torch.full((1, 1, 32, 32), -8.0)
        logits[:, :, 14:18, 4:28] = 8.0
        observable = torch.ones_like(logits)
        east_wind = torch.tensor([[1.0, 0.0]])
        north_wind = torch.tensor([[0.0, 1.0]])
        east_geometry = soft_plume_geometry(logits, observable, east_wind)
        north_geometry = soft_plume_geometry(logits, observable, north_wind)
        self.assertGreater(float(east_geometry[0, 8]), 0.0)
        self.assertLess(float(north_geometry[0, 8]), 0.0)

    def test_rejects_wrong_input_contract(self) -> None:
        model = MarsV3Model()
        with self.assertRaisesRegex(ValueError, "Expected"):
            model(torch.zeros(1, 15, 32, 32), torch.ones(1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
