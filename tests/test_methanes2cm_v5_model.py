from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODULE) not in sys.path:
    sys.path.insert(0, str(MODULE))

from methanes2cm_adapter import V5_INPUT_CHANNELS  # noqa: E402
from methanes2cm_v5_model import MethaneS2CMV5Model  # noqa: E402


class MethaneS2CMV5ModelTests(unittest.TestCase):
    def test_forward_contract_and_segmentation_derived_presence(self) -> None:
        torch.manual_seed(7)
        model = MethaneS2CMV5Model()
        inputs = torch.rand(2, len(V5_INPUT_CHANNELS), 32, 32)
        inputs[:, :2] += 0.5
        observable = torch.ones(2, 1, 32, 32)
        output = model(inputs, observable)
        self.assertEqual(output["segmentation_logits"].shape, (2, 1, 32, 32))
        self.assertEqual(output["scene_logit"].shape, (2,))
        expected_count = max(1, int(32 * 32 * model.scene_topk_fraction))
        logits = output["segmentation_logits"].flatten(1)
        top = torch.topk(logits, expected_count, dim=1).values
        expected = (
            (1.0 - model.scene_max_weight) * top.mean(dim=1)
            + model.scene_max_weight * top.max(dim=1).values
        )
        torch.testing.assert_close(output["scene_logit"], expected)

    def test_rejects_wrong_channel_contract(self) -> None:
        model = MethaneS2CMV5Model()
        with self.assertRaisesRegex(ValueError, "Expected"):
            model(torch.zeros(1, 19, 32, 32), torch.ones(1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
