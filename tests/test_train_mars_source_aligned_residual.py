from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
if str(ROOT / "EarthRemoteSensingRapidResponse") not in sys.path:
    sys.path.insert(0, str(ROOT / "EarthRemoteSensingRapidResponse"))

from mars_paper_model import MarsPaperResidualModel  # noqa: E402
from train_mars_source_aligned_residual import (  # noqa: E402
    contract_residual_strength,
    source_aligned_loss,
    source_aligned_sampling_weights,
)


class MarsSourceAlignedResidualTests(unittest.TestCase):
    def test_sampling_balances_site_label_sensor_cells(self) -> None:
        records = [
            {"group_id": "a", "label_state": "PLUME", "sensor_family": "Sentinel-2"},
            {"group_id": "a", "label_state": "PLUME", "sensor_family": "Sentinel-2"},
            {"group_id": "a", "label_state": "NO_PLUME", "sensor_family": "Sentinel-2"},
        ]
        self.assertEqual(source_aligned_sampling_weights(records).tolist(), [0.5, 0.5, 1.0])

    def test_strength_contraction_matches_logit_interpolation(self) -> None:
        torch.manual_seed(4)
        original = MarsPaperResidualModel().eval()
        original.correction.output.weight.data.normal_(std=0.01)
        original.correction.output.bias.data.fill_(0.02)
        original.sensor_log_scale.data.copy_(torch.tensor([0.04, -0.03]))
        original.sensor_bias.data.copy_(torch.tensor([0.01, -0.02]))
        contracted = copy.deepcopy(original)
        contract_residual_strength(contracted, 0.5)
        values = torch.rand(2, 16, 32, 32)
        observable = torch.ones(2, 1, 32, 32)
        sensors = torch.tensor([0, 1])
        with torch.no_grad():
            before = original(values, observable, sensors)
            after = contracted(values, observable, sensors)
        expected = before["baseline_logits"] + 0.5 * (
            before["segmentation_logits"] - before["baseline_logits"]
        )
        torch.testing.assert_close(after["segmentation_logits"], expected, rtol=1e-5, atol=1e-6)

    def test_ch4_weighted_loss_is_finite(self) -> None:
        logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
        target = torch.zeros_like(logits)
        target[0, :, 2:4, 2:4] = 1
        output = {
            "segmentation_logits": logits,
            "baseline_logits": torch.zeros_like(logits),
            "correction_logits": logits,
            "scene_logit": torch.zeros(2, requires_grad=True),
        }
        batch = {
            "mask": target,
            "enhancement": target * 500,
            "observable": torch.ones_like(target),
            "presence": torch.tensor([1.0, 0.0]),
            "simulated": torch.tensor([1.0, 0.0]),
        }
        loss, parts = source_aligned_loss(
            output,
            batch,
            scene_weight=0.05,
            negative_upward_weight=0.1,
            positive_downward_weight=0.1,
            correction_l2_weight=0.001,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(parts["simulated_fraction"], 0.5)
        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
