from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_paper_model import MarsPaperResidualModel  # noqa: E402


class MarsPaperResidualModelTests(unittest.TestCase):
    def inputs(self, batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(17)
        values = torch.rand(batch, 16, 32, 32, generator=generator)
        values[:, 0] = 1.0 + 0.05 * (values[:, 0] - 0.5)
        observable = torch.ones(batch, 1, 32, 32)
        sensors = torch.tensor([index % 2 for index in range(batch)])
        return values, observable, sensors

    def test_zero_initialized_correction_preserves_released_logits(self) -> None:
        model = MarsPaperResidualModel().eval()
        values, observable, sensors = self.inputs()
        with torch.no_grad():
            output = model(values, observable, sensors)
        torch.testing.assert_close(
            output["segmentation_logits"], output["baseline_logits"]
        )
        self.assertEqual(tuple(output["scene_logit"].shape), (2,))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA autocast is unavailable")
    def test_zero_initialized_correction_preserves_cuda_autocast_logits(self) -> None:
        model = MarsPaperResidualModel().cuda().eval()
        values, observable, sensors = self.inputs()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(values.cuda(), observable.cuda(), sensors.cuda())
        self.assertEqual(
            output["segmentation_logits"].dtype,
            output["baseline_logits"].dtype,
        )
        self.assertTrue(
            torch.equal(
                output["segmentation_logits"], output["baseline_logits"]
            )
        )

    def test_backbone_is_frozen_and_correction_receives_gradients(self) -> None:
        model = MarsPaperResidualModel().train()
        values, observable, sensors = self.inputs()
        output = model(values, observable, sensors)
        output["segmentation_logits"].mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.parameters()))
        self.assertIsNotNone(model.correction.output.weight.grad)

    def test_invalid_sensor_index_fails(self) -> None:
        model = MarsPaperResidualModel().eval()
        values, observable, _ = self.inputs(batch=1)
        with self.assertRaises(ValueError):
            model(values, observable, torch.tensor([2]))


if __name__ == "__main__":
    unittest.main()
