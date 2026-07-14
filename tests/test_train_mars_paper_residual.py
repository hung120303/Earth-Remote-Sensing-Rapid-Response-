from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from train_mars_paper_residual import (  # noqa: E402
    sampling_weights,
    smoke_subset,
    successor_loss,
    validation_summary,
    verify_acquisition_receipt,
)


class MarsPaperResidualTrainingTests(unittest.TestCase):
    class IdentityModel:
        backbone_trainable = False

        def eval(self) -> None:
            return None

        def __call__(
            self,
            inputs: torch.Tensor,
            observable: torch.Tensor,
            sensor_index: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del observable, sensor_index
            logits = inputs[:, :1]
            return {
                "segmentation_logits": logits,
                "baseline_logits": logits,
            }

    def validation_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for sensor in (0, 1):
            for label in (0, 1):
                logit = 2.0 if label else -2.0
                rows.append(
                    {
                        "inputs": torch.full((1, 16, 16), logit),
                        "observable": torch.ones(1, 16, 16),
                        "clear": torch.ones(1, 16, 16),
                        "mask": torch.full((1, 16, 16), float(label)),
                        "presence": torch.tensor(float(label)),
                        "sensor_index": torch.tensor(sensor),
                        "sample_id": f"sample-{sensor}-{label}",
                        "group_id": f"group-{sensor}-{label}",
                    }
                )
        return rows

    def test_validation_reuses_only_matching_released_baseline(self) -> None:
        loader = DataLoader(self.validation_rows(), batch_size=2, shuffle=False)
        first = validation_summary(self.IdentityModel(), loader, torch.device("cpu"))
        cached = validation_summary(
            self.IdentityModel(),
            loader,
            torch.device("cpu"),
            baseline_reference=first,
        )
        self.assertEqual(cached["released_baseline"], first["released_baseline"])
        self.assertEqual(cached["delta"], first["delta"])
        altered = dict(first)
        altered["cohort_fingerprint"] = "wrong"
        with self.assertRaisesRegex(ValueError, "different validation cohort"):
            validation_summary(
                self.IdentityModel(),
                loader,
                torch.device("cpu"),
                baseline_reference=altered,
            )

    def test_sampling_equalizes_label_sensor_strata(self) -> None:
        records = [
            {"label_state": "PLUME", "sensor_family": "Sentinel-2"},
            {"label_state": "PLUME", "sensor_family": "Sentinel-2"},
            {"label_state": "NO_PLUME", "sensor_family": "Landsat"},
        ]
        weights = sampling_weights(records)
        self.assertEqual(weights.tolist(), [0.5, 0.5, 1.0])

    def test_smoke_subset_keeps_both_labels(self) -> None:
        records = [
            {"label_state": "PLUME", "sensor_family": "Sentinel-2"},
            {"label_state": "NO_PLUME", "sensor_family": "Sentinel-2"},
        ]
        self.assertEqual(smoke_subset(records, 1), records)

    def test_negative_upward_change_is_penalized(self) -> None:
        logits = torch.ones(1, 1, 4, 4, requires_grad=True)
        output = {
            "segmentation_logits": logits,
            "baseline_logits": torch.zeros_like(logits),
            "correction_logits": logits,
            "scene_logit": torch.zeros(1, requires_grad=True),
        }
        batch = {
            "mask": torch.zeros_like(logits),
            "observable": torch.ones_like(logits),
            "presence": torch.zeros(1),
        }
        loss, parts = successor_loss(output, batch)
        self.assertGreater(parts["negative_upward_penalty"], 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_positive_downward_change_is_penalized(self) -> None:
        logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
        output = {
            "segmentation_logits": logits,
            "baseline_logits": torch.ones_like(logits),
            "correction_logits": logits,
            "scene_logit": torch.zeros(1, requires_grad=True),
        }
        batch = {
            "mask": torch.ones_like(logits),
            "observable": torch.ones_like(logits),
            "presence": torch.ones(1),
        }
        loss, parts = successor_loss(output, batch)
        self.assertGreater(parts["positive_downward_penalty"], 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_receipt_must_match_development_manifest(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(
                json.dumps(
                    {
                        "result": {
                            "ok": True,
                            "manifest_filter": {"sha256": "expected"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            verify_acquisition_receipt(path, "expected")
            with self.assertRaises(ValueError):
                verify_acquisition_receipt(path, "different")


if __name__ == "__main__":
    unittest.main()
