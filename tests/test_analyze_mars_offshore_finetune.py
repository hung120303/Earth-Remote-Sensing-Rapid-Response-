from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from analyze_mars_offshore_finetune import (  # noqa: E402
    analyze,
    numeric_delta,
    sensor_family,
)


def assignment(
    sample_id: str,
    target: int,
    *,
    offshore: bool,
    test_only: bool,
    tile: str = "S2A_tile",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "location_name": f"site-{sample_id}",
        "tile": tile,
        "target": target,
        "is_offshore": offshore,
        "test_only_site": test_only,
    }


def prediction(sample_id: str, target: int, score: float, good_mask: bool) -> dict[str, str]:
    tp = 8 if target and good_mask else 2 if target else 0
    fn = 2 if target and good_mask else 8 if target else 0
    fp = 1 if good_mask else 5
    return {
        "id_loc_image": sample_id,
        "scene_pred": str(score),
        "target": str(target),
        "location_name": f"site-{sample_id}",
        "tile": "tile",
        "TP": str(tp),
        "FP": str(fp),
        "TN": "20",
        "FN": str(fn),
    }


class OffshoreFinetuneDiagnosticTests(unittest.TestCase):
    def test_sensor_family(self) -> None:
        self.assertEqual(sensor_family("LC09_foo"), "Landsat")
        self.assertEqual(sensor_family("S2A_foo"), "Sentinel-2")

    def test_numeric_delta_preserves_count_direction(self) -> None:
        keys = {
            "average_precision": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "false_positive_rate": 1.0,
            "pixel_iou": 1.0,
            "tp": 2,
            "fp": 2,
            "tn": 2,
            "fn": 2,
            "pixel_tp": 2,
            "pixel_fp": 2,
            "pixel_fn": 2,
        }
        baseline = {key: 0 for key in keys}
        self.assertEqual(numeric_delta(keys, baseline), keys)

    def test_hybrid_changes_only_offshore_rows(self) -> None:
        assignments = [
            assignment("a", 1, offshore=True, test_only=True),
            assignment("b", 0, offshore=True, test_only=True, tile="LC09_tile"),
            assignment("c", 1, offshore=False, test_only=False),
            assignment("d", 0, offshore=False, test_only=True),
        ]
        general = [
            prediction("a", 1, 0.4, False),
            prediction("b", 0, 0.8, False),
            prediction("c", 1, 0.9, True),
            prediction("d", 0, 0.1, True),
        ]
        fine = [
            prediction("a", 1, 0.9, True),
            prediction("b", 0, 0.1, True),
            prediction("c", 1, 0.1, False),
            prediction("d", 0, 0.9, False),
        ]
        result = analyze(assignments, general, fine)
        full = result["global_hybrid_effect"]["full"]
        self.assertEqual(full["offshore_rows"], 2)
        self.assertGreater(full["delta"]["average_precision"], 0)
        self.assertGreater(full["delta"]["pixel_iou"], 0)


if __name__ == "__main__":
    unittest.main()
