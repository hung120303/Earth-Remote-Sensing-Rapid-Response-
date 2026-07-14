from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from audit_mars_paper_benchmark import (  # noqa: E402
    PUBLISHED,
    assignment_line,
    metric_delta,
    scene_metrics,
)


def row(sample_id: str, label: int, score: float, location: str) -> dict[str, object]:
    return {
        "id_loc_image": sample_id,
        "location_name": location,
        "tile": f"tile-{sample_id}",
        "target": label,
        "scene_pred": score,
        "TP": 3 if label else 0,
        "FP": 1 if score > 0.5 else 0,
        "TN": 10,
        "FN": 1 if label else 0,
        "test_only_site": location == "new",
        "is_offshore": False,
        "public_metadata_available": True,
        "baseline_source": "general_model",
    }


class MarsPaperBenchmarkTests(unittest.TestCase):
    def test_scene_metrics_use_author_strict_half_threshold_and_micro_iou(self) -> None:
        rows = [
            row("a", 1, 0.9, "seen"),
            row("b", 1, 0.5, "new"),
            row("c", 0, 0.8, "new"),
            row("d", 0, 0.2, "seen"),
        ]
        result = scene_metrics(rows)
        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["false_positive_rate"], 0.5)
        self.assertAlmostEqual(result["pixel_iou"], 6 / 10)

    def test_assignment_serialization_is_compact_and_stable(self) -> None:
        payload = json.loads(assignment_line(row("a", 1, 0.9, "seen")))
        self.assertEqual(payload["sample_id"], "a")
        self.assertEqual(payload["target"], 1)
        self.assertEqual(payload["baseline_scene_score"], 0.9)
        self.assertNotIn("Q", payload)

    def test_metric_delta_only_compares_shared_float_metrics(self) -> None:
        observed = {"rows": 10, "recall": 0.8, "pixel_iou": 0.4}
        delta = metric_delta(observed, {"rows": 10, "recall": 0.7, "missing": 0.2})
        self.assertEqual(delta, {"recall": 0.10000000000000009})

    def test_published_contract_is_current_paper_v3_table(self) -> None:
        self.assertEqual(PUBLISHED["full"]["rows"], 43_529)
        self.assertEqual(PUBLISHED["test_only_sites"]["rows"], 15_655)
        self.assertEqual(PUBLISHED["full"]["average_precision"], 0.6408)
        self.assertEqual(PUBLISHED["full"]["pixel_iou"], 0.3224)


if __name__ == "__main__":
    unittest.main()
