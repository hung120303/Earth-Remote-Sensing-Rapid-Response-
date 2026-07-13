from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from build_research_report import build_data, render  # noqa: E402


class ResearchReportTests(unittest.TestCase):
    def test_data_is_bound_to_frozen_campaign(self) -> None:
        data = build_data(ROOT)
        self.assertFalse(data["gate"]["passed"])
        self.assertEqual(data["cohort"]["samples"], 4401)
        self.assertEqual([item["seed"] for item in data["seeds"]], [101, 202, 303, 404, 505])
        self.assertAlmostEqual(data["baseline"]["recall"], 0.6417910447761194)
        self.assertAlmostEqual(data["ersrr_mean"]["recall"], 0.31940298507462683)

    def test_render_embeds_data_once_without_placeholder(self) -> None:
        output = render(
            "<script type='application/json'>__ERSRR_REPORT_DATA__</script>",
            {"decision": "failed", "value": 3},
        )
        self.assertNotIn("__ERSRR_REPORT_DATA__", output)
        self.assertIn('"decision":"failed"', output)


if __name__ == "__main__":
    unittest.main()
