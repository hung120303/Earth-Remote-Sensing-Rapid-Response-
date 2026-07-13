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

    def test_publication_template_is_self_contained_and_complete(self) -> None:
        template = (ROOT / "tools/templates/ersrr_research_report.html").read_text(
            encoding="utf-8"
        )
        output = render(template, build_data(ROOT))
        self.assertNotIn("__ERSRR_REPORT_DATA__", output)
        self.assertNotIn('src="http', output)
        self.assertNotIn("@import", output)
        self.assertEqual(output.count("<section "), 7)
        self.assertIn("We cut false alarms.", output)
        self.assertIn("Promotion gate / failed", output)
        self.assertIn('"samples":4401', output)
        self.assertIn('"emit_sealed":55', output)
        self.assertIn("https://cds.climate.copernicus.eu/how-to-api", output)


if __name__ == "__main__":
    unittest.main()
