from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from build_research_report import build_data, render  # noqa: E402


class ResearchReportTests(unittest.TestCase):
    def test_data_is_bound_to_frozen_v5_1_location_test(self) -> None:
        data = build_data(ROOT)
        self.assertTrue(data["status"]["test_frozen"])
        self.assertFalse(data["status"]["retuning_permitted"])
        self.assertEqual(data["cohort"]["scenes"], 20789)
        self.assertEqual(
            [item["name"] for item in data["models"]],
            ["ersrr_v5_1", "ersrr_v4_3", "released_mars_s2l"],
        )
        self.assertAlmostEqual(
            data["models"][0]["metrics"]["scene"]["average_precision"],
            0.8180196480303749,
        )
        self.assertAlmostEqual(
            data["models"][0]["metrics"]["scene"]["false_positive_rate"],
            0.06066176470588235,
        )

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
        self.assertIn("The signal travels.", output)
        self.assertIn("Across-the-board MARS-S2L superiority", output)
        self.assertIn('"scenes":20789', output)
        self.assertIn('"retuning_permitted":false', output)
        self.assertIn("Prediction cache SHA-256", output)


if __name__ == "__main__":
    unittest.main()
