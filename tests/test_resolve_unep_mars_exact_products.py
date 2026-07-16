from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from resolve_unep_mars_exact_products import (
    contains,
    official_s2_href,
    s2_date,
    summarize,
)


class ExactProductResolverTests(unittest.TestCase):
    def test_sentinel_product_date_accepts_s2c(self) -> None:
        value = s2_date(
            "S2C_MSIL1C_20250420T074631_N0511_R135_T38TPQ_20250420T094856"
        )
        self.assertEqual(value.isoformat(), "2025-04-20T00:00:00+00:00")

    def test_l1c_bucket_correction_preserves_object_key(self) -> None:
        href = "s3://sentinel-s2-l2a/tiles/38/T/PQ/2025/4/20/0/B11.jp2"
        self.assertEqual(
            official_s2_href(href),
            "https://sentinel-s2-l1c.s3.amazonaws.com/tiles/38/T/PQ/2025/4/20/0/B11.jp2",
        )

    def test_coverage_check_is_inclusive(self) -> None:
        item = {"bbox": [10.0, 20.0, 11.0, 21.0]}
        self.assertTrue(contains(item, [10.0, 21.0]))
        self.assertFalse(contains(item, [9.9, 20.5]))

    def test_summary_stratifies_resolved_and_unresolved(self) -> None:
        records = [
            {
                "status": "resolved",
                "sensor_family": "Sentinel-2",
                "research_role": "auxiliary_training",
                "target": {"status": "resolved"},
                "reference": {"status": "resolved"},
            },
            {
                "status": "unresolved",
                "sensor_family": "Landsat",
                "research_role": "development",
                "target": {"status": "unavailable_exact_product"},
                "reference": {"status": "resolved"},
            },
        ]
        result = summarize(records)
        self.assertEqual(result["status"], {"resolved": 1, "unresolved": 1})
        self.assertEqual(result["by_sensor"]["Landsat"]["total"], 1)
        self.assertEqual(
            result["unresolved_product_sides"][
                "Landsat:target:unavailable_exact_product"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
