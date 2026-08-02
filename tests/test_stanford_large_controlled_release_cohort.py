from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_stanford_large_controlled_release_cohort import (
    extract_summary_urls,
    parse_summary,
    truth_stratum,
)


class StanfordLargeControlledReleaseTests(unittest.TestCase):
    def test_truth_contract_uses_only_exact_blanks_as_negatives(self) -> None:
        self.assertEqual(truth_stratum(0.0), "primary_negative")
        self.assertEqual(truth_stratum(0.1), "subthreshold_challenge")
        self.assertEqual(truth_stratum(999.9), "subthreshold_challenge")
        self.assertEqual(truth_stratum(1000.0), "primary_positive")

    def test_extract_summary_urls_filters_non_target_satellites(self) -> None:
        html = (
            '<a href="https://stacks.stanford.edu/file/qh001qt3946/root/01012025_S2B/01012025_S2B_summary.csv">S2</a>'
            '<a href="https://stacks.stanford.edu/file/qh001qt3946/root/01012025_EN/01012025_EN_summary.csv">EN</a>'
            '<a href="https://stacks.stanford.edu/file/qh001qt3946/root/01022025_LS9/01022025_LS9_summary.csv">LS</a>'
        )
        urls = extract_summary_urls(html)
        self.assertEqual(len(urls), 2)
        self.assertTrue(any("S2B" in url for url in urls))
        self.assertTrue(any("LS9" in url for url in urls))

    def test_parse_summary_preserves_sensor_and_metered_truth(self) -> None:
        text = (
            "release_ID,date,time_UTC,location,lat,lon,ch4_kgh_mean,ch4_kgh_sigma,"
            "ci95_lower,ci95_upper,PredInt95_lower,PredInt95_upper,PI_within_10pct_of_mean\n"
            "01082025_S2B,2025-01-08,18:15:05,Casa Grande,32.821749,-111.785795,"
            "0.0,0.0,0,0,0,0,True\n"
        )
        url = "https://stacks.stanford.edu/file/qh001qt3946/root/01082025_S2B/01082025_S2B_summary.csv"
        row = parse_summary(text, url)
        self.assertEqual(row["sensor"], "Sentinel-2")
        self.assertEqual(row["truth_stratum"], "primary_negative")
        self.assertEqual(row["observed_at_utc"], "2025-01-08T18:15:05+00:00")


if __name__ == "__main__":
    unittest.main()
