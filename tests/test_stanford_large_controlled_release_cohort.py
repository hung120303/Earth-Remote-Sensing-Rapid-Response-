from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_stanford_large_controlled_release_cohort import (
    column_index,
    excel_datetime,
    select_events,
    truth_stratum,
)


class StanfordLargeControlledReleaseTests(unittest.TestCase):
    def test_truth_contract_uses_only_exact_blanks_as_negatives(self) -> None:
        self.assertEqual(truth_stratum(0.0), "primary_negative")
        self.assertEqual(truth_stratum(0.1), "subthreshold_challenge")
        self.assertEqual(truth_stratum(999.9), "subthreshold_challenge")
        self.assertEqual(truth_stratum(1000.0), "primary_positive")

    def test_excel_columns_and_serial_datetime(self) -> None:
        self.assertEqual(column_index("A1"), 0)
        self.assertEqual(column_index("AS2408"), 44)
        observed = excel_datetime(45658, 0.7673842592592592)
        self.assertEqual(observed.isoformat(), "2025-01-01T18:25:02+00:00")

    def test_select_events_applies_paper_qc_and_deduplicates(self) -> None:
        base = {
            "release_ID": "01012025_S2B",
            "date": 45658,
            "time_UTC": 0.75,
            "location": "Casa Grande",
            "lat": 32.821749,
            "lon": -111.785795,
            "ch4_kgh_mean": 0,
            "ch4_kgh_sigma": 0,
            "SatelliteCode": "S2B",
            "SatellitePlotName": "Sentinel-2",
            "Acquisition status": "OK",
            "QC_ExperimentTeam": "OK",
            "Phase": 1,
        }
        canceled = {**base, "release_ID": "01022025_S2B", "QC_ExperimentTeam": "ET Canceled"}
        other = {**base, "release_ID": "01032025_EN", "SatellitePlotName": "EnMAP"}
        rows = select_events([base, dict(base), canceled, other])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["truth_stratum"], "primary_negative")
        self.assertEqual(rows[0]["sensor"], "Sentinel-2")


if __name__ == "__main__":
    unittest.main()
