from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_stanford_controlled_release_cohort import (
    haversine_km,
    mars_overlap,
    methane_rate,
    parse_rows,
    resolve_item,
    summarize,
    truth_stratum,
)


class StanfordControlledReleaseTests(unittest.TestCase):
    def test_truth_contract_keeps_intermediate_rates_separate(self) -> None:
        self.assertEqual(truth_stratum(0.0), "primary_negative")
        self.assertEqual(truth_stratum(4.95), "primary_negative")
        self.assertEqual(truth_stratum(73.9), "subthreshold_challenge")
        self.assertEqual(truth_stratum(999.9), "subthreshold_challenge")
        self.assertEqual(truth_stratum(1000.0), "primary_positive")

    def test_zero_gas_fills_missing_ch4_as_zero(self) -> None:
        self.assertEqual(methane_rate({"ch4_kgh_mean": "", "gas_kgh_mean": "0.0"}), 0.0)

    def test_parse_filters_other_satellites_and_normalizes_landsat(self) -> None:
        text = (
            ",Team,Satellite,Date,Timestamp (UTC),gas_kgh_mean,ch4_kgh_mean\n"
            "0,X,LandSat,2022-10-10,17:58:31,0.0,0.0\n"
            "1,X,PRISMA,2022-10-10,18:00:00,1000,950\n"
        )
        rows = parse_rows(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sensor"], "Landsat")
        self.assertEqual(rows[0]["truth_stratum"], "primary_negative")

    def test_resolver_rejects_wrong_landsat_generation_and_picks_nearest(self) -> None:
        def fake_post(_url, _payload):
            return {
                "features": [
                    {
                        "id": "LE07_L1TP_037037_20221010_20221105_02_T1",
                        "properties": {"datetime": "2022-10-10T17:58:31Z"},
                        "assets": {},
                    },
                    {
                        "id": "LC08_L1TP_036037_20221010_20221020_02_T1",
                        "properties": {"datetime": "2022-10-10T17:58:22Z"},
                        "assets": {
                            name: {"href": f"https://example/{name}.tif"}
                            for name in ("blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel")
                        },
                    },
                ]
            }

        result = resolve_item(
            "Landsat",
            datetime(2022, 10, 10, 17, 58, 31, tzinfo=timezone.utc),
            post=fake_post,
        )
        self.assertEqual(result["status"], "resolved")
        self.assertTrue(result["id"].startswith("LC08_"))
        self.assertEqual(result["scheduled_delta_seconds"], 9.0)

    def test_haversine_identity_and_summary(self) -> None:
        self.assertEqual(haversine_km(-111.0, 32.0, -111.0, 32.0), 0.0)
        summary = summarize(
            [
                {"sensor": "Sentinel-2", "truth_stratum": "primary_positive", "target": {"status": "resolved"}},
                {"sensor": "Landsat", "truth_stratum": "primary_negative", "target": {"status": "unresolved"}},
            ]
        )
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["sensor_truth"]["Landsat:primary_negative"], 1)

    def test_mars_overlap_separates_excluded_same_site_from_disjoint_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mars.csv"
            path.write_text(
                "tile,split_name,isplume,id_location,lon,lat\n"
                "TARGET,Not Used,True,site-a,-111.785773,32.8218205\n"
                "OTHER,Not Used,False,site-a,-111.785773,32.8218205\n"
                "FAR,train,False,site-b,0,0\n",
                encoding="utf-8",
            )
            result = mars_overlap(path, {"TARGET"})
        self.assertEqual(result["exact_target_product_matches"], 1)
        self.assertEqual(result["same_site_rows"], 2)
        self.assertEqual(result["same_site_split_label"]["Not Used:False"], 1)
        self.assertFalse(result["site_disjoint_at_25km"])


if __name__ == "__main__":
    unittest.main()
