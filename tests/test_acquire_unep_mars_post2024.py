from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from acquire_unep_mars_post2024 import (
    connected_source_groups,
    haversine_km,
    role_for_bucket,
    select_samples,
    split_bucket,
    valid_product_pair,
)


def record(
    plume_id: str,
    source: str,
    target: str,
    *,
    lon: float = 10.0,
    satellite: str = "Sentinel-2 - ESA",
) -> dict[str, str]:
    background = (
        "S2B_MSIL1C_20250101T100000_N0500_R000_T31AAA_20250101T110000"
        if satellite.startswith("Sentinel")
        else "LC08_L1TP_001001_20250101_20250102_02_T1"
    )
    return {
        "id_plume": plume_id,
        "source_name": source,
        "satellite": satellite,
        "tile_date": "2025-02-01T10:00:00",
        "tile": target,
        "tile_background": background,
        "lat": "0",
        "lon": str(lon),
        "actionable": "Yes",
    }


class UnepMarsAcquisitionTests(unittest.TestCase):
    def test_product_contract_accepts_supported_exact_products(self) -> None:
        self.assertTrue(
            valid_product_pair(
                "Sentinel-2 - ESA",
                "S2A_MSIL1C_20250201T100000_N0500_R000_T31AAA_20250201T110000",
                "S2B_MSIL1C_20250101T100000_N0500_R000_T31AAA_20250101T110000",
            )
        )
        self.assertTrue(
            valid_product_pair(
                "Landsat - NASA/USGS",
                "LC09_L1TP_001001_20250201_20250202_02_T1",
                "LC08_L1TP_001001_20250101_20250102_02_T1",
            )
        )
        self.assertFalse(
            valid_product_pair("Sentinel-2 - ESA", "S2-target", "S2-reference")
        )

    def test_spatial_components_prevent_nearby_source_leakage(self) -> None:
        groups = connected_source_groups(
            {"a": (0.0, 0.0), "b": (0.1, 0.0), "c": (1.0, 0.0)}, 25.0
        )
        self.assertEqual(groups["a"], groups["b"])
        self.assertNotEqual(groups["a"], groups["c"])
        self.assertGreater(haversine_km((0.0, 0.0), (1.0, 0.0)), 100.0)

    def test_hash_split_is_deterministic_and_total(self) -> None:
        bucket = split_bucket("unep25_example")
        self.assertEqual(bucket, split_bucket("unep25_example"))
        self.assertIn(bucket, range(10))
        self.assertIn(
            role_for_bucket(bucket),
            {"auxiliary_training", "development", "sealed_external"},
        )

    def test_sample_key_keeps_distant_sources_on_same_product(self) -> None:
        target = "S2A_MSIL1C_20250201T100000_N0500_R000_T31AAA_20250201T110000"
        rows = [
            record("p1", "source-a", target, lon=10.0),
            record("p2", "source-a", target, lon=10.0),
            record("p3", "source-b", target, lon=20.0),
        ]
        geometry = {
            key: {"type": "MultiPolygon", "coordinates": []}
            for key in ("p1", "p2", "p3")
        }
        samples, audit = select_samples(
            rows,
            geometry,
            cutoff=datetime(2025, 1, 1, tzinfo=timezone.utc),
            allowed_satellites={"Sentinel-2 - ESA"},
            required_fields=[
                "id_plume",
                "source_name",
                "satellite",
                "tile_date",
                "tile",
                "tile_background",
                "lat",
                "lon",
            ],
            paper_targets=set(),
            paper_locations=[],
            exclusion_km=25.0,
        )
        self.assertEqual(len(samples), 2)
        self.assertEqual(sorted(len(sample["plume_ids"]) for sample in samples), [1, 2])
        self.assertEqual(audit["merged_duplicate_plume_records"], 1)

    def test_paper_target_and_location_are_excluded(self) -> None:
        paper_target = "S2A_MSIL1C_20250201T100000_N0500_R000_T31AAA_20250201T110000"
        other_target = "S2A_MSIL1C_20250202T100000_N0500_R000_T31AAA_20250202T110000"
        rows = [
            record("target", "source-a", paper_target, lon=20.0),
            record("near", "source-b", other_target, lon=0.01),
        ]
        geometry = {key: None for key in ("target", "near")}
        samples, audit = select_samples(
            rows,
            geometry,
            cutoff=datetime(2025, 1, 1, tzinfo=timezone.utc),
            allowed_satellites={"Sentinel-2 - ESA"},
            required_fields=list(rows[0]),
            paper_targets={paper_target},
            paper_locations=[(0.0, 0.0)],
            exclusion_km=25.0,
        )
        self.assertEqual(samples, [])
        self.assertEqual(audit["exact_paper_test_target"], 1)
        self.assertEqual(audit["within_paper_test_exclusion"], 1)


if __name__ == "__main__":
    unittest.main()
