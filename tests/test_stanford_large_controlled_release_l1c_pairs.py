from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_stanford_large_controlled_release_l1c_pairs import (
    assert_no_outcome_fields,
    choose_reference,
    source_targets,
)


def item(
    scene_id: str,
    acquired: datetime,
    *,
    cloud: float | None,
    bbox: list[float] | None = None,
) -> dict[str, object]:
    properties: dict[str, object] = {"datetime": acquired.isoformat()}
    if cloud is not None:
        properties["eo:cloud_cover"] = cloud
    return {
        "id": scene_id,
        "bbox": bbox or [-112.0, 32.0, -111.0, 33.0],
        "properties": properties,
        "assets": {
            "blue": {},
            "green": {},
            "red": {},
            "nir": {},
            "swir16": {},
            "swir22": {},
        },
    }


class StanfordL1CPairContractTests(unittest.TestCase):
    def test_source_targets_omit_labels_and_collect_every_resolved_target_id(self) -> None:
        records = [
            {
                "release_id": "event-s2",
                "sensor": "Sentinel-2",
                "observed_at_utc": "2025-04-01T18:00:00+00:00",
                "latitude": 32.821749,
                "longitude": -111.785795,
                "metered_ch4_kgh": 1234.0,
                "metered_ch4_sigma": 12.0,
                "truth_stratum": "primary_positive",
                "target": {
                    "status": "resolved",
                    "id": "S2A_12SWA_20250401_0_L1C",
                    "datetime": "2025-04-01T18:00:00+00:00",
                },
            },
            {
                "release_id": "event-landsat",
                "sensor": "Landsat",
                "truth_stratum": "primary_negative",
                "target": {"status": "resolved", "id": "LC09_SOURCE_TARGET"},
            },
            {
                "release_id": "event-unresolved",
                "sensor": "Sentinel-2",
                "metered_ch4_kgh": 0.0,
                "target": {"status": "unresolved"},
            },
        ]

        targets, excluded = source_targets(records)

        self.assertEqual(len(targets), 1)
        self.assertEqual(
            excluded,
            {"S2A_12SWA_20250401_0_L1C", "LC09_SOURCE_TARGET"},
        )
        serialized = json.dumps(targets, sort_keys=True)
        for forbidden in ("metered_ch4_kgh", "metered_ch4_sigma", "truth_stratum"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(
            set(targets[0]),
            {
                "event_id",
                "observed_at_utc",
                "center",
                "target_scene_id",
                "target_datetime",
            },
        )
        assert_no_outcome_fields(targets)

    def test_outcome_field_guard_refuses_labels_in_any_output_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden outcome field"):
            assert_no_outcome_fields({"nested": [{"truth_stratum": "primary_negative"}]})
        with self.assertRaisesRegex(ValueError, "forbidden outcome field"):
            assert_no_outcome_fields({"label": 0})

    def test_reference_selection_is_prior_only_target_excluding_and_deterministic(self) -> None:
        target_time = datetime(2025, 4, 10, 18, tzinfo=timezone.utc)
        center = [-111.785795, 32.821749]
        excluded = {
            "S2A_12SWA_20250410_0_L1C",
            "S2B_12SWA_20250410_1_L1C",
            "S2C_12SWA_20250410_2_L1C",
        }
        candidates = [
            # Closest metadata candidate is another source-cohort target and must be excluded.
            item("S2B_12SWA_20250410_1_L1C", target_time - timedelta(hours=1), cloud=0),
            # Under one hour, future, over-cloud, missing-cloud, wrong-tile, and over-lookback.
            item("S2A_12SWA_20250409_0_L1C", target_time - timedelta(minutes=59), cloud=0),
            item("S2A_12SWA_20250411_0_L1C", target_time + timedelta(hours=1), cloud=0),
            item("S2A_12SWA_20250408_0_L1C", target_time - timedelta(hours=2), cloud=20.1),
            item("S2A_12SWA_20250408_1_L1C", target_time - timedelta(hours=2), cloud=None),
            item("S2A_12SWB_20250408_0_L1C", target_time - timedelta(hours=2), cloud=0),
            item("S2A_12SWA_20250301_0_L1C", target_time - timedelta(days=31, seconds=1), cloud=0),
            # Same gap/cloud tie resolves lexicographically by ID.
            item("S2C_12SWA_20250408_2_L1C", target_time - timedelta(hours=2), cloud=5),
            item("S2A_12SWA_20250408_2_L1C", target_time - timedelta(hours=2), cloud=5),
            # A lower-cloud but larger-gap item loses because gap is the first key.
            item("S2A_12SWA_20250407_0_L1C", target_time - timedelta(hours=3), cloud=0),
        ]

        selected = choose_reference(
            candidates,
            target_time=target_time,
            target_tile="12SWA",
            center=center,
            excluded_target_ids=excluded,
            min_gap_hours=1.0,
            max_lookback_days=31,
            max_cloud=20.0,
        )

        self.assertEqual(selected["id"], "S2A_12SWA_20250408_2_L1C")

    def test_seasonal_fallback_is_used_only_when_primary_window_is_empty(self) -> None:
        target_time = datetime(2025, 8, 9, 18, tzinfo=timezone.utc)
        center = [-111.785795, 32.821749]
        candidates = [
            item("S2A_12SWA_20240808_0_L1C", target_time - timedelta(days=366), cloud=1),
            item("S2B_12SWA_20240809_0_L1C", target_time - timedelta(days=365), cloud=5),
        ]
        selected = choose_reference(
            candidates,
            target_time=target_time,
            target_tile="12SWA",
            center=center,
            excluded_target_ids=set(),
            min_gap_hours=1.0,
            max_lookback_days=31,
            max_cloud=20.0,
        )
        self.assertEqual(selected["id"], "S2B_12SWA_20240809_0_L1C")

        candidates.append(
            item("S2A_12SWA_20250801_0_L1C", target_time - timedelta(days=8), cloud=19)
        )
        selected = choose_reference(
            candidates,
            target_time=target_time,
            target_tile="12SWA",
            center=center,
            excluded_target_ids=set(),
            min_gap_hours=1.0,
            max_lookback_days=31,
            max_cloud=20.0,
        )
        self.assertEqual(selected["id"], "S2A_12SWA_20250801_0_L1C")

    def test_reference_selection_excludes_campaign_utc_dates(self) -> None:
        target_time = datetime(2025, 12, 4, 18, tzinfo=timezone.utc)
        center = [-111.785795, 32.821749]
        candidates = [
            item("S2A_12SWA_20241204_0_L1C", target_time - timedelta(days=365), cloud=0),
            item("S2B_12SWA_20241203_0_L1C", target_time - timedelta(days=366), cloud=5),
        ]
        selected = choose_reference(
            candidates,
            target_time=target_time,
            target_tile="12SWA",
            center=center,
            excluded_target_ids=set(),
            excluded_utc_dates={"2024-12-04"},
            min_gap_hours=1.0,
            max_lookback_days=31,
            max_cloud=20.0,
        )
        self.assertEqual(selected["id"], "S2B_12SWA_20241203_0_L1C")


if __name__ == "__main__":
    unittest.main()
