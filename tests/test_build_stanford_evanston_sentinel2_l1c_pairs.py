from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_stanford_evanston_sentinel2_l1c_pairs import (
    load_bound_protocol,
    source_records,
)
from build_stanford_large_controlled_release_l1c_pairs import assert_no_outcome_fields


class EvanstonL1CPairTests(unittest.TestCase):
    def test_source_records_emit_only_pairing_metadata(self) -> None:
        manifest = {
            "center": [-110.931318658838, 41.275737966074],
            "rows": [
                {
                    "event_id": "08092024_S2A",
                    "utc_date": "2024-08-09",
                    "target": {
                        "scene_id": "S2A_12TVL_20240809_0_L1C",
                        "datetime": "2024-08-09T18:22:53.356000+00:00",
                    },
                }
            ],
        }
        records = source_records(manifest)
        self.assertEqual(
            records,
            [
                {
                    "release_id": "08092024_S2A",
                    "sensor": "Sentinel-2",
                    "observed_at_utc": "2024-08-09T18:22:53.356000+00:00",
                    "latitude": 41.275737966074,
                    "longitude": -110.931318658838,
                    "target": {
                        "status": "resolved",
                        "id": "S2A_12TVL_20240809_0_L1C",
                        "datetime": "2024-08-09T18:22:53.356000+00:00",
                    },
                }
            ],
        )
        assert_no_outcome_fields(records)
        serialized = json.dumps(records)
        for forbidden in ("metered", "truth", "label", "release_rate"):
            self.assertNotIn(forbidden, serialized)

    def test_bound_protocol_validates_nine_row_target_manifest(self) -> None:
        protocol, manifest_path, target_manifest = load_bound_protocol(
            ROOT / "configs/stanford_evanston_sentinel2_l1c_pair_protocol.json"
        )
        self.assertEqual(len(target_manifest["rows"]), 9)
        self.assertTrue(manifest_path.is_file())
        self.assertEqual(
            protocol["sentinel_2_l1c_stress_acquisition_contract"]["crop"]["shape_pixels"],
            [256, 256],
        )


if __name__ == "__main__":
    unittest.main()
