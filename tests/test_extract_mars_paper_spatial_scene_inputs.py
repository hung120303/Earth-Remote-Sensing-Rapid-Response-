from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from extract_mars_paper_spatial_scene_inputs import label_free_metadata  # noqa: E402


class MarsPaperSpatialSceneInputTests(unittest.TestCase):
    def test_metadata_has_no_label_field(self) -> None:
        records = [
            {
                "sample_id": "sample",
                "group_id": "site",
                "sensor_family": "Sentinel-2",
                "label_state": "PLUME",
            }
        ]
        metadata = label_free_metadata(records)
        self.assertEqual(set(metadata), {"sample_ids", "groups", "sensors"})
        self.assertEqual(metadata["sample_ids"].tolist(), ["sample"])


if __name__ == "__main__":
    unittest.main()
