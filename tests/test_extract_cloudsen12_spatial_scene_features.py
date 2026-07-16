from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from extract_cloudsen12_spatial_scene_features import validate_records


class CloudsenSpatialSceneFeatureTests(unittest.TestCase):
    def test_role_sensor_and_negative_contract(self) -> None:
        record = {
            "sample_id": "one",
            "research_role": "development",
            "label_state": "NO_PLUME",
            "sensor_family": "Sentinel-2",
        }
        validate_records([record], "development", 1)
        with self.assertRaisesRegex(ValueError, "negative-only"):
            validate_records([{**record, "label_state": "PLUME"}], "development", 1)
        with self.assertRaisesRegex(ValueError, "Sentinel-2-only"):
            validate_records([{**record, "sensor_family": "Landsat"}], "development", 1)

    def test_duplicate_samples_are_rejected(self) -> None:
        record = {
            "sample_id": "one",
            "research_role": "auxiliary_training",
            "label_state": "NO_PLUME",
            "sensor_family": "Sentinel-2",
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_records([record, record], "auxiliary_training", 2)

    def test_fresh_external_negative_role_is_supported(self) -> None:
        record = {
            "sample_id": "fresh",
            "research_role": "fresh_external_test",
            "label_state": "NO_PLUME",
            "sensor_family": "Sentinel-2",
        }
        validate_records([record], "fresh_external_test", 1)


if __name__ == "__main__":
    unittest.main()
