from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from acquire_unep_mars_exact_crops import (
    LANDSAT_DESCRIPTIONS,
    S2_DESCRIPTIONS,
    geometry_gate,
    landsat_cloud_classes,
    select_shard,
    source_contract,
    stable_identity,
)


class ExactCropTests(unittest.TestCase):
    def test_shards_are_disjoint_and_complete(self) -> None:
        records = [{"sample_id": str(index)} for index in range(11)]
        shards = [select_shard(records, 3, index) for index in range(3)]
        flattened = [item["sample_id"] for shard in shards for item in shard]
        self.assertEqual(sorted(flattened, key=int), [str(index) for index in range(11)])
        self.assertEqual(len(flattened), len(set(flattened)))
        with self.assertRaises(ValueError):
            select_shard(records, 3, 3)

    def test_geometry_gate_respects_positive_and_negative_labels(self) -> None:
        self.assertTrue(geometry_gate("PLUME", 1))
        self.assertFalse(geometry_gate("PLUME", 0))
        self.assertTrue(geometry_gate("NO_PLUME", 0))
        self.assertFalse(geometry_gate("NO_PLUME", 1))
        with self.assertRaises(ValueError):
            geometry_gate("UNKNOWN", 0)

    def test_sensor_band_descriptions_match_released_order(self) -> None:
        self.assertEqual(source_contract("Sentinel-2")[1], S2_DESCRIPTIONS)
        self.assertEqual(source_contract("Landsat")[1], LANDSAT_DESCRIPTIONS)
        self.assertEqual(LANDSAT_DESCRIPTIONS, ("B02", "B03", "B04", "B05", "B06", "B07"))

    def test_landsat_cloud_mapping_preserves_clear_and_water(self) -> None:
        qa = np.asarray([[0, 1 << 7, 1 << 3, 1 << 4, 1 << 5]], dtype=np.uint16)
        cloud = landsat_cloud_classes(qa)
        np.testing.assert_array_equal(cloud, [[0, 0, 1, 1, 1]])

    def test_landsat_cloud_mapping_includes_fill_dilated_and_cirrus(self) -> None:
        qa = np.asarray([[1 << 0, 1 << 1, 1 << 2]], dtype=np.uint16)
        np.testing.assert_array_equal(landsat_cloud_classes(qa), [[1, 1, 1]])

    def test_input_identity_changes_with_asset_contract(self) -> None:
        cohort = {"sample_id": "a", "plume_ids": ["p"]}
        left = stable_identity(cohort, {"target": "one"})
        right = stable_identity(cohort, {"target": "two"})
        self.assertNotEqual(left, right)
        self.assertEqual(left, stable_identity(cohort, {"target": "one"}))


if __name__ == "__main__":
    unittest.main()
