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
    crop_grid,
    geometry_gate,
    interpolate_s2ee_20m_bands,
    landsat_cloud_classes,
    select_shard,
    source_contract,
    stable_identity,
    validate_cached_contract,
)


class ExactCropTests(unittest.TestCase):
    def test_official_s2ee_20m_interpolation_is_typed_and_deterministic(self) -> None:
        values = np.arange(200 * 200, dtype=np.uint16).reshape(200, 200)
        first = interpolate_s2ee_20m_bands(values)
        second = interpolate_s2ee_20m_bands(values)
        self.assertEqual(first.shape, (200, 200))
        self.assertEqual(first.dtype, np.uint16)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, values))

    def test_published_crop_grid_is_preserved_exactly(self) -> None:
        cohort = {
            "sample_id": "sample",
            "source_grid": {
                "crs": "EPSG:32632",
                "transform": [10.0, 0.0, 408510.0, 0.0, -10.0, 3776960.0],
                "width": 200,
                "height": 200,
            },
        }
        crs, transform = crop_grid(cohort, "not-opened")
        self.assertEqual(crs.to_string(), "EPSG:32632")
        self.assertEqual(tuple(transform)[:6], tuple(cohort["source_grid"]["transform"]))

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

    def test_cached_contract_rejects_superseded_s2_resampling(self) -> None:
        cohort = {
            "sample_id": "sample",
            "group_id": "group",
            "research_role": "development",
            "label_state": "NO_PLUME",
            "sensor_family": "Sentinel-2",
            "target_product": "target",
            "background_product": "background",
            "source_grid": {
                "crs": "EPSG:32632",
                "transform": [10.0, 0.0, 408510.0, 0.0, -10.0, 3776960.0],
                "width": 200,
                "height": 200,
            },
        }
        manifest = {
            **{key: cohort[key] for key in (
                "sample_id",
                "group_id",
                "research_role",
                "label_state",
                "target_product",
                "background_product",
            )},
            "product_contract": {
                "shape": [200, 200],
                "resolution_m": 10.0,
                "crs": "EPSG:32632",
                "transform": cohort["source_grid"]["transform"],
                "band_order": [
                    *S2_DESCRIPTIONS,
                    *(f"{band}_bg" for band in S2_DESCRIPTIONS),
                ],
                "dtype": "uint16",
                "resampling": "bilinear spectral",
            },
        }
        with self.assertRaisesRegex(ValueError, "preprocessing contract"):
            validate_cached_contract(manifest, cohort)


if __name__ == "__main__":
    unittest.main()
