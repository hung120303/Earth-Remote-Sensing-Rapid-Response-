from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_unep_mars_model_manifest import MARS_BAND_ORDER, model_record


class UnepModelManifestTests(unittest.TestCase):
    def asset(self, root: Path, name: str) -> dict[str, object]:
        path = root / name
        path.write_bytes(name.encode("utf-8"))
        return {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def test_record_preserves_positive_target_and_neutralizes_wind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = self.asset(root, "image.tif")
            plume = self.asset(root, "plume.tif")
            cloud_asset = self.asset(root, "cloud.tif")
            cohort = {
                "sample_id": "sample",
                "group_id": "group",
                "research_role": "auxiliary_training",
                "target_product": "S2A_target",
                "background_product": "S2B_reference",
                "source_center": [1.0, 2.0],
                "source_name": "source",
                "tile_date": "2025-01-01T00:00:00+00:00",
            }
            crop = {
                "sample_id": "sample",
                "group_id": "group",
                "research_role": "auxiliary_training",
                "sensor_family": "Sentinel-2",
                "target_product": "S2A_target",
                "background_product": "S2B_reference",
                "product_contract": {"band_order": list(MARS_BAND_ORDER)},
                "assets": {"image": image, "plume_mask": plume},
            }
            cloud = {
                "sample_id": "sample",
                "group_id": "group",
                "research_role": "auxiliary_training",
                "quality": {"gate_pass": True},
                "asset": cloud_asset,
            }
            record = model_record(root, cohort, crop, cloud)
            self.assertEqual(record["label_state"], "PLUME")
            self.assertTrue(record["pixel_truth_available"])
            self.assertEqual((record["wind_u"], record["wind_v"]), (0.0, 0.0))
            self.assertNotIn("plume_geometries", json.dumps(record))

    def test_sealed_record_is_refused_before_asset_verification(self) -> None:
        cohort = {
            "sample_id": "sealed",
            "group_id": "group",
            "research_role": "sealed_external",
        }
        with self.assertRaisesRegex(ValueError, "Non-development role"):
            model_record(Path("."), cohort, cohort, cohort)


if __name__ == "__main__":
    unittest.main()
