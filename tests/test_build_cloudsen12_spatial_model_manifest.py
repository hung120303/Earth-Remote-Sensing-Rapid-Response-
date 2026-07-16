from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_cloudsen12_spatial_model_manifest import MARS_BAND_ORDER, model_record


class CloudsenSpatialManifestTests(unittest.TestCase):
    def asset(self, root: Path, name: str) -> dict[str, object]:
        path = root / name
        path.write_bytes(name.encode("utf-8"))
        return {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def test_negative_record_uses_published_wind_and_no_plume_asset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = self.asset(root, "image.tif")
            cloud = self.asset(root, "cloud.tif")
            cohort = {
                "sample_id": "sample",
                "group_id": "cloudsen12:roi",
                "research_role": "auxiliary_training",
                "label_state": "NO_PLUME",
                "target_product": "S2A_target",
                "background_product": "S2B_reference",
                "source_center": [1.0, 2.0],
                "source_name": "CloudSEN12+",
                "tile_date": "2025-01-01T00:00:00+00:00",
                "wind_u": 3.0,
                "wind_v": -2.0,
            }
            crop = {
                "sample_id": "sample",
                "group_id": "cloudsen12:roi",
                "research_role": "auxiliary_training",
                "label_state": "NO_PLUME",
                "sensor_family": "Sentinel-2",
                "target_product": "S2A_target",
                "background_product": "S2B_reference",
                "product_contract": {"band_order": list(MARS_BAND_ORDER)},
                "quality": {"gate_pass_before_cloud": True, "plume_pixels": 0},
                "assets": {"image": image},
            }
            record = model_record(root, cohort, crop, cloud)
            self.assertEqual(record["label_state"], "NO_PLUME")
            self.assertEqual((record["wind_u"], record["wind_v"]), (3.0, -2.0))
            self.assertEqual([item["role"] for item in record["assets"]], ["image", "cloud_mask"])

    def test_positive_or_sealed_record_is_refused(self) -> None:
        cohort = {"sample_id": "sample", "research_role": "sealed_external"}
        with self.assertRaisesRegex(ValueError, "Unsupported role"):
            model_record(Path("."), cohort, cohort, {})


if __name__ == "__main__":
    unittest.main()
