from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_cloudsen12_fresh_test_model_manifest import published_cloud_lookup


class FreshCloudsenManifestTests(unittest.TestCase):
    def test_published_cloud_lookup_rejects_incomplete_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.csv"
            stats = root / "stats.csv"
            metadata.write_text("id_loc_image,location_name\nsample,location\n", encoding="utf-8")
            stats.write_text(
                "id_loc_image,cloudmask_0.0,cloudmask_1.0\nlocation,39999,1\n",
                encoding="utf-8",
            )
            self.assertEqual(
                published_cloud_lookup(metadata, stats)["sample"],
                {"clear": 39999, "nonclear": 1},
            )
            stats.write_text(
                "id_loc_image,cloudmask_0.0,cloudmask_1.0\nlocation,39998,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "pixel count changed"):
                published_cloud_lookup(metadata, stats)


if __name__ == "__main__":
    unittest.main()
