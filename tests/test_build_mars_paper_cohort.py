from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from build_mars_cohort import source_url_at_revision  # noqa: E402
from build_mars_paper_cohort import assets_for, role_counts, site_group  # noqa: E402


class MarsPaperCohortTests(unittest.TestCase):
    def test_revision_url_is_pinned_and_path_encoded(self) -> None:
        url = source_url_at_revision("folder/a file.tif", "abc123")
        self.assertIn("/abc123/folder/a%20file.tif", url)
        self.assertTrue(url.endswith("?download=true"))

    def test_scene_without_released_pixel_truth_needs_only_image_and_cloud(self) -> None:
        meta = {
            "id_loc_image": "sample",
            "s2path": "data/sample_s2.tif",
            "cloudmaskpath": "data/sample_cloud.tif",
            "plumepath": "",
            "ch4path": "",
        }
        assets = assets_for(meta, True, include_pixel_truth=False)
        self.assertEqual([asset["role"] for asset in assets], ["image", "cloud_mask"])
        with self.assertRaises(ValueError):
            assets_for(meta, True, include_pixel_truth=True)

    def test_role_counts_keep_sensor_and_site_totals(self) -> None:
        rows = [
            {
                "research_role": "train",
                "label_state": "PLUME",
                "physical_location_id": "a",
                "sensor_family": "Sentinel-2",
            },
            {
                "research_role": "train",
                "label_state": "NO_PLUME",
                "physical_location_id": "a",
                "sensor_family": "Landsat",
            },
        ]
        result = role_counts(rows)["train"]
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["sites"], 1)
        self.assertEqual((result["sentinel2"], result["landsat"]), (1, 1))

    def test_site_group_is_stable_and_site_specific(self) -> None:
        self.assertEqual(site_group("same"), site_group("same"))
        self.assertNotEqual(site_group("same"), site_group("different"))


if __name__ == "__main__":
    unittest.main()
