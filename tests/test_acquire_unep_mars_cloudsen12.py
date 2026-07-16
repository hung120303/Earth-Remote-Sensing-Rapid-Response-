from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from acquire_unep_mars_cloudsen12 import extra_band_url, pair_identity


class UnepCloudSenTests(unittest.TestCase):
    def test_extra_band_url_preserves_exact_product_object_prefix(self) -> None:
        b02 = "https://sentinel-s2-l1c.s3.amazonaws.com/tiles/38/T/PQ/2025/4/20/0/B02.jp2"
        self.assertEqual(
            extra_band_url(b02, "B8A"),
            "https://sentinel-s2-l1c.s3.amazonaws.com/tiles/38/T/PQ/2025/4/20/0/B8A.jp2",
        )

    def test_extra_band_url_rejects_local_or_nonofficial_source(self) -> None:
        with self.assertRaises(ValueError):
            extra_band_url("https://example.com/B02.jp2", "B01")

    def test_pair_identity_binds_crop_and_remote_assets(self) -> None:
        left = pair_identity({"target": "a"}, {"sample_id": "x"})
        right = pair_identity({"target": "b"}, {"sample_id": "x"})
        self.assertNotEqual(left, right)
        self.assertEqual(left, pair_identity({"target": "a"}, {"sample_id": "x"}))


if __name__ == "__main__":
    unittest.main()
