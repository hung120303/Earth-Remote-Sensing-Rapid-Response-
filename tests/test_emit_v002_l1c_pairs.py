from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_emit_v002_l1c_pairs import corresponding_l1c_id, mgrs_tile, official_l1c_url


class L1CPairContractTests(unittest.TestCase):
    def test_corresponding_l1c_id_preserves_acquisition_identity(self) -> None:
        self.assertEqual(
            corresponding_l1c_id("S2A_40QCK_20230202_0_L2A"),
            "S2A_40QCK_20230202_0_L1C",
        )

    def test_corresponding_l1c_id_rejects_non_l2a(self) -> None:
        with self.assertRaises(ValueError):
            corresponding_l1c_id("S2A_40QCK_20230202_0_L1C")

    def test_mgrs_tile_requires_declared_scene_contract(self) -> None:
        self.assertEqual(mgrs_tile("S2B_13RFQ_20241210_0_L2A"), "13RFQ")
        with self.assertRaises(ValueError):
            mgrs_tile("not-a-sentinel-scene")

    def test_official_l1c_url_corrects_l2a_catalog_bucket(self) -> None:
        url, corrected = official_l1c_url(
            "s3://sentinel-s2-l2a/tiles/40/Q/CK/2023/2/2/0/B02.jp2"
        )
        self.assertTrue(corrected)
        self.assertEqual(
            url,
            "https://sentinel-s2-l1c.s3.amazonaws.com/tiles/40/Q/CK/2023/2/2/0/B02.jp2",
        )


if __name__ == "__main__":
    unittest.main()
