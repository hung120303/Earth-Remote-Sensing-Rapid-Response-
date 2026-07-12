from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_emit_v002_l1c_pairs import corresponding_l1c_id, mgrs_tile


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


if __name__ == "__main__":
    unittest.main()
