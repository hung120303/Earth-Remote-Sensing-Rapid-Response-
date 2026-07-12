from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from acquire_emit_v002_cloudsen12 import BAND_ORDER, LOCAL_BANDS, official_band_url, ordered_stack


class CloudSEN12ExternalContractTests(unittest.TestCase):
    def test_official_band_url_preserves_verified_l1c_tile(self) -> None:
        self.assertEqual(
            official_band_url(
                "https://sentinel-s2-l1c.s3.amazonaws.com/tiles/36/S/YA/2023/2/21/0/tileInfo.json",
                "B8A",
            ),
            "https://sentinel-s2-l1c.s3.amazonaws.com/tiles/36/S/YA/2023/2/21/0/B8A.jp2",
        )

    def test_official_band_url_rejects_non_l1c_authority(self) -> None:
        with self.assertRaises(ValueError):
            official_band_url(
                "https://sentinel-s2-l2a.s3.amazonaws.com/tiles/36/S/YA/tileInfo.json",
                "B01",
            )

    def test_ordered_stack_matches_exact_thirteen_band_contract(self) -> None:
        arrays = {band: np.full((2, 3), index, dtype=np.uint16) for index, band in enumerate(BAND_ORDER)}
        local = {band: arrays[band] for band in LOCAL_BANDS}
        extras = {band: arrays[band] for band in BAND_ORDER if band not in LOCAL_BANDS}
        stack = ordered_stack(local, extras)
        self.assertEqual(stack.shape, (13, 2, 3))
        self.assertEqual([int(stack[index, 0, 0]) for index in range(13)], list(range(13)))


if __name__ == "__main__":
    unittest.main()
