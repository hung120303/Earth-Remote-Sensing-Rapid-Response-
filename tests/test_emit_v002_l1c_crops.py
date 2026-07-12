from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from acquire_emit_v002_l1c_crops import clear_support, gdal_env, write_raster


class ExternalCropContractTests(unittest.TestCase):
    def test_clear_support_combines_radiometry_and_scl_semantics(self) -> None:
        stack = np.ones((6, 2, 3), dtype=np.uint16)
        stack[:, 0, 0] = 0
        scl = np.array([[4, 8, 7], [3, 6, 0]], dtype=np.uint8)
        support = clear_support(stack, scl)
        expected = np.array([[False, False, True], [False, True, False]])
        np.testing.assert_array_equal(support["clear"], expected)
        self.assertAlmostEqual(support["clear_fraction"], 2.0 / 6.0, places=8)

    def test_gdal_environment_allows_l1c_jp2_and_l2a_tif(self) -> None:
        self.assertEqual(gdal_env()["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"], ".tif,.jp2")

    def test_write_raster_preserves_native_stack_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stack.tif"
            data = np.arange(6 * 16 * 16, dtype=np.uint16).reshape(6, 16, 16)
            write_raster(
                path,
                data,
                crs="EPSG:32636",
                transform=Affine(10, 0, 500000, 0, -10, 4000000),
                descriptions=["B02", "B03", "B04", "B08", "B11", "B12"],
                nodata=0,
            )
            with rasterio.open(path) as source:
                self.assertEqual(source.shape, (16, 16))
                self.assertEqual(source.count, 6)
                self.assertEqual(source.dtypes, ("uint16",) * 6)
                self.assertEqual(
                    source.descriptions, ("B02", "B03", "B04", "B08", "B11", "B12")
                )


if __name__ == "__main__":
    unittest.main()
