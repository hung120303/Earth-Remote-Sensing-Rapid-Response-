from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from acquire_stanford_large_controlled_release_l1c_crops import (
    BAND_ORDER,
    band_resampling,
    centered_native_window,
    expected_pair_count,
    pair_manifest_binding,
    read_native_window,
    stack_bands,
)


class RecordingSource:
    dtypes = ("uint16",)

    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, object]]] = []

    def read(self, index: int, **kwargs: object) -> np.ndarray:
        self.calls.append((index, kwargs))
        window = kwargs["window"]
        assert isinstance(window, Window)
        return np.full((int(window.height), int(window.width)), 17, dtype=np.uint16)


class StanfordL1CCropContractTests(unittest.TestCase):
    def test_band_order_shape_and_raw_uint16_contract(self) -> None:
        self.assertEqual(BAND_ORDER, ("B02", "B03", "B04", "B08", "B11", "B12"))
        bands = {
            band: np.full((256, 256), index, dtype=np.uint16)
            for index, band in enumerate(reversed(BAND_ORDER), start=1)
        }

        stack = stack_bands(bands, size=256)

        self.assertEqual(stack.shape, (6, 256, 256))
        self.assertEqual(stack.dtype, np.uint16)
        self.assertEqual(
            [int(stack[index, 0, 0]) for index in range(6)],
            [int(bands[band][0, 0]) for band in BAND_ORDER],
        )
        self.assertNotIn("cloud", {band.lower() for band in BAND_ORDER})
        self.assertNotIn("scl", {band.lower() for band in BAND_ORDER})

    def test_stack_rejects_non_raw_or_wrong_shape_bands(self) -> None:
        raw = {band: np.zeros((256, 256), dtype=np.uint16) for band in BAND_ORDER}
        non_raw = dict(raw)
        non_raw["B11"] = np.zeros((256, 256), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "raw uint16"):
            stack_bands(non_raw, size=256)
        wrong_shape = dict(raw)
        wrong_shape["B12"] = np.zeros((255, 256), dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "256x256"):
            stack_bands(wrong_shape, size=256)

    def test_centered_window_is_fixed_256_and_stays_on_native_grid(self) -> None:
        transform = Affine(10, 0, 500000, 0, -10, 4000000)
        window = centered_native_window(
            transform,
            width=10980,
            height=10980,
            projected_x=510005,
            projected_y=3989995,
            size=256,
        )

        self.assertEqual((window.width, window.height), (256, 256))
        self.assertEqual(window.col_off, int(window.col_off))
        self.assertEqual(window.row_off, int(window.row_off))
        center_col = int(window.col_off) + 128
        center_row = int(window.row_off) + 128
        x0, y0 = transform * (center_col, center_row)
        x1, y1 = transform * (center_col + 1, center_row + 1)
        self.assertLessEqual(min(x0, x1), 510005)
        self.assertLessEqual(510005, max(x0, x1))
        self.assertLessEqual(min(y0, y1), 3989995)
        self.assertLessEqual(3989995, max(y0, y1))

    def test_native_read_is_explicitly_windowed(self) -> None:
        source = RecordingSource()
        window = Window(100, 200, 256, 256)

        data = read_native_window(source, window=window, size=256)

        self.assertEqual(data.shape, (256, 256))
        self.assertEqual(len(source.calls), 1)
        _, kwargs = source.calls[0]
        self.assertEqual(kwargs["window"], window)
        self.assertFalse(bool(kwargs["boundless"]))
        self.assertEqual(kwargs["out_dtype"], "uint16")

    def test_only_20m_bands_use_bilinear_resampling(self) -> None:
        self.assertIs(band_resampling("B02"), Resampling.nearest)
        self.assertIs(band_resampling("B03"), Resampling.nearest)
        self.assertIs(band_resampling("B04"), Resampling.nearest)
        self.assertIs(band_resampling("B08"), Resampling.nearest)
        self.assertIs(band_resampling("B11"), Resampling.bilinear)
        self.assertIs(band_resampling("B12"), Resampling.bilinear)

    def test_pair_count_defaults_to_casa_and_accepts_bound_evanston_count(self) -> None:
        self.assertEqual(expected_pair_count({}), 169)
        self.assertEqual(
            expected_pair_count({"source": {"target_manifest": {"rows": 9}}}),
            9,
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            expected_pair_count({"source": {"target_manifest": {"rows": 0}}})

    def test_pair_receipt_accepts_original_and_namespaced_binding_shapes(self) -> None:
        binding = {"path": "ignored/pairs.json", "sha256": "abc"}
        self.assertEqual(pair_manifest_binding({"pair_manifest": binding}), binding)
        self.assertEqual(
            pair_manifest_binding({"bindings": {"pair_manifest": binding}}),
            binding,
        )
        with self.assertRaisesRegex(ValueError, "lacks a pair-manifest binding"):
            pair_manifest_binding({})


if __name__ == "__main__":
    unittest.main()
