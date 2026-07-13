from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from methanes2cm_adapter import (  # noqa: E402
    MODEL_BAND_INDICES,
    PATCH_SHAPE,
    STACK_SHAPE,
    load_sample,
    safe_sample_path,
    v4_input,
    v5_input,
)


def write_sample(root: Path, *, label: int = 1) -> dict[str, str]:
    directory = root / "123"
    directory.mkdir(parents=True)
    base = np.stack(
        [np.full(PATCH_SHAPE, 1000 + index * 100, dtype=np.uint16) for index in range(12)]
    )
    assert base.shape == STACK_SHAPE
    tifffile.imwrite(directory / "s2.tif", base)
    tifffile.imwrite(directory / "s2_pre.tif", base + 10)
    tifffile.imwrite(directory / "s2_pre_pre.tif", base + 20)
    # Match the published dataset's binary-valued float64 TIFF contract.
    mask = np.zeros(PATCH_SHAPE, dtype=np.float64)
    if label:
        mask[4:8, 6:10] = 1
    tifffile.imwrite(directory / "plume.tif", mask)
    return {
        "id": "123",
        "s2_path": "123/s2.tif",
        "s2_pre_path": "123/s2_pre.tif",
        "s2_pre_pre_path": "123/s2_pre_pre.tif",
        "plume_mask_path": "123/plume.tif",
        "label": str(label),
        "latitude": "31.25",
        "longitude": "-102.50",
    }


class MethaneS2CMAdapterTests(unittest.TestCase):
    def test_reads_twelve_tiff_pages_and_constructs_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = write_sample(root)
            sample = load_sample(root, record)
            self.assertEqual(sample.target.shape, (6, 32, 32))
            self.assertTrue(
                np.isclose(
                    sample.target[0, 0, 0],
                    (1000 + MODEL_BAND_INDICES[0] * 100) / 10_000,
                )
            )
            self.assertTrue(sample.observable_mask.all())
            self.assertEqual(int(sample.plume_mask.sum()), 16)
            self.assertEqual(v4_input(sample).shape, (16, 32, 32))
            self.assertEqual(v5_input(sample).shape, (20, 32, 32))

    def test_rejects_path_escape_and_label_mask_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                safe_sample_path(root, "../s2.tif", "s2.tif")
            record = write_sample(root, label=0)
            record["label"] = "1"
            with self.assertRaisesRegex(ValueError, "Label/mask disagreement"):
                load_sample(root, record)


if __name__ == "__main__":
    unittest.main()
