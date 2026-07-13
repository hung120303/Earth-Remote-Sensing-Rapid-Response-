from __future__ import annotations

import csv
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from acquire_methanes2cm_v5_train import (  # noqa: E402
    checked_destination,
    expected_members,
    pack_selected,
)


class MethaneS2CMAcquisitionTests(unittest.TestCase):
    def test_expected_members_and_safe_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "l2a_location_split_32x32"
            split.mkdir()
            csv_path = split / "train.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(
                    destination,
                    fieldnames=[
                        "id",
                        "s2_path",
                        "s2_pre_path",
                        "s2_pre_pre_path",
                        "plume_mask_path",
                        "label",
                        "latitude",
                        "longitude",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "7",
                        "s2_path": "7/s2.tif",
                        "s2_pre_path": "7/s2_pre.tif",
                        "s2_pre_pre_path": "7/s2_pre_pre.tif",
                        "plume_mask_path": "7/plume.tif",
                        "label": "0",
                        "latitude": "31.5",
                        "longitude": "-102.5",
                    }
                )
            expected, rows = expected_members(split, csv_path)
            archive_path = root / "part.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for name in expected:
                    dataset = expected[name][1]
                    if dataset == "mask":
                        values = np.zeros((32, 32), dtype=np.uint8)
                    else:
                        values = np.full((12, 32, 32), 1000, dtype=np.uint16)
                    buffer = io.BytesIO()
                    tifffile.imwrite(buffer, values)
                    payload = buffer.getvalue()
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                unrelated = b"sealed test bytes"
                info = tarfile.TarInfo("l2a_location_split_32x32/99/s2.tif")
                info.size = len(unrelated)
                archive.addfile(info, io.BytesIO(unrelated))
            packed_path = split / "train.h5"
            result = pack_selected([archive_path], expected, rows, packed_path)
            self.assertEqual(result["archive_members_seen"], 4)
            with h5py.File(packed_path, "r") as packed:
                self.assertEqual(packed["target"].shape, (1, 12, 32, 32))
                self.assertEqual(packed["mask"].shape, (1, 32, 32))
            self.assertFalse((split / "99").exists())

    def test_rejects_unsafe_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                checked_destination(Path(directory), "../7/s2.tif", "s2.tif")


if __name__ == "__main__":
    unittest.main()
