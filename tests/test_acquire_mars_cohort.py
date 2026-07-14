from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from acquire_mars_cohort import incomplete_items, load_manifest_asset_paths  # noqa: E402


class AcquireMarsCohortTests(unittest.TestCase):
    def write(self, records: list[dict[str, object]]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "manifest.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def test_manifest_asset_paths_are_unique(self) -> None:
        path = self.write(
            [
                {"assets": [{"path": "a.tif"}, {"path": "shared.tif"}]},
                {"assets": [{"path": "b.tif"}, {"path": "shared.tif"}]},
            ]
        )
        self.assertEqual(
            load_manifest_asset_paths(path), {"a.tif", "b.tif", "shared.tif"}
        )

    def test_manifest_without_assets_fails(self) -> None:
        path = self.write([{"assets": []}])
        with self.assertRaisesRegex(ValueError, "no asset list"):
            load_manifest_asset_paths(path)

    def test_incomplete_items_skips_matching_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "complete.bin"
            complete.write_bytes(b"1234")
            items = [
                {"path": "complete.bin", "size": 4},
                {"path": "missing.bin", "size": 4},
            ]
            self.assertEqual(
                [item["path"] for item in incomplete_items(root, items)],
                ["missing.bin"],
            )


if __name__ == "__main__":
    unittest.main()
