from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from acquire_mars_hf_xet import (  # noqa: E402
    ASSET_REVISION,
    catalog_revision,
    load_worklist,
    select_manifest_catalog,
    write_worklist,
)


class AcquireMarsHfXetTests(unittest.TestCase):
    def item(self, path: str, revision: str = ASSET_REVISION) -> dict[str, object]:
        return {
            "path": path,
            "source_url": f"https://huggingface.co/datasets/x/resolve/{revision}/{path}",
        }

    def test_manifest_selection_is_exact_and_sorted(self) -> None:
        catalog = [self.item("b.tif"), self.item("a.tif")]
        selected = select_manifest_catalog(catalog, {"b.tif"})
        self.assertEqual([item["path"] for item in selected], ["b.tif"])

    def test_manifest_asset_absence_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent"):
            select_manifest_catalog([self.item("a.tif")], {"missing.tif"})

    def test_catalog_revision_must_match_frozen_asset_commit(self) -> None:
        self.assertEqual(catalog_revision([self.item("a.tif")]), ASSET_REVISION)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            catalog_revision([self.item("a.tif", "0" * 40)])

    def test_worklist_is_bound_to_catalog_and_manifest(self) -> None:
        selected = [self.item("a.tif"), self.item("b.tif")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worklist.json"
            write_worklist(
                path,
                [selected[1]],
                catalog_sha256="catalog",
                manifest_sha256="manifest",
            )
            loaded = load_worklist(
                path,
                selected,
                catalog_sha256="catalog",
                manifest_sha256="manifest",
            )
            self.assertEqual([item["path"] for item in loaded], ["b.tif"])
            with self.assertRaisesRegex(ValueError, "catalog"):
                load_worklist(
                    path,
                    selected,
                    catalog_sha256="different",
                    manifest_sha256="manifest",
                )


if __name__ == "__main__":
    unittest.main()
