from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import iter_development_manifest  # noqa: E402


class MarsMixedAdapterTests(unittest.TestCase):
    def write_manifest(self, records: list[dict[str, str]]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "samples.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def test_development_loader_accepts_only_development_roles(self) -> None:
        path = self.write_manifest(
            [
                {"sample_id": "train", "research_role": "development_training"},
                {
                    "sample_id": "validation",
                    "research_role": "development_validation",
                },
            ]
        )
        self.assertEqual(
            [record["sample_id"] for record in iter_development_manifest(path)],
            ["train", "validation"],
        )

    def test_development_loader_refuses_sealed_paper_test(self) -> None:
        path = self.write_manifest(
            [{"sample_id": "sealed", "research_role": "sealed_paper_test"}]
        )
        with self.assertRaisesRegex(ValueError, "refuses sealed role"):
            list(iter_development_manifest(path))

    def test_development_loader_refuses_unknown_role(self) -> None:
        path = self.write_manifest(
            [{"sample_id": "unknown", "research_role": "strict_spatial_test"}]
        )
        with self.assertRaisesRegex(ValueError, "Unsupported development role"):
            list(iter_development_manifest(path))


if __name__ == "__main__":
    unittest.main()
