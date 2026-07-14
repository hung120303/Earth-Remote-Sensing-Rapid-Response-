from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from evaluate_mars_context_scene_ranker_fold0 import DEFAULT_ARTIFACT_SHA256, DEFAULT_CACHE_SHA256  # noqa: E402


class ContextFold0ContractTests(unittest.TestCase):
    def test_frozen_hashes_are_full_sha256(self) -> None:
        self.assertEqual(len(DEFAULT_ARTIFACT_SHA256), 64)
        self.assertEqual(len(DEFAULT_CACHE_SHA256), 64)
        int(DEFAULT_ARTIFACT_SHA256, 16)
        int(DEFAULT_CACHE_SHA256, 16)


if __name__ == "__main__":
    unittest.main()
