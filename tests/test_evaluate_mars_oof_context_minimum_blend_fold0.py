from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from evaluate_mars_oof_context_minimum_blend_fold0 import (  # noqa: E402
    DEFAULT_ARTIFACT_SHA256,
    DEFAULT_CACHE_SHA256,
    DEFAULT_SELECTION_SHA256,
)


class MinimumBlendFold0ContractTests(unittest.TestCase):
    def test_frozen_hashes_are_sha256(self) -> None:
        for value in (DEFAULT_ARTIFACT_SHA256, DEFAULT_CACHE_SHA256, DEFAULT_SELECTION_SHA256):
            self.assertEqual(len(value), 64)
            int(value, 16)


if __name__ == "__main__":
    unittest.main()
