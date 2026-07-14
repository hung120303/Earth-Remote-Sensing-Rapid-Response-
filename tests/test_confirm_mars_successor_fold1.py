from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from confirm_mars_successor_fold1 import (  # noqa: E402
    DEFAULT_BASELINE_SHA256,
    DEFAULT_HEAD_SHA256,
    DEFAULT_RESIDUAL_SHA256,
    DEFAULT_SELECTION_SHA256,
)


class Fold1ConfirmationContractTests(unittest.TestCase):
    def test_all_inputs_are_pinned_sha256(self) -> None:
        for value in (
            DEFAULT_BASELINE_SHA256,
            DEFAULT_HEAD_SHA256,
            DEFAULT_RESIDUAL_SHA256,
            DEFAULT_SELECTION_SHA256,
        ):
            self.assertEqual(len(value), 64)
            int(value, 16)


if __name__ == "__main__":
    unittest.main()
