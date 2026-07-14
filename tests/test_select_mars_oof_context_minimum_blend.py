from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from select_mars_oof_context_minimum_blend import select_minimum_stable  # noqa: E402


class MinimumBlendTests(unittest.TestCase):
    def test_selection_uses_smallest_stable_blend(self) -> None:
        candidates = [
            {"blend_lambda": 0.5, "stability_checks": {"a": True}, "rank": [2, 2, 2]},
            {"blend_lambda": 0.25, "stability_checks": {"a": True}, "rank": [1, 1, 1]},
            {"blend_lambda": 0.125, "stability_checks": {"a": False}, "rank": [3, 3, 3]},
        ]
        self.assertEqual(select_minimum_stable(candidates)["blend_lambda"], 0.25)


if __name__ == "__main__":
    unittest.main()
