from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from diagnose_mars_scene_stacker_paper_cache import aligned_indices  # noqa: E402


class DiagnoseMarsSceneStackerPaperCacheTests(unittest.TestCase):
    def test_available_ids_are_aligned_in_available_order(self) -> None:
        aligned = np.asarray(["b", "a", "c"])
        available = np.asarray(["c", "b"])
        np.testing.assert_array_equal(aligned_indices(aligned, available), [2, 0])

    def test_unknown_available_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            aligned_indices(np.asarray(["a"]), np.asarray(["b"]))


if __name__ == "__main__":
    unittest.main()
