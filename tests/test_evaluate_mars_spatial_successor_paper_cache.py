from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from diagnose_mars_scene_stacker_paper_cache import aligned_indices  # noqa: E402


class MarsSpatialSuccessorPaperCacheTests(unittest.TestCase):
    def test_spatial_rows_align_into_exact_paper_order(self) -> None:
        exact = np.asarray(["missing", "a", "b"])
        spatial = np.asarray(["b", "a"])
        np.testing.assert_array_equal(aligned_indices(exact, spatial), [2, 1])


if __name__ == "__main__":
    unittest.main()
