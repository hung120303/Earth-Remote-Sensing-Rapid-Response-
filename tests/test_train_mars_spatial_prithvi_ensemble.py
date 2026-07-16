from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_spatial_prithvi_ensemble import align_prithvi_scores


class SpatialPrithviEnsembleTests(unittest.TestCase):
    def test_prithvi_alignment_uses_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            np.savez(path, sample_ids=np.asarray(["b", "a"]), scores=np.asarray([0.8, 0.2]))
            values = {"sample_ids": np.asarray(["a", "b"])}
            np.testing.assert_array_equal(align_prithvi_scores(values, path), [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()
