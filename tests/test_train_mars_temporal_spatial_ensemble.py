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

from train_mars_temporal_spatial_ensemble import align_spatial_scores, score_candidate


class TemporalSpatialEnsembleTests(unittest.TestCase):
    def test_score_alignment_uses_sample_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            np.savez(
                path,
                inner_sample_ids=np.asarray(["b"]), inner_scores=np.asarray([0.2]),
                fold0_sample_ids=np.asarray(["a"]), fold0_scores=np.asarray([0.1]),
                fold1_sample_ids=np.asarray(["c"]), fold1_scores=np.asarray([0.3]),
            )
            values = {"sample_ids": np.asarray(["c", "a", "b"])}
            np.testing.assert_allclose(align_spatial_scores(values, path), [0.3, 0.1, 0.2])

    def test_zero_weight_is_exact_spatial_score(self) -> None:
        spatial = np.asarray([0.1, 0.8, 0.3])
        result = score_candidate(spatial, np.asarray(["a", "a", "b"]), 1, 1, 0.0)
        np.testing.assert_array_equal(result, spatial)
        self.assertIsNot(result, spatial)


if __name__ == "__main__":
    unittest.main()
