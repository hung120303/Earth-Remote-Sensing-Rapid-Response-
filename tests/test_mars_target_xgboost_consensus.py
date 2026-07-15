from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from train_mars_target_xgboost_consensus import consensus_scores  # noqa: E402

class TargetXGBoostConsensusTests(unittest.TestCase):
    def test_consensus_is_convex_in_logit_space(self) -> None:
        current, target, boosted = np.asarray([0.2, 0.8]), np.asarray([0.4, 0.6]), np.asarray([0.3, 0.7])
        scores = consensus_scores(current, target, boosted, 0.3, 0.1)
        logits = 0.6*np.log(current/(1-current)) + 0.3*np.log(target/(1-target)) + 0.1*np.log(boosted/(1-boosted))
        np.testing.assert_allclose(scores, 1/(1+np.exp(-logits)))

    def test_consensus_requires_anchor_weight(self) -> None:
        values = np.asarray([0.5])
        with self.assertRaises(ValueError):
            consensus_scores(values, values, values, 0.8, 0.2)

if __name__ == "__main__":
    unittest.main()
