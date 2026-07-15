from __future__ import annotations
import sys, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; TOOLS=ROOT/"tools"
if str(TOOLS) not in sys.path: sys.path.insert(0,str(TOOLS))
from train_mars_adaptive_prithvi_probe import domain_normalize  # noqa: E402
class AdaptivePrithviTests(unittest.TestCase):
    def test_domains_receive_independent_finite_moments(self)->None:
        source=np.asarray([[0.,1.],[2.,1.]]); target=np.asarray([[10.,4.],[14.,4.]])
        left,right=domain_normalize(source,target)
        np.testing.assert_allclose(left.mean(0),0,atol=1e-8); np.testing.assert_allclose(right.mean(0),0,atol=1e-8)
        self.assertTrue(np.isfinite(left).all() and np.isfinite(right).all())
if __name__=="__main__": unittest.main()
