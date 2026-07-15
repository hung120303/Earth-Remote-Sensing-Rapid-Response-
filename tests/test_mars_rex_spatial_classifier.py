from __future__ import annotations
import sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];TOOLS=ROOT/"tools"
if str(TOOLS) not in sys.path:sys.path.insert(0,str(TOOLS))
from train_mars_rex_spatial_classifier import rex_objective  # noqa:E402
class RexSpatialTests(unittest.TestCase):
    def test_penalty_increases_for_unequal_environment_risk(self)->None:
        weights=torch.ones(4);env=torch.tensor([0,0,1,1]);equal=torch.tensor([1.,1.,1.,1.]);unequal=torch.tensor([0.,0.,2.,2.]);left,_=rex_objective(equal,env,weights,2.);right,_=rex_objective(unequal,env,weights,2.);self.assertGreater(float(right),float(left))
if __name__=="__main__":unittest.main()
