from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from train_mars_low_prevalence_set_context_head import build_current_site_context


class LowPrevalenceSetContextTests(unittest.TestCase):
    def test_context_is_permutation_equivariant(self) -> None:
        scores = np.asarray([0.1, 0.8, 0.2, 0.4], dtype=np.float64)
        groups = np.asarray(["a", "a", "b", "b"])
        expected, names = build_current_site_context(scores, groups)
        order = np.asarray([2, 0, 3, 1])
        permuted, permuted_names = build_current_site_context(scores[order], groups[order])
        inverse = np.argsort(order)
        self.assertEqual(names, permuted_names)
        np.testing.assert_allclose(expected, permuted[inverse], rtol=0.0, atol=1e-12)

    def test_singleton_leave_one_out_is_self(self) -> None:
        values, names = build_current_site_context(np.asarray([0.25]), np.asarray(["only"]))
        logit_column = names.index("set_current_logit")
        loo_column = names.index("set_current_logit_leave_one_out_max")
        self.assertAlmostEqual(values[0, logit_column], values[0, loo_column])


if __name__ == "__main__":
    unittest.main()
