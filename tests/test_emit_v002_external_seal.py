from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from seal_emit_v002_external_cohort import model_observability


class ExternalCohortSealTests(unittest.TestCase):
    def test_model_observability_combines_both_passes_cloud_and_plume(self) -> None:
        target = np.ones((6, 4, 5), dtype=np.uint16)
        reference = np.ones((6, 4, 5), dtype=np.uint16)
        reference[0, 0, 0] = 0
        cloud = np.zeros((4, 5), dtype=np.uint8)
        cloud[0, 1] = 2
        plume = np.zeros((4, 5), dtype=np.uint8)
        plume[0, :4] = 1
        result = model_observability(target, reference, cloud, plume)
        self.assertEqual(result["model_observable_fraction"], 0.9)
        self.assertEqual(result["model_observable_fraction_on_plume"], 0.5)

    def test_model_observability_rejects_empty_plume(self) -> None:
        values = np.ones((6, 2, 2), dtype=np.uint16)
        with self.assertRaises(ValueError):
            model_observability(
                values,
                values,
                np.zeros((2, 2), dtype=np.uint8),
                np.zeros((2, 2), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
