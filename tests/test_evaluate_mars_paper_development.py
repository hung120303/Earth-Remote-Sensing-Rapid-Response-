from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from evaluate_mars_paper_development import assert_zero_residual_equivalence  # noqa: E402


class EvaluateMarsPaperDevelopmentTests(unittest.TestCase):
    def summary(self, delta: float = 0.0) -> dict[str, object]:
        return {
            "delta": {"average_precision": delta, "pixel_iou": 0.0},
            "sensor_strata": {
                "Sentinel-2": {
                    "eligible_for_promotion": True,
                    "delta": {"average_precision": 0.0, "pixel_iou": 0.0},
                },
                "Landsat": {
                    "eligible_for_promotion": True,
                    "delta": {"average_precision": 0.0, "pixel_iou": 0.0},
                },
            },
        }

    def test_exact_equivalence_passes(self) -> None:
        assert_zero_residual_equivalence(self.summary())

    def test_any_delta_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "differs"):
            assert_zero_residual_equivalence(self.summary(1e-12))


if __name__ == "__main__":
    unittest.main()
