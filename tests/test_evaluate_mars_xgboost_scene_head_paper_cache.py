from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from evaluate_mars_xgboost_scene_head_paper_cache import operational_threshold  # noqa: E402


class XGBoostPaperEvaluatorTests(unittest.TestCase):
    def test_operational_threshold_uses_all_folds_conservatively(self) -> None:
        report = {
            "selected": {
                "per_fold": {
                    str(index): {
                        "versus_current": {
                            "metrics": {"operating_point": {"threshold": value}}
                        }
                    }
                    for index, value in enumerate([0.2, 0.4, 0.1, 0.3, 0.25])
                }
            }
        }
        self.assertEqual(operational_threshold(report), 0.4)

    def test_operational_threshold_requires_five_folds(self) -> None:
        report = {
            "selected": {
                "per_fold": {
                    "0": {
                        "versus_current": {
                            "metrics": {"operating_point": {"threshold": 0.2}}
                        }
                    }
                }
            }
        }
        with self.assertRaises(ValueError):
            operational_threshold(report)


if __name__ == "__main__":
    unittest.main()
