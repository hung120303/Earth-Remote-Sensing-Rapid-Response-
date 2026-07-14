from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from train_mars_oof_context_ranker import select_stable_candidate, stability_checks  # noqa: E402


class OOFContextRankerTests(unittest.TestCase):
    def test_stability_rejects_one_bad_fold(self) -> None:
        pooled = {
            "checks": {"pooled": True},
            "delta": {"recall_at_fpr_0_0713": 0.1},
        }
        per_fold = {
            "2": {"delta": {"recall_at_fpr_0_0713": 0.1, "average_precision": 0.1}},
            "3": {"delta": {"recall_at_fpr_0_0713": 0.1, "average_precision": 0.1}},
            "4": {"delta": {"recall_at_fpr_0_0713": -0.01, "average_precision": 0.1}},
        }
        self.assertFalse(stability_checks(pooled, per_fold, 100)["no_inner_fold_recall_regression"])

    def test_selection_prefers_stable_pass(self) -> None:
        fold = {"delta": {"recall_at_fpr_0_0713": 0.0}}
        failing = {"stability_checks": {"a": False}, "rank": [1, 1, 1], "per_fold": {"2": fold}}
        passing = {"stability_checks": {"a": True}, "rank": [0, 0, 0], "per_fold": {"2": fold}}
        self.assertIs(select_stable_candidate([failing, passing]), passing)


if __name__ == "__main__":
    unittest.main()
