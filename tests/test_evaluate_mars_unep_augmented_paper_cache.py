from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from evaluate_mars_unep_augmented_paper_cache import assert_paper_comparator, validate_candidate


class UnepAugmentedPaperCacheTests(unittest.TestCase):
    def contract(self) -> dict:
        return {
            "candidate": {
                "artifact_sha256": "hash",
                "candidate_logit_blend": 0.2,
                "operational_scene_threshold": 0.22,
            }
        }

    def test_candidate_must_be_promoted_and_hash_bound(self) -> None:
        artifact = {
            "kind": "mars_unep_positive_augmented_xgboost",
            "candidate_blend": 0.2,
            "operational_scene_threshold": 0.22,
        }
        development = {
            "all_promotion_gates_pass": True,
            "artifact": {"sha256": "hash"},
        }
        validate_candidate(artifact, development, self.contract())
        development["all_promotion_gates_pass"] = False
        with self.assertRaisesRegex(ValueError, "not promoted"):
            validate_candidate(artifact, development, self.contract())

    def test_threshold_mismatch_is_rejected(self) -> None:
        artifact = {
            "kind": "mars_unep_positive_augmented_xgboost",
            "candidate_blend": 0.2,
            "operational_scene_threshold": 0.3,
        }
        development = {
            "all_promotion_gates_pass": True,
            "artifact": {"sha256": "hash"},
        }
        with self.assertRaisesRegex(ValueError, "threshold"):
            validate_candidate(artifact, development, self.contract())

    def test_joint_candidate_uses_declared_promotion_gate_and_kind(self) -> None:
        contract = {
            "candidate": {
                "artifact_sha256": "joint",
                "artifact_kind": "mars_joint_calibration_aware_xgboost",
                "promotion_gate": "all_finalization_gates_pass",
                "candidate_logit_blend": 0.2,
                "operational_scene_threshold": 0.23,
            }
        }
        artifact = {
            "kind": "mars_joint_calibration_aware_xgboost",
            "candidate_blend": 0.2,
            "operational_scene_threshold": 0.23,
        }
        development = {
            "all_finalization_gates_pass": True,
            "artifact": {"sha256": "joint"},
        }
        validate_candidate(artifact, development, contract)


if __name__ == "__main__":
    unittest.main()
