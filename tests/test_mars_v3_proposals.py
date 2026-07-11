from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from mars_v3_model import INPUT_CHANNELS  # noqa: E402
from mars_v3_proposals import (  # noqa: E402
    extract_proposals,
    label_proposal,
    proposal_feature_names,
    proposal_features,
)
from evaluate_mars_v3 import calibration_summary  # noqa: E402
from train_mars_v3 import MarsV3Dataset  # noqa: E402
from train_mars_v3_proposals import scene_scores  # noqa: E402


class MarsV3ProposalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.probability = np.zeros((32, 32), dtype=np.float32)
        self.probability[13:18, 5:26] = 0.90
        self.observable = np.ones((32, 32), dtype=bool)

    def test_extracts_and_deduplicates_nested_components(self) -> None:
        proposals = extract_proposals(self.probability, self.observable)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].area, 105)
        self.assertEqual(proposals[0].source_threshold, 0.70)

    def test_labels_positive_negative_and_ambiguous(self) -> None:
        proposal = extract_proposals(self.probability, self.observable)[0]
        positive_truth = np.zeros_like(self.observable)
        positive_truth[14:17, 8:23] = True
        self.assertEqual(label_proposal(proposal, positive_truth)["label"], 1)
        self.assertEqual(label_proposal(proposal, np.zeros_like(self.observable))["label"], 0)
        tiny_overlap = np.zeros_like(self.observable)
        tiny_overlap[13, 5] = True
        self.assertIsNone(label_proposal(proposal, tiny_overlap)["label"])

    def test_feature_contract_includes_learned_context(self) -> None:
        proposal = extract_proposals(self.probability, self.observable)[0]
        inputs = np.zeros((len(INPUT_CHANNELS), 32, 32), dtype=np.float32)
        inputs[0, 13:18, 5:26] = 1.2
        inputs[13] = 0.5
        decoder = np.ones((4, 32, 32), dtype=np.float32)
        features = proposal_features(
            proposal, self.probability, inputs, self.observable, decoder
        )
        self.assertEqual(features.size, len(proposal_feature_names(4)))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_ambiguous_proposals_remain_in_deployable_scene_score(self) -> None:
        cache = {
            "proposal_roles": np.asarray(["internal_validation", "internal_validation"]),
            "proposal_sample_ids": np.asarray(["scene-a", "scene-b"]),
            "scene_roles": np.asarray(["internal_validation", "internal_validation"]),
            "scene_ids": np.asarray(["scene-a", "scene-b"]),
            "scene_labels": np.asarray([1, 0], dtype=np.uint8),
            "scene_groups": np.asarray(["group-a", "group-b"]),
            # The first proposal's -1 training target is deliberately not
            # consulted: deployment scoring cannot use overlap truth.
            "y": np.asarray([-1, 0], dtype=np.int8),
        }
        labels, scores, groups = scene_scores(
            cache, np.asarray([0.97, 0.12]), "internal_validation"
        )
        np.testing.assert_array_equal(labels, [1, 0])
        np.testing.assert_allclose(scores, [0.97, 0.12])
        np.testing.assert_array_equal(groups, ["group-a", "group-b"])

    def test_augmentation_stream_advances_and_repeats_from_seed(self) -> None:
        first = MarsV3Dataset(Path("."), [], {}, augment=True, seed=303)
        second = MarsV3Dataset(Path("."), [], {}, augment=True, seed=303)
        first_values = first.augmentation_rng().integers(0, 1_000_000, size=8)
        advanced_values = first.augmentation_rng().integers(0, 1_000_000, size=8)
        repeated_values = second.augmentation_rng().integers(0, 1_000_000, size=8)
        np.testing.assert_array_equal(first_values, repeated_values)
        self.assertFalse(np.array_equal(first_values, advanced_values))

    def test_calibration_summary_reports_fixed_bin_ece(self) -> None:
        perfect = calibration_summary(
            np.asarray([0, 1], dtype=np.uint8), np.asarray([0.0, 1.0])
        )
        self.assertAlmostEqual(perfect["expected_calibration_error"], 0.0)
        overconfident = calibration_summary(
            np.asarray([0, 0], dtype=np.uint8), np.asarray([0.9, 0.9])
        )
        self.assertAlmostEqual(overconfident["expected_calibration_error"], 0.9)


if __name__ == "__main__":
    unittest.main()
