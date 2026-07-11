from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_v3_model import INPUT_CHANNELS  # noqa: E402
from mars_v3_proposals import (  # noqa: E402
    extract_proposals,
    label_proposal,
    proposal_feature_names,
    proposal_features,
)


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


if __name__ == "__main__":
    unittest.main()
