from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from extract_mars_paper_scene_features_label_free import (  # noqa: E402
    FORBIDDEN_TOKENS,
    OUTPUT_FIELDS,
    extract_payload,
)


class LabelFreePaperFeatureTests(unittest.TestCase):
    def test_payload_aligns_current_scores_without_outcomes(self) -> None:
        cache = {
            "available_ids": np.asarray(["b", "a"]),
            "available_groups": np.asarray(["g2", "g1"]),
            "available_base_features": np.asarray([[2.0], [1.0]], dtype=np.float32),
            "base_feature_names": np.asarray(["signal"]),
            "aligned_sample_ids": np.asarray(["a", "b", "c"]),
            "candidate_scores": np.asarray([0.1, 0.2, 0.3]),
            "labels": np.asarray([1, 0, 1]),
        }
        payload = extract_payload(cache)
        self.assertEqual(tuple(payload), OUTPUT_FIELDS)
        np.testing.assert_allclose(payload["current_v3_scores"], [0.2, 0.1])
        self.assertTrue(
            all(token not in name.lower() for name in payload for token in FORBIDDEN_TOKENS)
        )


if __name__ == "__main__":
    unittest.main()
