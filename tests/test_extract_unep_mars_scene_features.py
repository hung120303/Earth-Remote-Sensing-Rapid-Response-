from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from extract_unep_mars_scene_features import validate_records


class UnepSceneFeatureTests(unittest.TestCase):
    def test_role_and_positive_contract(self) -> None:
        records = [
            {
                "sample_id": "one",
                "research_role": "auxiliary_training",
                "label_state": "PLUME",
            }
        ]
        validate_records(records, "auxiliary_training", 1)
        with self.assertRaisesRegex(ValueError, "positive-only"):
            validate_records(
                [{**records[0], "label_state": "NO_PLUME"}],
                "auxiliary_training",
                1,
            )

    def test_duplicate_samples_are_rejected(self) -> None:
        record = {
            "sample_id": "one",
            "research_role": "development",
            "label_state": "PLUME",
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_records([record, record], "development", 2)


if __name__ == "__main__":
    unittest.main()
