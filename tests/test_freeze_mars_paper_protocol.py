from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from freeze_mars_paper_protocol import assign_groups, assignment_hash  # noqa: E402


def records() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for site in range(15):
        for sensor in ("Sentinel-2", "Landsat"):
            result.append(
                {
                    "group_id": f"site_{site:02d}",
                    "label_state": "PLUME" if site % 3 == 0 else "NO_PLUME",
                    "sensor_family": sensor,
                }
            )
    return result


class MarsPaperProtocolTests(unittest.TestCase):
    def test_assignment_is_deterministic_and_site_disjoint(self) -> None:
        first = assign_groups(records(), n_folds=5, random_seed=7)
        second = assign_groups(records(), n_folds=5, random_seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 15)
        self.assertEqual(set(first.values()), set(range(5)))

    def test_assignment_hash_is_order_independent(self) -> None:
        first = {"b": 1, "a": 0}
        second = {"a": 0, "b": 1}
        self.assertEqual(assignment_hash(first), assignment_hash(second))

    def test_too_few_groups_fails(self) -> None:
        with self.assertRaises(ValueError):
            assign_groups(records()[:2], n_folds=5, random_seed=7)


if __name__ == "__main__":
    unittest.main()
