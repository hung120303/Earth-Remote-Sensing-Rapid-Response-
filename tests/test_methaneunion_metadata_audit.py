from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from audit_methaneunion_metadata import (  # noqa: E402
    distance_summary,
    has_sensor,
    haversine_km,
    select_novel_s2_rows,
)


def row(identifier: int, latitude: float, longitude: float, sensor: str = "S2") -> dict[str, str]:
    return {
        "id": str(identifier),
        "label": str(identifier % 2),
        "latitude": str(latitude),
        "longitude": str(longitude),
        "available_sensor": sensor,
        "S2_t0_path": f"data/{identifier}/t0.tif",
        "S2_pre_path": f"data/{identifier}/pre.tif",
        "S2_pre_pre_path": f"data/{identifier}/pre_pre.tif",
        "S2_plume_label_path": f"data/{identifier}/mask.tif",
    }


class MethaneUnionMetadataAuditTests(unittest.TestCase):
    def test_haversine_known_distance(self) -> None:
        self.assertAlmostEqual(haversine_km((0.0, 0.0), (0.0, 1.0)), 111.195, places=3)

    def test_distance_summary_uses_unique_coordinates_and_strict_boundary(self) -> None:
        result = distance_summary([(0.0, 0.0), (0.0, 0.0), (0.0, 1.0)], [(0.0, 0.0)])
        self.assertEqual(result["coordinates"], 2)
        self.assertEqual(result["within_1km"], 1)
        self.assertEqual(result["beyond_25km"], 1)

    def test_sensor_membership_is_token_based(self) -> None:
        self.assertTrue(has_sensor({"available_sensor": "S2,L89"}, "S2"))
        self.assertFalse(has_sensor({"available_sensor": "S5p"}, "S2"))

    def test_novel_selection_filters_distance_and_sensor(self) -> None:
        rows = [
            row(1, 0.0, 0.0),
            row(2, 0.0, 1.0),
            row(3, 0.0, 2.0, sensor="L89"),
        ]
        selected = select_novel_s2_rows(rows, [(0.0, 0.0)], 25.0)
        self.assertEqual([item["id"] for item in selected], [2])
        self.assertGreater(selected[0]["nearest_known_km"], 25.0)

    def test_novel_s2_row_requires_all_four_paths(self) -> None:
        candidate = row(2, 0.0, 1.0)
        candidate["S2_plume_label_path"] = ""
        with self.assertRaises(ValueError):
            select_novel_s2_rows([candidate], [(0.0, 0.0)], 25.0)


if __name__ == "__main__":
    unittest.main()
