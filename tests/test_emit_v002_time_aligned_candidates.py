from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_emit_v002_time_aligned_candidates import (
    bbox_extent_km,
    haversine_km,
    independent_best,
)


def candidate(
    granule: str,
    *,
    center: tuple[float, float],
    offset: float,
    cloud: float,
    source: str,
    s2_scene: str,
) -> dict[str, object]:
    return {
        "granule_id": granule,
        "center": list(center),
        "offset_hours": offset,
        "scene_cloud_cover_pct": cloud,
        "source_scenes": [source],
        "s2_scene_id": s2_scene,
    }


class CandidateGroupingTests(unittest.TestCase):
    def test_haversine_distance_is_symmetric_and_geographic(self) -> None:
        one_degree_equator = haversine_km([0.0, 0.0], [1.0, 0.0])
        self.assertGreater(one_degree_equator, 111.0)
        self.assertLess(one_degree_equator, 112.0)
        self.assertEqual(haversine_km([1.0, 0.0], [0.0, 0.0]), one_degree_equator)

    def test_bbox_extent_reports_width_height_and_maximum(self) -> None:
        width, height, maximum = bbox_extent_km([0.0, 0.0, 0.01, 0.02])
        self.assertGreater(width, 1.0)
        self.assertGreater(height, 2.0)
        self.assertEqual(maximum, height)

    def test_independent_best_deduplicates_source_and_sentinel_scene(self) -> None:
        records = [
            candidate(
                "plume-a",
                center=(0.0, 0.0),
                offset=2.0,
                cloud=1.0,
                source="emit-source-a",
                s2_scene="s2-a",
            ),
            candidate(
                "plume-a-better",
                center=(0.01, 0.0),
                offset=0.5,
                cloud=2.0,
                source="emit-source-a",
                s2_scene="s2-b",
            ),
            candidate(
                "plume-same-s2",
                center=(20.0, 0.0),
                offset=1.0,
                cloud=0.0,
                source="emit-source-b",
                s2_scene="s2-b",
            ),
            candidate(
                "plume-independent",
                center=(40.0, 0.0),
                offset=1.5,
                cloud=0.0,
                source="emit-source-c",
                s2_scene="s2-c",
            ),
        ]
        selected = independent_best(records, radius_km=25.0)
        self.assertEqual(
            {item["granule_id"] for item in selected},
            {"plume-a-better", "plume-independent"},
        )
        self.assertEqual(len({item["s2_scene_id"] for item in selected}), len(selected))
        self.assertEqual(len({item["source_scenes"][0] for item in selected}), len(selected))

    def test_independent_best_collapses_connected_25km_group(self) -> None:
        records = [
            candidate(
                "near-worse",
                center=(0.0, 0.0),
                offset=1.0,
                cloud=0.0,
                source="source-a",
                s2_scene="s2-a",
            ),
            candidate(
                "near-better",
                center=(0.1, 0.0),
                offset=0.5,
                cloud=0.0,
                source="source-b",
                s2_scene="s2-b",
            ),
            candidate(
                "far",
                center=(1.0, 0.0),
                offset=2.0,
                cloud=0.0,
                source="source-c",
                s2_scene="s2-c",
            ),
        ]
        selected = independent_best(records, radius_km=25.0)
        self.assertEqual(
            {item["granule_id"] for item in selected}, {"near-better", "far"}
        )
        self.assertTrue(
            all(str(item["group_id"]).startswith("emit25km-") for item in selected)
        )


if __name__ == "__main__":
    unittest.main()
