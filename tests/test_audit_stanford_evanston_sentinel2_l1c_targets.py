from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from audit_stanford_evanston_sentinel2_l1c_targets import (
    load_protocol,
    select_target_item,
)

CENTER = [-110.931318658838, 41.275737966074]
EVENT = {"event_id": "08092024_S2A", "utc_date": "2024-08-09", "platform": "S2A"}


def item(
    scene_id: str,
    *,
    acquired: str = "2024-08-09T18:12:01+00:00",
    cloud: float | None = 5.0,
    collection: str = "sentinel-2-l1c",
    bbox: list[float] | None = None,
    missing_asset: str | None = None,
) -> dict[str, object]:
    assets = {
        "blue": {},
        "green": {},
        "red": {},
        "nir": {},
        "swir16": {},
        "swir22": {},
        "tileinfo_metadata": {},
    }
    if missing_asset is not None:
        assets.pop(missing_asset)
    properties: dict[str, object] = {"datetime": acquired}
    if cloud is not None:
        properties["eo:cloud_cover"] = cloud
    return {
        "id": scene_id,
        "collection": collection,
        "bbox": bbox or [-111.2, 41.0, -110.7, 41.5],
        "properties": properties,
        "assets": assets,
    }


class EvanstonTargetAuditTests(unittest.TestCase):
    def test_select_target_is_exact_platform_date_point_and_l1c(self) -> None:
        features = [
            item("S2B_12TWL_20240809_0_L1C", cloud=0.0),
            item("S2A_12TWL_20240808_0_L1C", acquired="2024-08-08T18:12:01+00:00"),
            item("S2A_12TWL_20240809_1_L1C", bbox=[-110.0, 41.0, -109.0, 42.0]),
            item("S2A_12TWL_20240809_2_L1C", collection="sentinel-2-l2a"),
            item("S2A_12TWL_20240809_3_L1C", missing_asset="tileinfo_metadata"),
            item("S2A_12TWL_20240809_4_L1C", cloud=None),
            item("S2A_12TWL_20240809_7_L1C", cloud=7.0),
            item("S2A_12TWL_20240809_6_L1C", cloud=3.0),
            item("S2A_12TWL_20240809_5_L1C", cloud=3.0),
        ]

        selected, eligible_count = select_target_item(features, EVENT, CENTER)

        self.assertEqual(eligible_count, 3)
        self.assertEqual(selected["id"], "S2A_12TWL_20240809_5_L1C")

    def test_select_target_refuses_absent_exact_product(self) -> None:
        with self.assertRaisesRegex(ValueError, "No exact-date S2A"):
            select_target_item(
                [item("S2B_12TWL_20240809_0_L1C")],
                EVENT,
                CENTER,
            )

    def test_protocol_validation_refuses_event_identity_drift(self) -> None:
        protocol = json.loads(
            (ROOT / "configs/stanford_evanston_sentinel2_l1c_target_audit_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        protocol["events"][0]["platform"] = "S2B"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity is inconsistent"):
                load_protocol(path)

    def test_frozen_protocol_has_nine_unique_events(self) -> None:
        protocol = load_protocol(
            ROOT / "configs/stanford_evanston_sentinel2_l1c_target_audit_protocol.json"
        )
        event_ids = [event["event_id"] for event in protocol["events"]]
        self.assertEqual(len(event_ids), 9)
        self.assertEqual(len(event_ids), len(set(event_ids)))


if __name__ == "__main__":
    unittest.main()
