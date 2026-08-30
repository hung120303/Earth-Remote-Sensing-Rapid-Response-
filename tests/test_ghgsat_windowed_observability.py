from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import acquire_ghgsat_windowed_observability as audit


class FrozenValidationTests(unittest.TestCase):
    def test_exact_protocol_and_all_hash_bound_inputs_validate(self) -> None:
        protocol = audit.load_protocol()
        receipts = audit.validate_frozen_inputs(protocol)
        self.assertEqual(set(receipts), {
            "reference_catalog_protocol", "reference_catalog_report",
            "target_reference_pairs", "selected_targets", "cloudsen12_weights",
        })
        self.assertEqual(receipts["cloudsen12_weights"]["sha256"],
                         "218fa69aa3c7212d4e690b48af88ac6f3c976fc50d07f275b8fd623909183d7a")

    def test_all_76_pairs_join_exactly(self) -> None:
        rows = audit.join_frozen_pairs(audit.load_protocol())
        self.assertEqual(len(rows), 76)
        self.assertEqual(sum(row["target_sensor"] == audit.S2_SENSOR for row in rows), 47)
        self.assertEqual(sum(row["target_sensor"] == audit.LANDSAT_SENSOR for row in rows), 29)
        self.assertTrue(all(isinstance(row["representative_longitude"], (int, float)) for row in rows))

    def test_default_mode_has_zero_access_boundary(self) -> None:
        with mock.patch.object(audit, "MetadataClient", side_effect=AssertionError("client")), \
             mock.patch.object(audit, "load_cloudsen_model", side_effect=AssertionError("model")), \
             mock.patch.object(audit, "target_grid", side_effect=AssertionError("raster")):
            result = audit.validation_plan()
        self.assertEqual(result["pairs_joined"], 76)
        self.assertFalse(result["network_client_created"])
        self.assertFalse(result["asset_url_observed"])
        self.assertFalse(result["remote_raster_opened"])
        self.assertFalse(result["cloudsen_model_loaded"])
        self.assertFalse(result["outputs_written"])

    def test_production_cli_has_no_subset_or_threshold_overrides(self) -> None:
        actions = {action.dest for action in audit.build_parser()._actions}
        self.assertEqual(actions, {"help", "execute_network"})

    def test_offline_default_command_succeeds(self) -> None:
        completed = subprocess.run(
            [sys.executable, "tools/acquire_ghgsat_windowed_observability.py"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["mode"], "validation_only")
        self.assertFalse(payload["network_executed"])


class CrosswalkAndAssetTests(unittest.TestCase):
    CDSE = "S2A_MSIL1C_20220429T051701_N0400_R062_T43RFM_20220429T072408"

    @staticmethod
    def s2_item(item_id: str = "mirror-b") -> dict[str, object]:
        return {
            "type": "Feature", "id": item_id, "collection": "sentinel-2-l1c",
            "properties": {"s2:product_uri": "S2A_MSIL1C_20220429T051701_N0500_R062_T43RFM_20240101T000000.SAFE"},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]},
            "assets": {},
        }

    def test_crosswalk_enforces_spacecraft_time_mgrs_and_lexical_tie_break(self) -> None:
        later = self.s2_item("z-record")
        earlier = self.s2_item("a-record")
        wrong = self.s2_item("wrong")
        wrong["properties"] = {"s2:product_uri": "S2B_MSIL1C_20220429T051701_N0500_R062_T43RFM_20240101T000000.SAFE"}
        chosen = audit.select_s2_crosswalk([later, wrong, earlier], self.CDSE, longitude=1, latitude=1)
        self.assertEqual(chosen["item_id"], "a-record")
        self.assertEqual(chosen["cdse_product_uri"], self.CDSE)
        self.assertNotEqual(chosen["cdse_product_uri"], chosen["mirror_product_uri"])

    def test_crosswalk_rejects_more_than_one_second_or_wrong_tile(self) -> None:
        wrong_time = self.s2_item()
        wrong_time["properties"] = {"s2:product_uri": "S2A_MSIL1C_20220429T051703_N0500_R062_T43RFM_20240101T000000.SAFE"}
        wrong_tile = self.s2_item()
        wrong_tile["properties"] = {"s2:product_uri": "S2A_MSIL1C_20220429T051701_N0500_R062_T43RFN_20240101T000000.SAFE"}
        with self.assertRaises(audit.ObservabilityAuditError):
            audit.select_s2_crosswalk([wrong_time, wrong_tile], self.CDSE, longitude=1, latitude=1)

    def test_sentinel_assets_share_official_tile_root_and_derive_extras(self) -> None:
        root = audit.S2_PREFIX + "tiles/43/R/FM/2022/4/29/0"
        aliases = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08",
                   "swir16": "B11", "swir22": "B12"}
        assets = {name: {"href": f"{root}/{band}.jp2"} for name, band in aliases.items()}
        assets["tileinfo_metadata"] = {"href": f"{root}/tileInfo.json"}
        resolved = audit.validate_s2_assets({"assets": assets})
        self.assertEqual(resolved["B01"], f"{root}/B01.jp2")
        self.assertEqual(resolved["B10"], f"{root}/B10.jp2")
        bad = json.loads(json.dumps(assets))
        bad["red"]["href"] = audit.S2_PREFIX + "other/B04.jp2"
        with self.assertRaisesRegex(audit.ObservabilityAuditError, "one tile root"):
            audit.validate_s2_assets({"assets": bad})

    def test_asset_url_allowlists_reject_wrong_hosts_and_signed_urls(self) -> None:
        with self.assertRaises(audit.ObservabilityAuditError):
            audit._asset_href({"href": audit.S2_PREFIX + "x/B02.jp2?token=secret"}, name="B02")
        item = {
            "type": "Feature", "id": "LC09_L1TP_123045_20230101_20230102_02_T1",
            "collection": "landsat-c2l1",
            "properties": {"landsat:correction": "L1TP", "landsat:collection_category": "T1"},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]},
            "assets": {name: {"href": "https://evil.example/" + name + ".TIF"}
                       for name in (*audit.LANDSAT_TRAINING_BANDS, "QA_PIXEL")},
        }
        with self.assertRaisesRegex(audit.ObservabilityAuditError, "outside official"):
            audit.validate_landsat_item(item, item["id"])

    def test_tileinfo_binds_product_mgrs_and_sensing_time(self) -> None:
        record = {
            "mirror_product_uri": "S2A_MSIL1C_20220429T051701_N0500_R062_T43RFM_20240101T000000.SAFE",
            "mgrs_tile": "43RFM",
        }
        tileinfo = {"productName": record["mirror_product_uri"].removesuffix(".SAFE"),
                    "utmZone": 43, "latitudeBand": "R", "gridSquare": "FM",
                    "timestamp": "2022-04-29T05:17:01Z"}
        audit.validate_tileinfo(tileinfo, record)
        tileinfo["gridSquare"] = "FN"
        with self.assertRaisesRegex(audit.ObservabilityAuditError, "MGRS"):
            audit.validate_tileinfo(tileinfo, record)

    def test_exact_landsat_item_and_required_assets(self) -> None:
        item_id = "LC08_L1TP_153042_20220421_20220428_02_T1"
        roles = {"blue": "B2", "green": "B3", "red": "B4", "nir08": "B5",
                 "swir16": "B6", "swir22": "B7", "qa_pixel": "QA_PIXEL"}
        assets = {role: {"href": audit.LANDSAT_PREFIX + "2022/153/042/" + item_id + f"/{item_id}_{band}.TIF"}
                  for role, band in roles.items()}
        item = {"type": "Feature", "id": item_id, "collection": "landsat-c2l1",
                "properties": {"landsat:correction": "L1TP", "landsat:collection_category": "T1"},
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]},
                "assets": assets}
        self.assertEqual(audit.validate_landsat_item(item, item_id)["item_id"], item_id)
        item["properties"]["landsat:collection_category"] = "T2"
        with self.assertRaises(audit.ObservabilityAuditError):
            audit.validate_landsat_item(item, item_id)

    def test_failure_tolerant_resolution_attempts_later_physical_items(self) -> None:
        rows = [
            {"target_item_id": "bad", "reference_item_id": "good", "site_ID": "s1", "obs_ID": "o1",
             "date": "2022-01-01", "sat_ID": 1, "target_sensor": audit.LANDSAT_SENSOR,
             "component_id": "c1", "representative_longitude": 1.0, "representative_latitude": 1.0},
            {"target_item_id": "good", "reference_item_id": "good", "site_ID": "s2", "obs_ID": "o2",
             "date": "2022-01-02", "sat_ID": 1, "target_sensor": audit.LANDSAT_SENSOR,
             "component_id": "c2", "representative_longitude": 1.0, "representative_latitude": 1.0},
        ]
        calls: list[str] = []

        class Client:
            def json(self, method, endpoint, *, body=None):
                item_id = endpoint.rsplit("/", 1)[-1]
                calls.append(item_id)
                if item_id == "bad":
                    raise audit.ObservabilityAuditError("synthetic unavailable item")
                return {"id": "good"}

        good = {"item_id": "good", "geometry": {"type": "Polygon", "coordinates": []},
                "assets": {}, "sensor": audit.LANDSAT_SENSOR}
        with mock.patch.object(audit, "validate_landsat_item", return_value=good), \
             mock.patch.object(audit.reference, "geometry_covers_point", return_value=True):
            resolved, failures = audit.resolve_assets(
                rows, audit.load_protocol(), Client(), continue_on_pair_failure=True
            )
        self.assertCountEqual(calls, ["bad", "good"])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["resolution_status"], "unavailable_or_invalid")

    def test_resolution_deduplicates_sentinel_processing_editions(self) -> None:
        first = self.CDSE
        second = "S2A_MSIL1C_20220429T051701_N0500_R062_T43RFM_20240101T000000"
        root = audit.S2_PREFIX + "tiles/43/R/FM/2022/4/29/0"
        item = self.s2_item("mirror")
        item["assets"] = {
            name: {"href": f"{root}/{band}.jp2"}
            for name, band in {"blue": "B02", "green": "B03", "red": "B04",
                               "nir": "B08", "swir16": "B11", "swir22": "B12"}.items()
        }
        item["assets"]["tileinfo_metadata"] = {"href": f"{root}/tileInfo.json"}
        tileinfo = {"productName": str(item["properties"]["s2:product_uri"]).removesuffix(".SAFE"),
                    "utmZone": 43, "latitudeBand": "R", "gridSquare": "FM",
                    "timestamp": "2022-04-29T05:17:01Z"}

        class Client:
            calls: list[tuple[str, str]] = []
            def json(self, method, endpoint, *, body=None):
                self.calls.append((method, endpoint))
                return {"type": "FeatureCollection", "features": [item]} if method == "POST" else tileinfo

        row = {"target_item_id": first, "reference_item_id": second,
               "representative_longitude": 1.0, "representative_latitude": 1.0}
        client = Client()
        resolved = audit.resolve_assets([row], audit.load_protocol(), client)
        self.assertEqual(sum(method == "POST" for method, _ in client.calls), 1)
        self.assertIs(resolved[0]["target"], resolved[0]["reference"])


class WindowAndGateTests(unittest.TestCase):
    def test_window_reader_never_calls_unwindowed_read(self) -> None:
        class Source:
            crs = "EPSG:32643"
            transform = Affine(10, 0, 0, 0, -10, 4000)
            width = height = 400
            nodata = 0
            read_kwargs: dict[str, object] | None = None
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def read(self, *args, **kwargs):
                self.read_kwargs = kwargs
                window = kwargs["window"]
                return np.ones((int(window.height), int(window.width)), dtype=np.uint16)
            def window_transform(self, window):
                return Affine(10, 0, window.col_off * 10, 0, -10, 4000 - window.row_off * 10)
        source = Source()
        destination_transform = Affine(10, 0, 1000, 0, -10, 3000)
        with mock.patch("rasterio.open", return_value=source), \
             mock.patch("rasterio.warp.reproject", side_effect=lambda **kwargs: kwargs["destination"].fill(1)):
            from rasterio.enums import Resampling
            result = audit.read_window_to_grid("https://example.invalid/B02.jp2", dst_crs=source.crs,
                                               dst_transform=destination_transform,
                                               resampling=Resampling.nearest, dtype="uint16")
        self.assertEqual(result.shape, (200, 200))
        self.assertIn("window", source.read_kwargs)
        self.assertFalse(source.read_kwargs["boundless"])

    def test_radiometry_gate_is_union_across_all_twelve_bands(self) -> None:
        target = np.ones((6, 200, 200), dtype=np.uint16)
        reference = np.ones_like(target)
        target[:, :20] = 0
        reference[:, 20:40] = 0
        self.assertAlmostEqual(audit.radiometric_valid_fraction(target, reference), 0.8)
        self.assertTrue(audit.observability_gate(0.8, 0.8))
        target[0, 40, 0] = 0
        self.assertLess(audit.radiometric_valid_fraction(target, reference), 0.8)

    def test_cloudsen_union_nonclear_and_invalid(self) -> None:
        target_pred = np.zeros((200, 200), dtype=np.uint8)
        reference_pred = np.zeros_like(target_pred)
        target_pred[:10] = 1
        reference_pred[10:20] = 3
        target_stack = np.ones((13, 200, 200), dtype=np.uint16)
        reference_stack = np.ones_like(target_stack)
        reference_stack[:, 20:30] = 0
        nonclear, clear = audit.cloudsen_union_clear(target_pred, reference_pred, target_stack, reference_stack)
        self.assertEqual(np.count_nonzero(nonclear), 30 * 200)
        self.assertAlmostEqual(clear, 0.85)

    def test_landsat_qa_uses_exact_bits_zero_through_five_and_unions_frames(self) -> None:
        target = np.zeros((200, 200), dtype=np.uint16)
        reference = np.zeros_like(target)
        for bit in range(6):
            target[bit, 0] = 1 << bit
        reference[6, 0] = 1 << 5
        target[7, 0] = 1 << 7
        nonclear, clear = audit.landsat_union_clear(target, reference)
        self.assertEqual(np.count_nonzero(nonclear), 7)
        self.assertFalse(nonclear[7, 0])
        self.assertAlmostEqual(clear, 1 - 7 / 40000)

    def test_duplicate_sensor_pairs_count_once_at_observation_gate(self) -> None:
        base = {"site_ID": "s", "obs_ID": "o", "date": "d", "sat_ID": 1, "component_id": "c"}
        rows = [{**base, "target_sensor": audit.S2_SENSOR},
                {**base, "target_sensor": audit.LANDSAT_SENSOR}]
        _, counts = audit.evaluate_final_gates(rows)
        self.assertEqual(counts["source_sensor_pairs"], 2)
        self.assertEqual(counts["distinct_source_observations"], 1)
        self.assertEqual(counts["distinct_sites"], 1)

    def test_dense_or_zero_mask_cannot_be_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(audit.ObservabilityAuditError, "Only six-band"):
                audit.write_crop(Path(directory) / "zero_mask.tif", np.zeros((200, 200), dtype=np.uint8),
                                 crs="EPSG:32643", transform=Affine.identity(), descriptions=("mask",))

    def test_pair_failure_removes_partial_scene_directory(self) -> None:
        row = {"site_ID": "s", "obs_ID": "o", "date": "d", "sat_ID": 1,
               "target_sensor": audit.LANDSAT_SENSOR, "target_item_id": "target",
               "reference_item_id": "reference", "component_id": "c",
               "representative_longitude": 0.0, "representative_latitude": 0.0,
               "target": {"assets": {"B02": "x"}}, "reference": {"assets": {"B02": "y"}}}
        identity = audit.sha256_value({name: row[name] for name in audit.PAIR_JOIN_KEY})[:24]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crop_root = root / "crops"
            partial = crop_root / identity
            partial.mkdir(parents=True)
            (partial / "partial").write_text("x")
            with mock.patch.object(audit, "target_grid", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    audit.process_pair(root, crop_root, row, None)
            self.assertFalse(partial.exists())

    def test_preflight_failure_cleans_stale_outputs_and_redacts_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("ignored_asset_manifest", "ignored_observable_manifest"):
                path = root / audit.EXPECTED_OUTPUTS[name]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("stale")
            crop = root / audit.EXPECTED_OUTPUTS["ignored_crop_root"]
            crop.mkdir(parents=True)
            (crop / "zero_mask.tif").write_text("stale")
            leaked = audit.S2_PREFIX + "tiles/leaked/B02.jp2"
            with mock.patch.object(audit, "_execute_network_audit", side_effect=RuntimeError(leaked)):
                with self.assertRaises(audit.ObservabilityAuditError):
                    audit.execute_network_audit(root=root, session=object())
            self.assertFalse(crop.exists())
            compact = (root / audit.EXPECTED_OUTPUTS["compact_json"]).read_text()
            markdown = (root / audit.EXPECTED_OUTPUTS["compact_markdown"]).read_text()
            self.assertNotIn(leaked, compact + markdown)
            self.assertEqual(json.loads(compact)["decision"], "FAIL")


if __name__ == "__main__":
    unittest.main()
