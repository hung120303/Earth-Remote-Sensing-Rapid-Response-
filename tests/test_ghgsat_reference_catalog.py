from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import tools.audit_ghgsat_reference_catalog as audit


S2_TARGET = "S2A_MSIL1C_20220429T051008_N0500_R092_T37PDL_20230526T191453"
S2_REFERENCE = "S2B_MSIL1C_20220420T051009_N0500_R092_T37PDL_20230526T191454"
LANDSAT_TARGET = "LC09_L1TP_153042_20220429_20230418_02_T1"
LANDSAT_REFERENCE = "LC08_L1TP_153042_20220420_20230418_02_T1"


def protocol() -> dict[str, Any]:
    return copy.deepcopy(audit.load_protocol())


def source_pair(
    *, sensor: str = "sentinel_2_l1c", index: int = 0,
    target_time: str = "2022-04-29T05:10:08Z",
) -> dict[str, object]:
    return {
        "site_ID": f"site-{index}", "obs_ID": f"obs-{index}",
        "date": "2022-04-29 05:10:08", "sat_ID": 1,
        "component_id": f"component-{index}", "target_sensor": sensor,
        "target_item_id": S2_TARGET if sensor == "sentinel_2_l1c" else LANDSAT_TARGET,
        "target_collection": "sentinel-2-l1c" if sensor == "sentinel_2_l1c" else "landsat-c2l1",
        "target_datetime": target_time, "source_target_offset_seconds": 0.0,
        "absolute_source_target_offset_seconds": 0.0, "eo_cloud_cover": 1.0,
        "geometry": polygon(), "bbox": [-1.0, -1.0, 1.0, 1.0],
        "representative_longitude": 0.0, "representative_latitude": 0.0,
    }


def polygon() -> dict[str, object]:
    return {
        "type": "Polygon",
        "coordinates": [[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]]],
    }


def item(
    *, sensor: str = "sentinel_2_l1c", item_id: str | None = None,
    when: str = "2022-04-20T05:10:08Z", cloud: float = 10.0,
    geometry: dict[str, object] | None = None,
) -> dict[str, object]:
    if sensor == "sentinel_2_l1c":
        identifier = item_id or S2_REFERENCE
        collection = "sentinel-2-l1c"
        properties: dict[str, object] = {"datetime": when, "eo:cloud_cover": cloud}
    else:
        identifier = item_id or LANDSAT_REFERENCE
        collection = "landsat-c2l1"
        properties = {
            "datetime": when, "eo:cloud_cover": cloud,
            "landsat:correction": "L1TP", "landsat:collection_category": "T1",
        }
    return {
        "type": "Feature", "id": identifier, "collection": collection,
        "geometry": geometry or polygon(), "bbox": [-1.0, -1.0, 1.0, 1.0],
        "properties": properties,
    }


def collection(features: list[dict[str, object]]) -> bytes:
    return json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")).encode()


class Response:
    def __init__(
        self, payload: bytes, *, status: int = 200, content_type: str = "application/geo+json",
        url: str | None = None, declared: int | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status
        self.url = url
        self.history: list[object] = []
        self.headers = {"Content-Type": content_type}
        self.closed = False
        if declared is not None:
            self.headers["Content-Length"] = str(declared)

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self.payload[offset:offset + chunk_size] for offset in range(0, len(self.payload), chunk_size)]

    def close(self) -> None:
        self.closed = True


class Session:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def post(self, *args: object, **kwargs: object) -> Response:
        self.calls.append((args, kwargs))
        response = self.responses.pop(0)
        if response.url is None:
            response.url = str(args[0])
        return response


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def client(
    tmp_path: Path, value: dict[str, Any], responses: list[Response],
) -> tuple[audit.STACReferenceAuditClient, Session, Clock]:
    session = Session(responses)
    clock = Clock()
    result = audit.STACReferenceAuditClient(
        session=session, protocol=value, request_log=tmp_path / "requests.jsonl",
        response_log=tmp_path / "responses.jsonl", sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    return result, session, clock


def test_exact_frozen_protocol_and_all_79_pairs_validate_offline() -> None:
    plan = audit.validation_plan()
    assert plan["protocol"]["sha256"] == audit.EXPECTED_PROTOCOL_SHA256
    assert plan["source_pairs_validated"] == 79
    assert plan["distinct_target_item_ids_validated"] == 78
    assert plan["network_client_created"] is False
    assert plan["network_executed"] is False
    assert plan["catalog_response_opened"] is False
    assert plan["reference_catalog_accessed"] is False


def test_default_cli_never_creates_network_client(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(audit, "validation_plan", lambda: {
        "mode": "validation_only", "network_client_created": False,
        "network_executed": False, "catalog_response_opened": False,
    })
    monkeypatch.setattr(
        audit.target.requests, "Session",
        lambda: (_ for _ in ()).throw(AssertionError("network client created")),
    )
    monkeypatch.setattr(
        audit, "execute_network_audit",
        lambda: (_ for _ in ()).throw(AssertionError("network executed")),
    )
    assert audit.main([]) == 0
    assert '"network_client_created": false' in capsys.readouterr().out
    assert audit.build_parser().parse_args([]).execute_network is False


def test_protocol_and_frozen_input_hash_mismatches_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = tmp_path / "protocol.json"
    bad.write_bytes(audit.EXPECTED_PROTOCOL.read_bytes() + b" ")
    monkeypatch.setattr(audit, "EXPECTED_PROTOCOL", bad)
    with pytest.raises(audit.ReferenceCatalogAuditError, match="SHA-256 mismatch"):
        audit.load_protocol(bad)
    monkeypatch.setattr(audit, "EXPECTED_PROTOCOL", audit.ROOT / audit.PROTOCOL_RELATIVE_PATH)

    value = protocol()
    payload = b"wrong"
    (tmp_path / "input.json").write_bytes(payload)
    for spec in value["frozen_inputs"].values():
        spec.update(path="input.json", bytes=len(payload), sha256="0" * 64)
    with pytest.raises(audit.ReferenceCatalogAuditError, match="hash mismatch"):
        audit.validate_frozen_inputs(value, root=tmp_path)


def test_primary_and_seasonal_requests_are_exact_metadata_only_and_deterministic() -> None:
    value = protocol()
    row = source_pair(target_time="2022-04-29T05:10:08.125000Z")
    primary = audit.build_request(row, "primary", value)
    seasonal = audit.build_request(copy.deepcopy(row), "seasonal", value)
    assert primary == audit.build_request(copy.deepcopy(row), "primary", value)
    assert primary["body"]["datetime"] == "2022-03-29T05:10:08.125000Z/2022-04-29T04:10:08.125000Z"
    assert seasonal["body"]["datetime"] == "2021-03-15T05:10:08.125000Z/2021-05-30T05:10:08.125000Z"
    for request in (primary, seasonal):
        body = request["body"]
        assert body["intersects"] == {"type": "Point", "coordinates": [0.0, 0.0]}
        assert body["limit"] == 100
        assert body["fields"]["exclude"] == ["assets", "links"]
        assert "href" not in request["canonical_request"]
        assert request["canonical_request_sha256"] == hashlib.sha256(
            str(request["canonical_request"]).encode()
        ).hexdigest()
        assert request["target_identity"]["target_item_id"] == S2_TARGET


def test_exact_mgrs_and_wrs_granule_parsing() -> None:
    assert audit._granule(S2_TARGET, "sentinel_2_l1c") == "37PDL"
    assert audit._granule(S2_REFERENCE, "sentinel_2_l1c") == "37PDL"
    assert audit._granule(LANDSAT_TARGET, "landsat_8_9_level_1") == "153042"
    assert audit._granule(LANDSAT_REFERENCE, "landsat_8_9_level_1") == "153042"
    assert audit._granule("S2A_MSIL2A_20220429T051008_N0500_R092_T37PDL_20230526T191453", "sentinel_2_l1c") is None
    assert audit._granule("LC08_L1GT_153042_20220420_20230418_02_T1", "landsat_8_9_level_1") is None


def test_real_frozen_target_pairs_all_supply_same_granule_keys() -> None:
    rows = audit.load_source_pairs(protocol())
    assert len(rows) == 79
    assert all(
        audit._granule(str(row["target_item_id"]), str(row["target_sensor"])) is not None
        for row in rows
    )


def test_unrelated_missions_products_collections_and_granules_are_non_candidates() -> None:
    value = protocol()
    row = source_pair()
    unrelated = [
        item(item_id="S3A_unrelated"),
        item(item_id="S2A_MSIL2A_20220420T051009_N0500_R092_T37PDL_20230526T191454"),
        item(item_id="S2A_MSIL1C_20220420T051009_N0500_R092_T38PDL_20230526T191454"),
    ]
    wrong_collection = item()
    wrong_collection["collection"] = "other"
    unrelated.append(wrong_collection)
    parsed = audit.parse_feature_collection(
        {"type": "FeatureCollection", "features": unrelated}, row, "primary", value,
        excluded_target_ids=set(),
    )
    assert parsed == []


def test_all_target_ids_for_sensor_are_excluded_globally() -> None:
    value = protocol()
    row = source_pair()
    other_row_target = "S2B_MSIL1C_20220420T051009_N0500_R092_T37PDL_20230526T191454"
    assert audit.validate_candidate(
        item(item_id=other_row_target), row, "primary", value,
        excluded_target_ids={S2_TARGET, other_row_target},
    ) is None


def test_provider_extras_are_received_but_never_persisted(tmp_path: Path) -> None:
    value = protocol()
    extra = item()
    extra["assets"] = {"band": {"href": "https://example.invalid/asset.tif"}}
    extra["links"] = [{"href": "https://example.invalid/item"}]
    extra["properties"]["provider:url"] = "https://example.invalid/other"
    extra["geometry"]["provider:url"] = "https://example.invalid/geometry"
    payload = collection([extra])
    c, _, _ = client(tmp_path, value, [Response(payload)])
    row = source_pair()
    candidates, _ = c.execute(
        audit.build_request(row, "primary", value), row, excluded_target_ids={S2_TARGET}
    )
    assert len(candidates) == 1
    receipt_text = (tmp_path / "responses.jsonl").read_text()
    assert "example.invalid" not in receipt_text
    assert "assets" not in receipt_text and "links" not in receipt_text
    receipt = json.loads(receipt_text)
    assert receipt["response_bytes"] == len(payload)
    assert receipt["response_sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    ("when", "kind", "valid"),
    [
        ("2022-04-29T04:10:08Z", "primary", True),
        ("2022-03-29T05:10:08Z", "primary", True),
        ("2022-04-29T04:10:08.000001Z", "primary", False),
        ("2022-03-29T05:10:07.999999Z", "primary", False),
        ("2021-05-30T05:10:08Z", "seasonal", True),
        ("2021-03-15T05:10:08Z", "seasonal", True),
        ("2021-05-30T05:10:08.000001Z", "seasonal", False),
        ("2021-03-15T05:10:07.999999Z", "seasonal", False),
    ],
)
def test_closed_time_windows_are_strict(when: str, kind: str, valid: bool) -> None:
    candidate = audit.validate_candidate(
        item(when=when), source_pair(), kind, protocol(), excluded_target_ids={S2_TARGET}
    )
    assert (candidate is not None) is valid


def test_cloud_geometry_platform_tier_and_product_filters_are_strict() -> None:
    value = protocol()
    s2row = source_pair()
    assert audit.validate_candidate(
        item(cloud=20.0), s2row, "primary", value, excluded_target_ids={S2_TARGET}
    ) is not None
    assert audit.validate_candidate(
        item(cloud=20.0001), s2row, "primary", value, excluded_target_ids={S2_TARGET}
    ) is None
    outside = item(geometry={"type": "Point", "coordinates": [1, 1]})
    assert audit.validate_candidate(
        outside, s2row, "primary", value, excluded_target_ids={S2_TARGET}
    ) is None
    with pytest.raises(audit.ReferenceCatalogAuditError, match="cloud"):
        audit.validate_candidate(
            item(cloud=float("nan")), s2row, "primary", value,
            excluded_target_ids={S2_TARGET},
        )

    lrow = source_pair(sensor="landsat_8_9_level_1")
    wrong_tier = item(sensor="landsat_8_9_level_1")
    wrong_tier["properties"]["landsat:collection_category"] = "T2"
    assert audit.validate_candidate(
        wrong_tier, lrow, "primary", value, excluded_target_ids={LANDSAT_TARGET}
    ) is None
    wrong_level = item(sensor="landsat_8_9_level_1")
    wrong_level["properties"]["landsat:correction"] = "L1GT"
    assert audit.validate_candidate(
        wrong_level, lrow, "primary", value, excluded_target_ids={LANDSAT_TARGET}
    ) is None


def test_fractional_reference_timestamp_is_preserved() -> None:
    candidate = audit.validate_candidate(
        item(when="2022-04-20T05:10:08.125Z"), source_pair(), "primary", protocol(),
        excluded_target_ids={S2_TARGET},
    )
    assert candidate is not None
    assert candidate["reference_datetime"] == "2022-04-20T05:10:08.125000Z"
    assert candidate["target_reference_gap_seconds"] == pytest.approx(777599.875)


def test_primary_and_seasonal_selection_orders_are_deterministic() -> None:
    base = audit.validate_candidate(
        item(), source_pair(), "primary", protocol(), excluded_target_ids={S2_TARGET}
    )
    assert base is not None
    primary = []
    for identifier, gap, cloud in (("z", 10, 1), ("b", 5, 9), ("a", 5, 9)):
        primary.append({
            **base, "reference_item_id": identifier,
            "target_reference_gap_seconds": gap, "eo_cloud_cover": cloud,
        })
    assert audit.select_candidate(list(reversed(primary)), "primary")["reference_item_id"] == "a"
    seasonal = [
        {**base, "reference_item_id": identifier, "seasonal_distance_from_365_days": distance, "eo_cloud_cover": cloud}
        for identifier, distance, cloud in (("z", 2, 1), ("b", 1, 9), ("a", 1, 9))
    ]
    assert audit.select_candidate(list(reversed(seasonal)), "seasonal")["reference_item_id"] == "a"


def test_response_ceiling_retry_pacing_and_streaming_caps(tmp_path: Path) -> None:
    value = protocol()
    row = source_pair()
    with pytest.raises(audit.ReferenceCatalogAuditError, match="100-item"):
        audit.parse_feature_collection(
            {"type": "FeatureCollection", "features": [item(item_id=f"S3_{index}") for index in range(100)]},
            row, "primary", value, excluded_target_ids=set(),
        )

    payload = collection([])
    c, session, clock = client(tmp_path, value, [
        Response(b"<html>busy</html>", status=503, content_type="text/html"),
        Response(payload), Response(payload),
    ])
    request = audit.build_request(row, "primary", value)
    c.execute(request, row, excluded_target_ids={S2_TARGET})
    c.execute(request, row, excluded_target_ids={S2_TARGET})
    assert len(session.calls) == 3
    assert clock.sleeps == [1, pytest.approx(0.25)]

    limited = protocol()
    limited["query_contract"]["maximum_response_bytes_each"] = 8
    limited["query_contract"]["maximum_response_bytes_total"] = 12
    c, _, _ = client(tmp_path / "caps", limited, [Response(b"012345678")])
    with pytest.raises(audit.ReferenceCatalogAuditError, match="Streamed"):
        c.execute(
            audit.build_request(row, "primary", limited), row,
            excluded_target_ids={S2_TARGET},
        )
    assert c.total_network_bytes == 9


def test_resume_binds_request_endpoint_query_source_target_and_candidates(tmp_path: Path) -> None:
    value = protocol()
    row = source_pair()
    request = audit.build_request(row, "primary", value)
    candidate = audit.validate_candidate(
        item(), row, "primary", value, excluded_target_ids={S2_TARGET}
    )
    assert candidate is not None
    raw_body = collection([item()])
    receipt: dict[str, object] = {
        "schema_version": 1, "attempt": 1, "status": 200, "accepted": True,
        "body_read_complete": True,
        "endpoint": request["endpoint"], "sensor": request["sensor"],
        "query_kind": request["query_kind"], "source_identity": request["source_identity"],
        "target_identity": request["target_identity"],
        "canonical_request_sha256": request["canonical_request_sha256"],
        "response_bytes": len(raw_body), "response_sha256": hashlib.sha256(raw_body).hexdigest(),
        "candidate_count": 1, "parsed_candidates": [candidate],
        "parsed_candidates_sha256": audit.sha256_bytes(audit.canonical_json_bytes([candidate])),
    }
    receipt["checkpoint_sha256"] = audit.response_checkpoint_sha256(receipt)
    response_log = tmp_path / "responses.jsonl"
    response_log.write_text(json.dumps(receipt) + "\n")
    request_log = tmp_path / "requests.jsonl"
    request_log.write_text(json.dumps({
        "schema_version": 1, "attempt": 1, "endpoint": request["endpoint"],
        "sensor": request["sensor"], "query_kind": request["query_kind"],
        "source_identity": request["source_identity"], "target_identity": request["target_identity"],
        "canonical_request": request["canonical_request"],
        "canonical_request_sha256": request["canonical_request_sha256"],
    }) + "\n")
    loaded = audit.load_resume_receipts(response_log, request_log=request_log)
    assert audit.resume_key(request) in loaded
    assert audit._resume_candidates(
        loaded[audit.resume_key(request)], request, row, value,
        excluded_target_ids={S2_TARGET},
    ) == [candidate]

    tampered = dict(receipt)
    tampered["target_identity"] = {**request["target_identity"], "target_item_id": "other"}
    tampered["checkpoint_sha256"] = audit.response_checkpoint_sha256(tampered)
    with pytest.raises(audit.ReferenceCatalogAuditError, match="identity"):
        audit._resume_candidates(
            tampered, request, row, value, excluded_target_ids={S2_TARGET}
        )

    for field, replacement, message in (
        ("body_read_complete", False, "accepted HTTP 200"),
        ("response_sha256", "not-a-hash", "byte/hash"),
        ("response_bytes", value["query_contract"]["maximum_response_bytes_each"] + 1, "byte/hash"),
    ):
        malformed = dict(receipt)
        malformed[field] = replacement
        malformed["checkpoint_sha256"] = audit.response_checkpoint_sha256(malformed)
        with pytest.raises(audit.ReferenceCatalogAuditError, match=message):
            audit._resume_candidates(
                malformed, request, row, value, excluded_target_ids={S2_TARGET}
            )


def test_historical_response_receipts_enforce_hash_shape_and_total_cap(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    rows = [
        {"schema_version": 1, "response_bytes": 7, "response_sha256": "a" * 64,
         "body_read_complete": True, "accepted": False},
        {"schema_version": 1, "response_bytes": 6, "response_sha256": "b" * 64,
         "body_read_complete": False, "accepted": False},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert audit.historical_response_bytes(path, maximum=13) == 13
    with pytest.raises(audit.ReferenceCatalogAuditError, match="total byte cap"):
        audit.historical_response_bytes(path, maximum=12)
    rows[0]["response_sha256"] = "not-a-hash"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(audit.ReferenceCatalogAuditError, match="malformed"):
        audit.historical_response_bytes(path, maximum=13)


def _orchestration_candidate(row: dict[str, object], kind: str) -> dict[str, object]:
    gap = 9 * 86400.0 if kind == "primary" else 365 * 86400.0
    return {
        **audit._candidate_identity_fields(row), "selection_window": kind,
        "reference_item_id": f"ref-{kind}-{row['obs_ID']}",
        "reference_collection": row["target_collection"],
        "reference_datetime": "2022-04-20T05:10:08Z" if kind == "primary" else "2021-04-29T05:10:08Z",
        "target_reference_gap_seconds": gap, "target_reference_gap_days": gap / 86400,
        "seasonal_distance_from_365_days": abs(gap / 86400 - 365),
        "eo_cloud_cover": 1.0, "granule_id": "37PDL",
        "geometry": polygon(), "bbox": [-1.0, -1.0, 1.0, 1.0],
    }


def test_execution_runs_79_primary_and_seasonal_only_for_empty_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = protocol()
    value["outputs"] = {
        **value["outputs"], "ignored_root": "ignored",
        "ignored_requests": "ignored/requests.jsonl", "ignored_responses": "ignored/responses.jsonl",
        "ignored_candidates": "ignored/candidates.jsonl", "ignored_pairs": "ignored/pairs.jsonl",
        "compact_json": "reports/report.json", "compact_markdown": "reports/report.md",
    }
    rows = [source_pair(index=index) for index in range(79)]
    calls: list[tuple[str, str]] = []

    def fake_execute(
        self: audit.STACReferenceAuditClient, request: dict[str, object], row: dict[str, object],
        *, excluded_target_ids: set[str],
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        kind = str(request["query_kind"])
        calls.append((str(row["obs_ID"]), kind))
        if kind == "primary" and row["obs_ID"] in {"obs-0", "obs-1"}:
            return [], {}
        return [_orchestration_candidate(row, kind)], {}

    monkeypatch.setattr(audit, "load_protocol", lambda: value)
    monkeypatch.setattr(audit, "validate_frozen_inputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(audit, "load_source_pairs", lambda *args, **kwargs: rows)
    monkeypatch.setattr(audit.STACReferenceAuditClient, "execute", fake_execute)
    report = audit.execute_network_audit(root=tmp_path, session=Session([]))
    assert report["queries"]["primary_logical_queries"] == 79
    assert report["queries"]["seasonal_logical_queries"] == 2
    assert len([call for call in calls if call[1] == "primary"]) == 79
    assert [call for call in calls if call[1] == "seasonal"] == [
        ("obs-0", "seasonal"), ("obs-1", "seasonal")
    ]
    assert report["distinct_counts"]["selected_target_reference_pairs"] == 79


def test_gate_counts_do_not_let_dual_sensor_rows_manufacture_observations() -> None:
    value = protocol()
    selected = []
    for index in range(14):
        s2 = source_pair(index=index)
        landsat = source_pair(index=index, sensor="landsat_8_9_level_1")
        selected.extend([_orchestration_candidate(s2, "primary"), _orchestration_candidate(landsat, "primary")])
    gates, counts = audit.evaluate_gates(selected, value, all_valid=True)
    assert gates["minimum_selected_target_reference_pairs"]["pass"] is True
    assert gates["minimum_distinct_source_observations_with_reference"]["pass"] is False
    assert counts["selected_target_reference_pairs"] == 28
    assert counts["distinct_source_observations"] == 14


def test_fail_closed_removes_only_derived_outputs_and_preserves_append_logs(tmp_path: Path) -> None:
    value = protocol()
    value["outputs"] = {
        **value["outputs"], "ignored_candidates": "ignored/candidates.jsonl",
        "ignored_pairs": "ignored/pairs.jsonl", "compact_json": "reports/report.json",
        "compact_markdown": "reports/report.md",
    }
    for name in ("ignored_candidates", "ignored_pairs"):
        path = tmp_path / value["outputs"][name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n")
    requests = tmp_path / value["outputs"]["ignored_requests"]
    responses = tmp_path / value["outputs"]["ignored_responses"]
    requests.parent.mkdir(parents=True, exist_ok=True)
    requests.write_text("append-only\n")
    responses.write_text("append-only\n")
    audit._remove_stale_outputs(value, root=tmp_path)
    assert not (tmp_path / value["outputs"]["ignored_candidates"]).exists()
    assert not (tmp_path / value["outputs"]["ignored_pairs"]).exists()
    assert requests.read_text() == "append-only\n"
    assert responses.read_text() == "append-only\n"

    report = audit.build_failure_report(value, audit.ReferenceCatalogAuditError("fixture"))
    audit._write_reports(value, report, root=tmp_path)
    stored = json.loads((tmp_path / "reports/report.json").read_text())
    assert stored["decision"] == "FAIL"
    assert all(not gate["pass"] for gate in stored["gates"].values())
    assert stored["access_boundary_proof"]["item_detail_queried"] is False
    assert stored["access_boundary_proof"]["raster_bytes_accessed"] is False
