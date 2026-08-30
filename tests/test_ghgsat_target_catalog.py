from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import tools.audit_ghgsat_target_catalog as audit


def protocol() -> dict[str, Any]:
    return copy.deepcopy(audit.load_protocol())


def source_row(
    *, site: str = "site-a", obs: str = "obs-a",
    date: str = "2021-01-02 03:04:05", component: str = "component-a",
    latitude: float = 0.0, longitude: float = 0.0,
) -> dict[str, object]:
    return {
        "site_ID": site, "obs_ID": obs, "date": date, "sat_ID": 1,
        "year": 2021, "observation_state": "null", "plume_row_count": 0,
        "source_data_rows": [1], "representative_latitude": latitude,
        "representative_longitude": longitude, "representative_positive_data_row": 2,
        "positive_coordinate_span_km": 0.0, "nearest_official_mars_test_km": 100.0,
        "nearest_official_mars_test_location": "safe-location",
        "nearest_prior_negative_km": 100.0, "nearest_prior_negative_id": "safe-prior",
        "excluded_by_official_mars_test_radius": False,
        "excluded_by_prior_negative_radius": False,
        "passes_protected_distance_filter": True,
        "eligible_for_target_catalog": False, "component_id": component,
    }


def item(
    *, sensor: str = "sentinel_2_l1c", item_id: str | None = None,
    when: str = "2021-01-02T03:04:05Z", cloud: float = 10.0,
    geometry: dict[str, object] | None = None,
) -> dict[str, object]:
    if sensor == "sentinel_2_l1c":
        identifier = item_id or "S2A_fixture"
        collection = "sentinel-2-l1c"
        properties: dict[str, object] = {"datetime": when, "eo:cloud_cover": cloud}
    else:
        identifier = item_id or "LC08_fixture"
        collection = "landsat-c2l1"
        properties = {
            "datetime": when, "eo:cloud_cover": cloud,
            "landsat:correction": "L1TP", "landsat:collection_category": "T1",
        }
    return {
        "type": "Feature", "id": identifier, "collection": collection,
        "geometry": geometry or {
            "type": "Polygon",
            "coordinates": [[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]]],
        },
        "bbox": [-1.0, -1.0, 1.0, 1.0], "properties": properties,
    }


def feature_collection(features: list[dict[str, object]]) -> bytes:
    return json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")).encode()


class Response:
    def __init__(
        self, payload: bytes, *, status: int = 200, url: str | None = None,
        declared: int | None = None, content_type: str = "application/geo+json",
    ) -> None:
        self.payload = payload
        self.status_code = status
        self.url = url
        self.history: list[object] = []
        self.closed = False
        self.headers = {"Content-Type": content_type}
        if declared is not None:
            self.headers["Content-Length"] = str(declared)

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self.payload[index:index + chunk_size] for index in range(0, len(self.payload), chunk_size)]

    def close(self) -> None:
        self.closed = True


class Session:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.closed = False

    def post(self, *args: object, **kwargs: object) -> Response:
        self.calls.append((args, kwargs))
        response = self.responses.pop(0)
        if response.url is None:
            response.url = str(args[0])
        return response

    def close(self) -> None:
        self.closed = True


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def client(tmp_path: Path, value: dict[str, Any], responses: list[Response], *, clock: Clock | None = None) -> tuple[audit.STACAuditClient, Session]:
    session = Session(responses)
    fake_clock = clock or Clock()
    result = audit.STACAuditClient(
        session=session, protocol=value,
        request_log=tmp_path / "requests.jsonl", response_log=tmp_path / "responses.jsonl",
        sleep=fake_clock.sleep, monotonic=fake_clock.monotonic,
    )
    return result, session


def selected_row(
    index: int, *, sensor: str = "sentinel_2_l1c", site: str | None = None,
    component: str | None = None, target: str | None = None,
) -> dict[str, object]:
    row = source_row(
        site=site or f"site-{index}", obs=f"obs-{index}",
        component=component or f"component-{index}",
    )
    return {
        **audit.source_identity(row), "component_id": row["component_id"],
        "target_sensor": sensor, "target_item_id": target or f"item-{index}",
        "target_collection": "sentinel-2-l1c", "target_datetime": "2021-01-02T03:04:05Z",
        "source_target_offset_seconds": 0.0, "absolute_source_target_offset_seconds": 0.0,
        "eo_cloud_cover": 10.0, "geometry": item()["geometry"], "bbox": [-1, -1, 1, 1],
    }


def test_exact_frozen_protocol_inputs_and_all_176_rows_validate_offline() -> None:
    plan = audit.validation_plan()
    assert plan["protocol"]["sha256"] == audit.EXPECTED_PROTOCOL_SHA256
    assert plan["source_rows_validated"] == 176
    assert plan["eligible_for_target_catalog_expected_false_rows"] == 176
    assert plan["network_client_created"] is False
    assert plan["network_executed"] is False
    assert plan["target_response_opened"] is False


def test_safe_default_cli_never_creates_session_or_executes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(audit, "validation_plan", lambda: {
        "mode": "validation_only", "network_client_created": False,
        "network_executed": False, "target_response_opened": False,
    })
    monkeypatch.setattr(audit.requests, "Session", lambda: (_ for _ in ()).throw(AssertionError("network client")))
    monkeypatch.setattr(audit, "execute_network_audit", lambda: (_ for _ in ()).throw(AssertionError("network")))
    assert audit.main([]) == 0
    output = capsys.readouterr().out
    assert '"network_client_created": false' in output
    assert '"network_executed": false' in output
    assert audit.build_parser().parse_args([]).execute_network is False


def test_protocol_and_frozen_input_hash_mismatches_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad_protocol = tmp_path / "protocol.json"
    bad_protocol.write_bytes(audit.EXPECTED_PROTOCOL.read_bytes() + b" ")
    monkeypatch.setattr(audit, "EXPECTED_PROTOCOL", bad_protocol)
    with pytest.raises(audit.TargetCatalogAuditError, match="SHA-256 mismatch"):
        audit.load_protocol(bad_protocol)
    monkeypatch.setattr(
        audit, "EXPECTED_PROTOCOL",
        audit.ROOT / audit.PROTOCOL_RELATIVE_PATH,
    )

    value = protocol()
    path = tmp_path / "input.json"
    path.write_bytes(b"wrong")
    value["frozen_source_inputs"] = {
        "metadata_protocol": {"path": "input.json", "bytes": 5, "sha256": "0" * 64},
        "metadata_report": {"path": "input.json", "bytes": 5, "sha256": hashlib.sha256(b"wrong").hexdigest(), "required_decision": "PASS"},
        "eligible_observations": {"path": "input.json", "bytes": 5, "sha256": hashlib.sha256(b"wrong").hexdigest(), "rows": 1},
    }
    with pytest.raises(audit.TargetCatalogAuditError, match="hash mismatch"):
        audit.validate_frozen_inputs(value, root=tmp_path)


def test_source_rows_require_exact_fields_values_false_eligibility_and_count(tmp_path: Path) -> None:
    value = protocol()
    path = tmp_path / "rows.jsonl"
    value["frozen_source_inputs"]["eligible_observations"].update(path="rows.jsonl", rows=1)
    path.write_text(json.dumps(source_row()) + "\n", encoding="utf-8")
    assert len(audit.load_source_rows(value, root=tmp_path)) == 1
    bad = source_row()
    bad["eligible_for_target_catalog"] = True
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(audit.TargetCatalogAuditError, match="must remain false"):
        audit.load_source_rows(value, root=tmp_path)
    bad = source_row()
    bad["unexpected"] = 1
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(audit.TargetCatalogAuditError, match="field set"):
        audit.load_source_rows(value, root=tmp_path)


def test_request_canonicalization_is_exact_deterministic_and_metadata_only() -> None:
    value = protocol()
    row = source_row(longitude=-0.0)
    first = audit.build_request(row, "sentinel_2_l1c", value)
    second = audit.build_request(copy.deepcopy(row), "sentinel_2_l1c", value)
    assert first["canonical_request"] == second["canonical_request"]
    assert first["canonical_request_sha256"] == hashlib.sha256(str(first["canonical_request"]).encode()).hexdigest()
    body = first["body"]
    assert body["collections"] == ["sentinel-2-l1c"]
    assert body["datetime"] == "2021-01-02T02:04:05Z/2021-01-02T04:04:05Z"
    assert body["limit"] == 100
    assert body["fields"]["include"] == [
        "type", *value["catalogs"]["sentinel_2_l1c"]["required_item_fields"],
    ]
    assert body["fields"]["exclude"] == ["assets", "links"]
    encoded = str(first["canonical_request"])
    assert '"assets"' not in encoded.replace('"exclude":["assets","links"]', "")
    assert "href" not in encoded


def test_resume_requires_request_endpoint_body_hash_and_parsed_identity(tmp_path: Path) -> None:
    value = protocol()
    row = source_row()
    request = audit.build_request(row, "sentinel_2_l1c", value)
    payload = feature_collection([item()])
    parsed_candidates = [audit.validate_candidate(item(), row, "sentinel_2_l1c", value)]
    receipt = {
        "schema_version": 1, "status": 200, "accepted": True,
        "canonical_request_sha256": request["canonical_request_sha256"],
        "endpoint": request["endpoint"], "sensor": request["sensor"],
        "source_identity": request["source_identity"],
        "parsed_observation_identity": request["source_identity"],
        "response_bytes": len(payload), "response_sha256": hashlib.sha256(payload).hexdigest(),
        "parsed_candidates": parsed_candidates,
        "parsed_candidates_sha256": hashlib.sha256(audit.canonical_json_bytes(parsed_candidates)).hexdigest(),
        "candidate_count": 1,
    }
    receipt["checkpoint_sha256"] = audit.response_checkpoint_sha256(receipt)
    path = tmp_path / "responses.jsonl"
    path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    key = audit.resume_key(
        request["canonical_request_sha256"], request["endpoint"],
        request["sensor"], request["source_identity"],
    )
    loaded = audit.load_resume_receipts(path)[key]
    assert len(audit._resume_candidates(loaded, request, row, value)) == 1

    cases = [
        ("endpoint", "https://example.invalid/search", "endpoint"),
        ("parsed_observation_identity", {**audit.source_identity(row), "obs_ID": "other"}, "identity"),
        ("response_sha256", "0" * 64, "byte/hash"),
    ]
    for key, replacement, message in cases:
        tampered = dict(receipt)
        tampered[key] = replacement
        if key != "response_sha256":
            tampered["checkpoint_sha256"] = audit.response_checkpoint_sha256(tampered)
        with pytest.raises(audit.TargetCatalogAuditError, match=message):
            audit._resume_candidates(tampered, request, row, value)

    request_log = tmp_path / "requests.jsonl"
    request_log.write_text(json.dumps({
        "schema_version": 1, "canonical_request_sha256": request["canonical_request_sha256"],
        "canonical_request": request["canonical_request"],
        "endpoint": request["endpoint"], "sensor": request["sensor"],
        "source_identity": request["source_identity"],
    }) + "\n", encoding="utf-8")
    assert audit.load_resume_receipts(path, request_log=request_log)
    request_log.write_text("", encoding="utf-8")
    with pytest.raises(audit.TargetCatalogAuditError, match="matching request"):
        audit.load_resume_receipts(path, request_log=request_log)


def test_duplicate_accepted_resume_receipt_is_rejected(tmp_path: Path) -> None:
    body = feature_collection([])
    receipt = {
        "schema_version": 1, "status": 200, "accepted": True,
        "canonical_request_sha256": "a" * 64,
        "endpoint": "https://example.test/search", "sensor": "sentinel_2_l1c",
        "source_identity": {"site_ID": "s", "obs_ID": "o", "date": "2021-01-01", "sat_ID": 1},
        "parsed_observation_identity": {"site_ID": "s", "obs_ID": "o", "date": "2021-01-01", "sat_ID": 1},
        "response_bytes": len(body), "response_sha256": hashlib.sha256(body).hexdigest(),
        "parsed_candidates": [],
        "parsed_candidates_sha256": hashlib.sha256(audit.canonical_json_bytes([])).hexdigest(),
        "candidate_count": 0,
    }
    receipt["checkpoint_sha256"] = audit.response_checkpoint_sha256(receipt)
    path = tmp_path / "responses.jsonl"
    path.write_text(json.dumps(receipt) + "\n" + json.dumps(receipt) + "\n", encoding="utf-8")
    with pytest.raises(audit.TargetCatalogAuditError, match="duplicate"):
        audit.load_resume_receipts(path)


def test_identical_canonical_request_hashes_remain_distinct_by_source_identity(tmp_path: Path) -> None:
    value = protocol()
    first_row = source_row(obs="first")
    second_row = source_row(obs="second")
    first_request = audit.build_request(first_row, "sentinel_2_l1c", value)
    second_request = audit.build_request(second_row, "sentinel_2_l1c", value)
    assert first_request["canonical_request_sha256"] == second_request["canonical_request_sha256"]
    receipts = []
    for request in (first_request, second_request):
        receipt = {
            "schema_version": 1, "status": 200, "accepted": True,
            "canonical_request_sha256": request["canonical_request_sha256"],
            "endpoint": request["endpoint"], "sensor": request["sensor"],
            "source_identity": request["source_identity"],
            "parsed_observation_identity": request["source_identity"],
            "response_bytes": 42, "response_sha256": "a" * 64,
            "parsed_candidates": [],
            "parsed_candidates_sha256": hashlib.sha256(audit.canonical_json_bytes([])).hexdigest(),
            "candidate_count": 0,
        }
        receipt["checkpoint_sha256"] = audit.response_checkpoint_sha256(receipt)
        receipts.append(receipt)
    path = tmp_path / "responses.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in receipts), encoding="utf-8")
    assert len(audit.load_resume_receipts(path)) == 2


def test_global_minimum_request_spacing_applies_across_logical_queries(tmp_path: Path) -> None:
    value = protocol()
    clock = Clock()
    payload = feature_collection([])
    c, _ = client(tmp_path, value, [Response(payload), Response(payload)], clock=clock)
    row_a = source_row(obs="a")
    row_b = source_row(obs="b")
    c.execute(audit.build_request(row_a, "sentinel_2_l1c", value), row_a)
    c.execute(audit.build_request(row_b, "sentinel_2_l1c", value), row_b)
    assert c.http_attempts == 2
    assert clock.sleeps == [pytest.approx(0.25)]


def test_retry_policy_uses_only_frozen_statuses_and_at_most_five_attempts(tmp_path: Path) -> None:
    value = protocol()
    payload = feature_collection([])
    c, session = client(tmp_path, value, [Response(b"", status=503), Response(payload)])
    row = source_row()
    candidates, receipt = c.execute(audit.build_request(row, "sentinel_2_l1c", value), row)
    assert candidates == [] and receipt["attempt"] == 2
    assert len(session.calls) == 2

    c, session = client(tmp_path / "nonretry", value, [Response(b"", status=501), Response(payload)])
    with pytest.raises(audit.TargetCatalogAuditError, match="501"):
        c.execute(audit.build_request(row, "sentinel_2_l1c", value), row)
    assert len(session.calls) == 1

    c, session = client(tmp_path / "five", value, [Response(b"", status=503) for _ in range(5)])
    with pytest.raises(audit.TargetCatalogAuditError, match="503"):
        c.execute(audit.build_request(row, "sentinel_2_l1c", value), row)
    assert len(session.calls) == 5


def test_retryable_status_does_not_require_json_error_body(tmp_path: Path) -> None:
    value = protocol()
    payload = feature_collection([])
    c, session = client(tmp_path, value, [
        Response(b"<html>busy</html>", status=503, content_type="text/html"),
        Response(payload),
    ])
    row = source_row()
    candidates, receipt = c.execute(audit.build_request(row, "sentinel_2_l1c", value), row)
    assert candidates == [] and receipt["attempt"] == 2
    assert len(session.calls) == 2


def test_response_receipt_retains_hash_and_parsed_metadata_but_not_raw_urls(tmp_path: Path) -> None:
    value = protocol()
    raw = json.dumps({
        "type": "FeatureCollection", "features": [item()],
        "context": {"forbidden_raw_url": "https://example.invalid/asset.tif"},
    }).encode()
    c, _ = client(tmp_path, value, [Response(raw)])
    row = source_row()
    c.execute(audit.build_request(row, "sentinel_2_l1c", value), row)
    receipt_text = (tmp_path / "responses.jsonl").read_text()
    assert "example.invalid" not in receipt_text
    receipt = json.loads(receipt_text)
    assert receipt["response_bytes"] == len(raw)
    assert receipt["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["parsed_candidates"][0]["target_item_id"] == "S2A_fixture"


def test_per_response_and_total_byte_caps_are_nonrefundable(tmp_path: Path) -> None:
    value = protocol()
    value["query_contract"]["maximum_response_bytes_each"] = 8
    value["query_contract"]["maximum_response_bytes_total"] = 12
    c, _ = client(tmp_path, value, [Response(b"012345678")])
    with pytest.raises(audit.TargetCatalogAuditError, match="Streamed"):
        c.execute(audit.build_request(source_row(), "sentinel_2_l1c", value), source_row())
    assert c.total_network_bytes == 9

    value["query_contract"]["maximum_response_bytes_each"] = 20
    c, _ = client(tmp_path / "total", value, [Response(b"1234567"), Response(b"123456")])
    with pytest.raises(audit.TargetCatalogAuditError, match="frozen byte cap"):
        c._read_body(Response(b"1234567"))
        c._read_body(Response(b"123456"))
    assert c.total_network_bytes == 13


def test_historical_response_bytes_enforce_total_cap_across_resumes(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    rows = [
        {"response_bytes": 7, "response_sha256": "a" * 64},
        {"response_bytes": 6, "response_sha256": "b" * 64},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(audit.TargetCatalogAuditError, match="Historical"):
        audit.historical_response_bytes(path, maximum=12)
    assert audit.historical_response_bytes(path, maximum=13) == 13


@pytest.mark.parametrize("offset", [-3600, 3600])
def test_closed_time_boundaries_are_inclusive(offset: int) -> None:
    value = protocol()
    row = source_row()
    when = "2021-01-02T02:04:05Z" if offset < 0 else "2021-01-02T04:04:05Z"
    candidate = audit.validate_candidate(item(when=when), row, "sentinel_2_l1c", value)
    assert candidate["source_target_offset_seconds"] == offset


def test_time_outside_boundary_and_non_utc_are_rejected() -> None:
    value = protocol()
    row = source_row()
    with pytest.raises(audit.TargetCatalogAuditError, match="outside"):
        audit.validate_candidate(item(when="2021-01-02T04:04:06Z"), row, "sentinel_2_l1c", value)
    with pytest.raises(audit.TargetCatalogAuditError, match="explicitly be UTC"):
        audit.validate_candidate(item(when="2021-01-02T03:04:05+01:00"), row, "sentinel_2_l1c", value)


def test_point_multipoint_polygon_and_multipolygon_geometry() -> None:
    assert audit.geometry_covers_point({"type": "Point", "coordinates": [1, 2]}, 1, 2)
    assert audit.geometry_covers_point({"type": "MultiPoint", "coordinates": [[0, 0], [1, 2]]}, 1, 2)
    polygon = {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}
    assert audit.geometry_covers_point(polygon, 1, 1)
    multi = {"type": "MultiPolygon", "coordinates": [polygon["coordinates"], [[[10, 10], [11, 10], [11, 11], [10, 11], [10, 10]]]]}
    assert audit.geometry_covers_point(multi, 10.5, 10.5)
    assert not audit.geometry_covers_point(multi, 5, 5)


def test_antimeridian_holes_and_boundaries_use_full_geometry_not_bbox() -> None:
    antimeridian = {
        "type": "Polygon",
        "coordinates": [[[170, -10], [-170, -10], [-170, 10], [170, 10], [170, -10]]],
    }
    assert audit.geometry_covers_point(antimeridian, 180, 0)
    assert audit.geometry_covers_point(antimeridian, -180, 0)
    assert not audit.geometry_covers_point(antimeridian, 0, 0)
    hole = {
        "type": "Polygon", "coordinates": [
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
            [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]],
        ],
    }
    assert not audit.geometry_covers_point(hole, 5, 5)
    assert audit.geometry_covers_point(hole, 4, 5)  # touching a hole boundary touches polygon
    assert audit.geometry_covers_point(hole, 0, 5)
    assert audit.geometry_covers_point(hole, 2, 2)


def test_malformed_geometry_is_rejected_and_bbox_never_substitutes() -> None:
    malformed = [
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1]]]},
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1]]]},
        {"type": "Polygon", "coordinates": [[[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]]]},
        {"type": "GeometryCollection", "geometries": []},
        {"type": "Point", "coordinates": [float("nan"), 0]},
    ]
    for geometry in malformed:
        with pytest.raises(audit.TargetCatalogAuditError):
            audit.geometry_covers_point(geometry, 0, 0)
    outside = item(geometry={"type": "Point", "coordinates": [1, 1]})
    outside["bbox"] = [-10, -10, 10, 10]
    with pytest.raises(audit.TargetCatalogAuditError, match="Full item geometry"):
        audit.validate_candidate(outside, source_row(), "sentinel_2_l1c", protocol())


def test_matching_multi_geometry_still_validates_malformed_trailing_members() -> None:
    multi_point = {"type": "MultiPoint", "coordinates": [[0, 0], [float("nan"), 1]]}
    with pytest.raises(audit.TargetCatalogAuditError):
        audit.geometry_covers_point(multi_point, 0, 0)
    multi_polygon = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]],
            [[[10, 10], [11, 10], [11, 11]]],
        ],
    }
    with pytest.raises(audit.TargetCatalogAuditError):
        audit.geometry_covers_point(multi_polygon, 0, 0)


def test_fractional_item_timestamp_is_preserved_exactly() -> None:
    candidate = audit.validate_candidate(
        item(when="2021-01-02T03:04:05.125Z"), source_row(),
        "sentinel_2_l1c", protocol(),
    )
    assert candidate["target_datetime"] == "2021-01-02T03:04:05.125000Z"
    assert candidate["source_target_offset_seconds"] == pytest.approx(0.125)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"collection": "wrong"}, "collection"),
        ({"id": "S3_fixture"}, "platform"),
        ({"properties.landsat:correction": "L1GT"}, "L1TP"),
        ({"properties.landsat:collection_category": "T2"}, "Tier 1"),
    ],
)
def test_collection_platform_and_landsat_tier_filters(mutation: dict[str, object], message: str) -> None:
    value = protocol()
    sensor = "landsat_8_9_level_1" if any("landsat:" in key for key in mutation) else "sentinel_2_l1c"
    candidate = item(sensor=sensor)
    for key, replacement in mutation.items():
        if key.startswith("properties."):
            candidate["properties"][key.removeprefix("properties.")] = replacement
        else:
            candidate[key] = replacement
    with pytest.raises(audit.TargetCatalogAuditError, match=message):
        audit.validate_candidate(candidate, source_row(), sensor, value)


def test_provider_extras_are_sanitized_but_cloud_and_100_item_ceiling_fail() -> None:
    value = protocol()
    extra = item()
    extra["assets"] = {"unexpected": {"href": "https://example.invalid/asset.tif"}}
    extra["links"] = [{"href": "https://example.invalid/item"}]
    extra["stac_version"] = "1.0.0"
    extra["properties"]["platform"] = "sentinel-2a"
    sanitized = audit.validate_candidate(extra, source_row(), "sentinel_2_l1c", value)
    encoded = json.dumps(sanitized)
    assert "assets" not in sanitized and "links" not in sanitized
    assert "platform" not in encoded and "example.invalid" not in encoded

    collection = {
        "type": "FeatureCollection", "features": [extra],
        "links": [{"href": "https://example.invalid/next"}],
        "stac_version": "1.0.0",
    }
    parsed = audit.parse_feature_collection(collection, source_row(), "sentinel_2_l1c", value)
    assert len(parsed) == 1 and "example.invalid" not in json.dumps(parsed)
    for cloud in (-0.1, 100.1, float("nan")):
        with pytest.raises(audit.TargetCatalogAuditError):
            audit.validate_candidate(item(cloud=cloud), source_row(), "sentinel_2_l1c", value)
    payload = {"type": "FeatureCollection", "features": [item(item_id=f"S2A_{index}") for index in range(100)]}
    with pytest.raises(audit.TargetCatalogAuditError, match="100-item"):
        audit.parse_feature_collection(payload, source_row(), "sentinel_2_l1c", value)


def test_two_and_three_dimensional_bbox_indices_are_validated_correctly() -> None:
    assert audit._validate_bbox([-10, -5, 10, 5]) == [-10.0, -5.0, 10.0, 5.0]
    assert audit._validate_bbox([-10, -5, 100, 10, 5, 200]) == [
        -10.0, -5.0, 100.0, 10.0, 5.0, 200.0,
    ]
    with pytest.raises(audit.TargetCatalogAuditError, match="longitude"):
        audit._validate_bbox([-10, -5, 100, 181, 5, 200])
    with pytest.raises(audit.TargetCatalogAuditError, match="latitude"):
        audit._validate_bbox([-10, -5, 100, 10, 91, 200])
    with pytest.raises(audit.TargetCatalogAuditError, match="vertical"):
        audit._validate_bbox([-10, -5, 200, 10, 5, 100])


def test_selection_is_deterministic_by_offset_cloud_then_id_and_keeps_both_sensors() -> None:
    row = source_row()
    base = selected_row(1)
    candidates = []
    for target, offset, cloud in (("S2A_z", 10, 1), ("S2A_b", 5, 9), ("S2A_a", 5, 9)):
        candidates.append({
            **base, **audit.source_identity(row), "target_sensor": "sentinel_2_l1c",
            "target_item_id": target, "absolute_source_target_offset_seconds": offset,
            "source_target_offset_seconds": offset, "eo_cloud_cover": cloud,
        })
    candidates.append({
        **base, **audit.source_identity(row), "target_sensor": "landsat_8_9_level_1",
        "target_item_id": "LC08_a", "absolute_source_target_offset_seconds": 50,
    })
    selected = audit.select_candidates(list(reversed(candidates)))
    assert len(selected) == 2
    assert {entry["target_item_id"] for entry in selected} == {"S2A_a", "LC08_a"}


def test_dual_sensor_pairs_cannot_manufacture_distinct_source_gate() -> None:
    value = protocol()
    selected: list[dict[str, object]] = []
    for index in range(14):
        selected.append(selected_row(index, sensor="sentinel_2_l1c", target=f"S2A_{index}"))
        selected.append(selected_row(index, sensor="landsat_8_9_level_1", target=f"LC08_{index}"))
    gates, counts = audit.evaluate_gates(selected, value, all_valid=True)
    assert gates["minimum_selected_source_sensor_pairs"]["pass"] is True
    assert gates["minimum_distinct_source_observations_with_pair"]["pass"] is False
    assert counts["selected_source_sensor_pairs"] == 28
    assert counts["distinct_source_observations"] == 14


def test_20_site_component_and_item_gate_boundaries() -> None:
    value = protocol()
    selected = [selected_row(index) for index in range(28)]
    gates, _ = audit.evaluate_gates(selected, value, all_valid=True)
    for name in (
        "minimum_distinct_sites_with_pair", "minimum_novel_25km_components_with_pair",
        "minimum_distinct_target_item_ids",
    ):
        assert gates[name]["pass"] is True
    collapsed = [
        {**row, "site_ID": f"site-{index % 19}", "component_id": f"component-{index % 19}", "target_item_id": f"item-{index % 19}"}
        for index, row in enumerate(selected)
    ]
    gates, _ = audit.evaluate_gates(collapsed, value, all_valid=True)
    assert gates["minimum_distinct_sites_with_pair"]["pass"] is False
    assert gates["minimum_novel_25km_components_with_pair"]["pass"] is False
    assert gates["minimum_distinct_target_item_ids"]["pass"] is False


def test_fail_closed_reports_remove_only_stale_candidate_outputs_and_prove_forbidden_resources(tmp_path: Path) -> None:
    value = protocol()
    value["outputs"] = {
        **value["outputs"], "ignored_candidates": "ignored/candidates.jsonl",
        "ignored_selected_pairs": "ignored/selected.jsonl",
        "compact_json": "reports/report.json", "compact_markdown": "reports/report.md",
    }
    for key in ("ignored_candidates", "ignored_selected_pairs"):
        path = tmp_path / value["outputs"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n", encoding="utf-8")
    request_log = tmp_path / "ignored/requests.jsonl"
    response_log = tmp_path / "ignored/responses.jsonl"
    request_log.write_text("append-only\n", encoding="utf-8")
    response_log.write_text("append-only\n", encoding="utf-8")
    audit._remove_stale_outputs(value, root=tmp_path)
    assert not (tmp_path / value["outputs"]["ignored_candidates"]).exists()
    assert not (tmp_path / value["outputs"]["ignored_selected_pairs"]).exists()
    assert request_log.read_text() == "append-only\n"
    assert response_log.read_text() == "append-only\n"

    report = audit.build_failure_report(value, audit.TargetCatalogAuditError("fixture"))
    audit._write_reports(value, report, root=tmp_path)
    stored = json.loads((tmp_path / "reports/report.json").read_text())
    assert stored["decision"] == "FAIL"
    assert all(gate["pass"] is False for gate in stored["gates"].values())
    assert all(accessed is False for accessed in stored["access_boundary_proof"].values())
    markdown = (tmp_path / "reports/report.md").read_text()
    assert "target_assets_accessed: False" in markdown
    assert "reference_catalog_queried: False" in markdown
    assert "protected_outcomes_accessed: False" in markdown
    assert "model_artifacts_accessed: False" in markdown
    assert "Source JSONL mutated: False" in markdown


def test_execution_preflight_failure_removes_stale_outputs_and_writes_fail_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = protocol()
    value["outputs"] = {
        **value["outputs"], "ignored_candidates": "ignored/candidates.jsonl",
        "ignored_selected_pairs": "ignored/selected.jsonl",
        "compact_json": "reports/report.json", "compact_markdown": "reports/report.md",
    }
    for key in ("ignored_candidates", "ignored_selected_pairs"):
        path = tmp_path / value["outputs"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(audit, "load_protocol", lambda: value)
    monkeypatch.setattr(
        audit, "validate_frozen_inputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(audit.TargetCatalogAuditError("preflight")),
    )
    with pytest.raises(audit.TargetCatalogAuditError, match="preflight"):
        audit.execute_network_audit(root=tmp_path, session=Session([]))
    assert not (tmp_path / value["outputs"]["ignored_candidates"]).exists()
    assert not (tmp_path / value["outputs"]["ignored_selected_pairs"]).exists()
    report = json.loads((tmp_path / "reports/report.json").read_text())
    assert report["decision"] == "FAIL"
    assert report["queries"]["expected"] == 352
    assert all(gate["pass"] is False for gate in report["gates"].values())
