from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tools.audit_carbon_mapper_tanager_nulls as audit

SCENE_ID = "11111111-1111-4111-8111-111111111111"
SCENE_NAME = "tan20260101t120000"
SOURCE_NAME = "CH4_oilgas_100m_-102.300_31.800"


def _annotated_scene(
    *,
    scene_id: str = SCENE_ID,
    name: str = SCENE_NAME,
    mission_phase: str = "production",
    cloud: int = 10,
    plume_count: int = 0,
) -> dict[str, object]:
    return {
        "id": scene_id,
        "name": name,
        "instrument": "tan",
        "mission_phase": mission_phase,
        "timestamp": "2026-01-01T12:00:00Z",
        "published_at": "2026-01-02T12:00:00Z",
        "not_cloudy": True,
        "cloud_cover_pct_assessed": cloud,
        "bounds": [-102.5, 31.5, -102.0, 32.0],
        "published_plume_count": plume_count,
    }


def _candidate() -> dict[str, object]:
    return {
        "source_name": SOURCE_NAME,
        "latitude": 31.8,
        "longitude": -102.3,
        "observation_date_count": 2,
        "detection_date_count": 1,
        "observation_scenes_names": [SCENE_NAME],
    }


def _detail(*, plume_count: int = 0) -> dict[str, object]:
    scene = {
        "id": SCENE_ID,
        "name": SCENE_NAME,
        "timestamp": "2026-01-01T12:00:00Z",
        "instrument": "tan",
    }
    daily_scene = {
        **scene,
        "counts_as_null_detection": True,
        "counts_toward_daily_emissions": True,
        "has_detection": False,
        "has_non_null_emission": False,
    }
    return {
        "source_name": SOURCE_NAME,
        "point": {"type": "Point", "coordinates": [-102.3, 31.8]},
        "scenes": [scene],
        "observation_dates": ["2026-01-01"],
        "detection_dates": [],
        "explanation": {
            "summary": {"number_of_null_detect_days": 1},
            "daily_breakdown": [
                {
                    "date": "2026-01-01",
                    "has_null_detection": True,
                    "scene_names": [SCENE_NAME],
                    "observation_scene_count": 1,
                    "daily_emission_scene_count": 1,
                    "null_detection_scene_count": 1,
                    "detection_scene_count": 0,
                    "plume_count": plume_count,
                    "scenes": [daily_scene],
                }
            ],
        },
    }


def test_frozen_protocol_local_inputs_and_openapi_validate_offline() -> None:
    plan = audit.validation_plan()
    assert plan["mode"] == "validation_only"
    assert plan["network_executed"] is False
    assert plan["carbon_mapper_catalog_enumerated"] is False
    assert plan["image_assets_accessed"] is False
    assert plan["target_catalog_accessed"] is False
    assert plan["protocol"]["sha256"] == audit.EXPECTED_PROTOCOL_SHA256
    assert plan["openapi"]["optional_auth_routes_validated"] == 3


def test_safe_default_cli_does_not_execute(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        audit,
        "validation_plan",
        lambda: {"mode": "validation_only", "network_executed": False},
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("safe default attempted catalog enumeration")

    monkeypatch.setattr(audit, "execute_metadata_audit", forbidden)
    assert audit.main([]) == 0
    assert '"network_executed": false' in capsys.readouterr().out
    assert audit.build_parser().parse_args([]).execute_metadata_audit is False


@pytest.mark.parametrize(
    "url",
    [
        "http://api.carbonmapper.org/api/v1/catalog/sources.geojson",
        "https://example.test/api/v1/catalog/sources.geojson",
        audit.SOURCES_URL + "&limit=1",
        "https://api.carbonmapper.org/api/v1/catalog/scenes",
        "https://api.carbonmapper.org/api/v1/catalog/plumes",
        audit.ANNOTATED_SCENES_TEMPLATE.format(offset=10_000),
    ],
)
def test_url_allowlist_rejects_nonfrozen_routes(url: str) -> None:
    with pytest.raises(audit.CarbonMapperAuditError):
        audit.validate_api_url(url)
    audit.validate_api_url(audit.SOURCES_URL)
    audit.validate_api_url(audit.SOURCE_DETAIL_TEMPLATE.format(source_name=SOURCE_NAME))


class _Response:
    def __init__(
        self,
        payload: bytes,
        *,
        url: str = audit.SOURCES_URL,
        status: int = 200,
        content_type: str = "application/json",
        history: list[object] | None = None,
        declared_length: int | None = None,
    ) -> None:
        self.payload = payload
        self.url = url
        self.status_code = status
        self.history = [] if history is None else history
        self.closed = False
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(
                len(payload) if declared_length is None else declared_length
            ),
        }

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [
            self.payload[index : index + chunk_size]
            for index in range(0, len(self.payload), chunk_size)
        ]

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def get(self, *args: object, **kwargs: object) -> _Response:
        self.calls.append((args, kwargs))
        return self.responses.pop(0)


def test_http_client_rejects_redirect_html_auth_and_caps(tmp_path: Path) -> None:
    cases = [
        (_Response(b"{}", history=[object()]), "Redirected"),
        (_Response(b"<html>login</html>", content_type="text/html"), "not JSON"),
        (_Response(b"{}", status=401), "status rejected"),
        (
            _Response(b"{}", declared_length=audit.MAX_RESPONSE_BYTES + 1),
            "per-response cap",
        ),
    ]
    for response, message in cases:
        client = audit.HttpAuditClient(_Session([response]), sleep=lambda _: None)
        with pytest.raises(audit.CarbonMapperAuditError, match=message):
            client.get_json(audit.SOURCES_URL, cache_path=tmp_path / hashlib.sha256(message.encode()).hexdigest())


def test_http_client_retries_only_frozen_statuses_and_caches(tmp_path: Path) -> None:
    payload = json.dumps({"type": "FeatureCollection", "features": []}).encode()
    retry = _Response(b"", status=503)
    success = _Response(payload)
    session = _Session([retry, success])
    client = audit.HttpAuditClient(session, sleep=lambda _: None)
    cache = tmp_path / "sources.geojson"
    value, receipt = client.get_json(audit.SOURCES_URL, cache_path=cache)
    assert value["features"] == []
    assert receipt["attempts"] == 2
    assert retry.closed is True
    assert len(session.calls) == 2
    second = audit.HttpAuditClient(_Session([]), sleep=lambda _: None)
    _, cached = second.get_json(audit.SOURCES_URL, cache_path=cache)
    assert cached["cache_hit"] is True
    assert second.total_network_bytes == 0


def test_source_feature_filter_and_request_order_are_deterministic() -> None:
    features = []
    for suffix, observations, detections in [("a", 2, 1), ("b", 3, 1)]:
        name = f"CH4_{suffix}_100m_-102.300_31.800"
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-102.3, 31.8]},
                "properties": {
                    "source_name": name,
                    "gas": "CH4",
                    "observation_date_count": observations,
                    "detection_date_count": detections,
                    "observation_scenes_names": [SCENE_NAME],
                },
            }
        )
    rejected = json.loads(json.dumps(features[0]))
    rejected["properties"]["observation_date_count"] = 1
    rejected["properties"]["detection_date_count"] = 1
    rejected["properties"]["source_name"] = "CH4_c_100m_-102.300_31.800"
    payload = {"type": "FeatureCollection", "features": [features[1], rejected, features[0]]}
    rows, counts = audit.parse_source_features(payload)
    expected = sorted(
        [features[0]["properties"]["source_name"], features[1]["properties"]["source_name"]],
        key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value),
    )
    assert [row["source_name"] for row in rows] == expected
    assert counts["no_null_day_headroom"] == 1


def test_annotated_scene_rejects_first_light_cloud_and_bad_bounds() -> None:
    with pytest.raises(audit.CarbonMapperAuditError, match="production-phase"):
        audit.parse_annotated_scene(_annotated_scene(mission_phase="first_light"))
    with pytest.raises(audit.CarbonMapperAuditError, match="25%"):
        audit.parse_annotated_scene(_annotated_scene(cloud=26))
    bad = _annotated_scene()
    bad["bounds"] = [-102.0, 31.5, -102.5, 32.0]
    with pytest.raises(audit.CarbonMapperAuditError, match="bounds"):
        audit.parse_annotated_scene(bad)


def test_authoritative_null_is_source_local_and_ignores_scene_wide_plume_count() -> None:
    annotated = audit.parse_annotated_scene(_annotated_scene(plume_count=17))
    rows = audit.authoritative_null_rows(
        candidate=_candidate(),
        detail=_detail(plume_count=3),
        annotated_by_id={SCENE_ID: annotated},
    )
    assert len(rows) == 1
    assert rows[0]["published_plume_count_scene_wide_not_used_for_label"] == 17
    assert rows[0]["passes_authoritative_null_contract"] is True
    assert rows[0]["eligible_for_target_catalog"] is False
    assert "not physical zero methane" in rows[0]["label_claim"]


def test_authoritative_null_rejects_counter_and_identity_disagreement() -> None:
    annotated = audit.parse_annotated_scene(_annotated_scene())
    bad_count = _detail()
    bad_count["explanation"]["daily_breakdown"][0]["null_detection_scene_count"] = 0
    with pytest.raises(audit.CarbonMapperAuditError, match="counter mismatch"):
        audit.authoritative_null_rows(
            candidate=_candidate(), detail=bad_count, annotated_by_id={SCENE_ID: annotated}
        )
    bad_name = _detail()
    bad_name["explanation"]["daily_breakdown"][0]["scenes"][0]["name"] = "other"
    with pytest.raises(audit.CarbonMapperAuditError, match="scene-name mismatch|scene mismatch"):
        audit.authoritative_null_rows(
            candidate=_candidate(), detail=bad_name, annotated_by_id={SCENE_ID: annotated}
        )


def test_selection_cap_is_deterministic_and_rows_remain_unauthorized() -> None:
    rows = [
        {
            "source_name": SOURCE_NAME,
            "scene_id": f"{index:08x}-1111-4111-8111-{index:012x}",
            "eligible_for_target_catalog": False,
        }
        for index in range(8)
    ]
    selected = audit.select_rows_per_source(list(reversed(rows)))
    expected = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(
                (SOURCE_NAME + "\0" + str(row["scene_id"])).encode()
            ).hexdigest(),
            row["scene_id"],
        ),
    )[:4]
    assert [row["scene_id"] for row in selected] == [
        row["scene_id"] for row in expected
    ]
    assert all(row["eligible_for_target_catalog"] is False for row in selected)


def test_spatial_filter_uses_inclusive_exclusion_and_conservative_component_gate() -> None:
    rows = [
        {
            "row_id": "a",
            "source_name": "source-a",
            "latitude": 0.0,
            "longitude": 0.0,
            "eligible_for_target_catalog": False,
        },
        {
            "row_id": "b",
            "source_name": "source-b",
            "latitude": 10.0,
            "longitude": 10.0,
            "eligible_for_target_catalog": False,
        },
    ]
    filtered = audit.spatial_filter_rows(
        rows,
        all_mars_locations={"mars": (0.0, 0.0)},
        protected_mars_locations={"test": (0.0, 0.0)},
        prior_negative_coordinates={"prior": (20.0, 20.0)},
    )
    assert filtered[0]["passes_frozen_spatial_filter"] is False
    assert filtered[1]["passes_frozen_spatial_filter"] is True
    assert filtered[1]["component_novel_beyond_all_mars_25km"] is True
    assert all(row["eligible_for_target_catalog"] is False for row in filtered)
