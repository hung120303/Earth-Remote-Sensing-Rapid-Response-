from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import tools.audit_starcop_negative_supplement as audit

HEADER = (
    "id,has_plume,window_col_off,window_row_off,window_width,window_height,qplume\n"
)


def _id(
    flight: str = "ang20200101t120000",
    *,
    row: int = 0,
    column: int = 0,
    width: int = 512,
    height: int = 512,
) -> str:
    return f"{flight}_r{row}_c{column}_w{width}_h{height}"


def _csv_row(
    sample_id: str,
    *,
    has_plume: str = "false",
    qplume: str = "0",
    row_override: str | None = None,
) -> str:
    match = audit.ID_RE.fullmatch(sample_id)
    assert match is not None
    row = "0" if row_override is None else row_override
    return (
        f"{sample_id},{has_plume},0,{row},512,512,{qplume}\n"
    )


def _payload(rows: list[str]) -> bytes:
    return (HEADER + "".join(rows)).encode("utf-8")


def test_frozen_protocol_and_local_inputs_validate_without_manifest_access() -> None:
    plan = audit.validation_plan()
    assert plan["mode"] == "validation_only"
    assert plan["network_executed"] is False
    assert plan["starcop_manifest_opened"] is False
    assert plan["test_manifest_accessed"] is False
    assert plan["archive_accessed"] is False
    assert plan["target_catalog_accessed"] is False
    assert plan["protocol"]["sha256"] == audit.EXPECTED_PROTOCOL_SHA256


def test_safe_default_cli_never_calls_execution(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        audit,
        "validation_plan",
        lambda: {"mode": "validation_only", "network_executed": False},
    )

    def forbidden_execution() -> None:
        raise AssertionError("safe default attempted network execution")

    monkeypatch.setattr(audit, "execute_stage_a", forbidden_execution)
    assert audit.main([]) == 0
    assert '"network_executed": false' in capsys.readouterr().out
    assert audit.build_parser().parse_args([]).execute_train_manifest is False


@pytest.mark.parametrize("value", ["1", "0", "yes", "no", "", "null"])
def test_boolean_parser_is_strict(value: str) -> None:
    with pytest.raises(audit.StarcopAuditError, match="true or false"):
        audit.parse_strict_bool(value)
    assert audit.parse_strict_bool(" TRUE ") is True
    assert audit.parse_strict_bool("false") is False


def test_parser_rejects_duplicate_ids_and_bad_cached_window() -> None:
    sample_id = _id(row=512)
    duplicate = _payload([_csv_row(sample_id), _csv_row(sample_id)])
    with pytest.raises(audit.StarcopAuditError, match="Duplicate STARCOP ID"):
        audit.parse_manifest_payload(duplicate)

    mismatch = _payload([_csv_row(sample_id, row_override="512")])
    with pytest.raises(audit.StarcopAuditError, match="Cached chip-local window"):
        audit.parse_manifest_payload(mismatch)

    variable_source_window = _id(width=151, height=151)
    parsed = audit.parse_manifest_payload(_payload([_csv_row(variable_source_window)]))
    assert parsed[0].source_width == 151
    assert parsed[0].source_height == 151

    empty_source_window = _id(width=0, height=512)
    with pytest.raises(audit.StarcopAuditError, match="source window is empty"):
        audit.parse_manifest_payload(_payload([_csv_row(empty_source_window)]))

    invalid_time = _id(flight="ang20200231t120000")
    with pytest.raises(audit.StarcopAuditError, match="Invalid UTC"):
        audit.parse_manifest_payload(_payload([_csv_row(invalid_time)]))


def test_train_filename_gate_fires_before_open(tmp_path: Path) -> None:
    forbidden = tmp_path / "test.csv"
    # Invalid bytes prove the filename check precedes any open/CSV parsing.
    forbidden.write_bytes(b"not,a,csv")
    with pytest.raises(audit.StarcopAuditError, match="released train.csv filename"):
        audit.read_train_manifest(forbidden)


def test_selection_is_deterministic_and_independent_of_qplume() -> None:
    flight = "ang20200101t120000"
    ids = [_id(flight, row=index * 512) for index in range(8)]
    payload_a = _payload(
        [_csv_row(sample_id, qplume=str(index)) for index, sample_id in enumerate(ids)]
    )
    payload_b = _payload(
        [
            _csv_row(sample_id, qplume=str(100_000 - index))
            for index, sample_id in enumerate(ids)
        ]
    )
    selected_a = audit.select_negative_rows(audit.parse_manifest_payload(payload_a))
    selected_b = audit.select_negative_rows(audit.parse_manifest_payload(payload_b))
    expected = sorted(
        ids, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value)
    )[:4]
    assert [row.sample_id for row in selected_a] == expected
    assert [row.sample_id for row in selected_b] == expected
    assert all("qplume" not in audit.selected_record(row) for row in selected_a)
    assert all(audit.selected_record(row)["eligible_for_target_catalog"] is False for row in selected_a)
    assert selected_a[0].source_row_offset == int(
        audit.ID_RE.fullmatch(selected_a[0].sample_id).group("row")
    )


def test_stage_a_aggregate_gates_and_per_flight_cap() -> None:
    rows: list[audit.ManifestRow] = []
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for flight_index in range(50):
        flight = (start + timedelta(minutes=flight_index)).strftime("ang%Y%m%dt%H%M%S")
        for row_index in range(20):
            payload = _payload([_csv_row(_id(flight, row=row_index * 512))])
            rows.extend(audit.parse_manifest_payload(payload))
    _selected, report = audit.audit_rows(rows)
    assert report["decision"] == "PASS"
    assert report["counts"]["negative_rows"] == 1000
    assert report["counts"]["negative_bearing_flightlines"] == 50
    assert report["counts"]["selected_negative_rows"] == 200
    assert report["gates"]["maximum_selected_rows_per_flightline"]["observed_maximum"] == 4

    _, failed = audit.audit_rows(rows[:-20])
    assert failed["decision"] == "FAIL"
    assert failed["gates"]["minimum_negative_rows"]["passed"] is False
    assert failed["gates"]["minimum_negative_bearing_flightlines"]["passed"] is False


class _Response:
    def __init__(
        self,
        payload: bytes,
        *,
        url: str = audit.TRAIN_MANIFEST_URL,
        content_type: str = "text/csv",
        declared_length: int | None = None,
        history: list[object] | None = None,
    ) -> None:
        self.payload = payload
        self.url = url
        self.status_code = 200
        self.history = [] if history is None else history
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(
                len(payload) if declared_length is None else declared_length
            ),
        }

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [
            self.payload[index : index + chunk_size]
            for index in range(0, len(self.payload), chunk_size)
        ]


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def get(self, *args: object, **kwargs: object) -> _Response:
        self.calls.append((args, kwargs))
        return self.response


def test_url_html_redirect_and_stream_caps_fail_closed() -> None:
    forbidden_urls = [
        "https://zenodo.org/api/records/7863343/files/test.csv/content",
        "https://example.test/api/records/7863343/files/train.csv/content",
        audit.TRAIN_MANIFEST_URL + "?download=1",
        "http://zenodo.org/api/records/7863343/files/train.csv/content",
    ]
    for url in forbidden_urls:
        with pytest.raises(audit.StarcopAuditError, match="exact frozen Zenodo"):
            audit.validate_train_url(url)

    with pytest.raises(audit.StarcopAuditError, match="Content-Length"):
        audit.stream_manifest_payload(
            _Response(b"x", declared_length=audit.MAX_MANIFEST_BYTES + 1)
        )
    with pytest.raises(audit.StarcopAuditError, match="HTML/auth"):
        audit.stream_manifest_payload(
            _Response(b"<html>login</html>", content_type="text/html")
        )
    with pytest.raises(audit.StarcopAuditError, match="Redirected"):
        audit.stream_manifest_payload(_Response(b"x", history=[object()]))

    failed = _Response(b"x")
    failed.status_code = 401
    with pytest.raises(audit.StarcopAuditError, match="HTTP status 200"):
        audit.stream_manifest_payload(failed)

    oversized = _Response(b"x" * (audit.MAX_MANIFEST_BYTES + 1))
    oversized.headers.pop("Content-Length")
    with pytest.raises(audit.StarcopAuditError, match="Streamed"):
        audit.stream_manifest_payload(oversized)


def test_exact_identity_and_atomic_ignored_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload([_csv_row(_id())])
    monkeypatch.setattr(audit, "TRAIN_MANIFEST_BYTES", len(payload))
    monkeypatch.setattr(audit, "TRAIN_MANIFEST_MD5", audit.md5_bytes(payload))
    monkeypatch.setattr(audit, "IGNORED_ROOT", tmp_path / ".research/starcop")
    monkeypatch.setattr(
        audit, "CACHED_TRAIN_MANIFEST", audit.IGNORED_ROOT / "train.csv"
    )
    session = _Session(_Response(payload))
    path = audit.acquire_train_manifest(session=session)
    assert path.read_bytes() == payload
    assert not path.with_name("train.csv.part").exists()
    assert session.calls[0][0] == (audit.TRAIN_MANIFEST_URL,)
    assert session.calls[0][1]["allow_redirects"] is False
    assert "headers" not in session.calls[0][1]

    monkeypatch.setattr(audit, "TRAIN_MANIFEST_MD5", "0" * 32)
    with pytest.raises(audit.StarcopAuditError, match="MD5 mismatch"):
        audit.stream_manifest_payload(_Response(payload))
