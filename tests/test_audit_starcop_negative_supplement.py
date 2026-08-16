from __future__ import annotations

import hashlib
import io
import struct
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

import tools.audit_starcop_negative_supplement as audit
import tools.starcop_sparse_zip as sparse

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

    padded_source_window = _id(row=-29, column=575, width=151, height=151)
    parsed = audit.parse_manifest_payload(
        _payload([_csv_row(padded_source_window, has_plume="true")])
    )
    assert parsed[0].source_row_offset == -29

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


class _SparseResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        url: str,
        start: int,
        end: int,
        total: int,
        status: int = 206,
        content_type: str = "application/octet-stream",
        content_encoding: str | None = None,
        content_range: str | None = None,
        history: list[object] | None = None,
    ) -> None:
        self.payload = payload
        self.url = url
        self.status_code = status
        self.history = [] if history is None else history
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(payload)),
            "Content-Range": (
                f"bytes {start}-{end}/{total}"
                if content_range is None
                else content_range
            ),
        }
        if content_encoding is not None:
            self.headers["Content-Encoding"] = content_encoding

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [
            self.payload[index : index + chunk_size]
            for index in range(0, len(self.payload), chunk_size)
        ]


class _BytesRangeSession:
    def __init__(self, payload: bytes, url: str) -> None:
        self.payload = payload
        self.url = url
        self.status = 206
        self.content_type = "application/octet-stream"
        self.content_encoding: str | None = None
        self.content_range: str | None = None
        self.history: list[object] = []
        self.calls: list[tuple[int, int]] = []

    def get(self, url: str, **kwargs: object) -> _SparseResponse:
        assert url == self.url
        raw_range = str(kwargs["headers"]["Range"])
        start, end = (int(value) for value in raw_range.removeprefix("bytes=").split("-"))
        self.calls.append((start, end))
        return _SparseResponse(
            self.payload[start : end + 1],
            url=url,
            start=start,
            end=end,
            total=len(self.payload),
            status=self.status,
            content_type=self.content_type,
            content_encoding=self.content_encoding,
            content_range=self.content_range,
            history=self.history,
        )


def _archive_spec(payload: bytes) -> sparse.ArchiveSpec:
    name = "STARCOP_train_easy.zip"
    return sparse.ArchiveSpec(
        name=name,
        size=len(payload),
        declared_md5="0" * 32,
        url=sparse.archive_url(name),
    )


def _reader(payload: bytes) -> tuple[sparse.RangeReader, _BytesRangeSession]:
    spec = _archive_spec(payload)
    session = _BytesRangeSession(payload, spec.url)
    return (
        sparse.RangeReader(
            spec=spec,
            session=session,
            budget=sparse.RangeBudget(10_000_000, 10_000_000),
        ),
        session,
    )


def _zip_payload(
    members: list[tuple[str, bytes]], *, compression: int = zipfile.ZIP_DEFLATED
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, payload in members:
            archive.writestr(name, payload)
    return output.getvalue()


def test_stage_b_authorization_binds_pass_manifest_and_selected_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, rows = audit.authorize_stage_b()
    assert len(rows) == 1009
    assert all(row["eligible_for_target_catalog"] is False for row in rows)
    assert len(audit.stage_b_archive_specs(protocol)) == 6
    parser = audit.build_parser()
    assert parser.parse_args([]).execute_stage_b_sparse_masks is False
    assert parser.parse_args(
        ["--execute-stage-b-sparse-masks"]
    ).execute_stage_b_sparse_masks is True
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--execute-train-manifest", "--execute-stage-b-sparse-masks"]
        )

    monkeypatch.setattr(audit, "EXPECTED_STAGE_A_REPORT_SHA256", "0" * 64)
    with pytest.raises(audit.StarcopAuditError, match="PASS receipt SHA-256"):
        audit.authorize_stage_b()


def test_archive_urls_and_ranges_fail_closed_and_failed_attempts_keep_budget() -> None:
    payload = b"0123456789"
    spec = _archive_spec(payload)
    sparse.validate_archive_spec(spec, frozenset({spec.name}))
    forbidden = replace(spec, url=spec.url + "?download=1")
    with pytest.raises(sparse.SparseZipError, match="exact frozen Zenodo"):
        sparse.validate_archive_spec(forbidden, frozenset({spec.name}))
    test_archive = replace(
        spec,
        name="STARCOP_test.zip",
        url=sparse.archive_url("STARCOP_test.zip"),
    )
    with pytest.raises(sparse.SparseZipError, match="six frozen train"):
        sparse.validate_archive_spec(test_archive, frozenset({spec.name}))

    session = _BytesRangeSession(payload, spec.url)
    budget = sparse.RangeBudget(central_limit=4, label_limit=4)
    reader = sparse.RangeReader(spec=spec, session=session, budget=budget)
    assert reader.read(0, 2, category="central_directory") == b"012"
    assert budget.central_bytes == 3
    with pytest.raises(sparse.SparseZipError, match="cap exceeded"):
        reader.read(3, 4, category="central_directory")

    failed_session = _BytesRangeSession(payload, spec.url)
    failed_session.status = 200
    failed_budget = sparse.RangeBudget(central_limit=4, label_limit=4)
    failed_reader = sparse.RangeReader(
        spec=spec, session=failed_session, budget=failed_budget
    )
    with pytest.raises(sparse.SparseZipError, match="status 206"):
        failed_reader.read(0, 2, category="central_directory")
    assert failed_budget.central_bytes == 3
    with pytest.raises(sparse.SparseZipError, match="cap exceeded"):
        failed_reader.read(3, 4, category="central_directory")

    malformed_session = _BytesRangeSession(payload, spec.url)
    malformed_session.content_range = "bytes 0-2/*"
    malformed = sparse.RangeReader(
        spec=spec,
        session=malformed_session,
        budget=sparse.RangeBudget(10, 10),
    )
    with pytest.raises(sparse.SparseZipError, match="Content-Range"):
        malformed.read(0, 2, category="central_directory")

    encoded_session = _BytesRangeSession(payload, spec.url)
    encoded_session.content_encoding = "gzip"
    encoded = sparse.RangeReader(
        spec=spec,
        session=encoded_session,
        budget=sparse.RangeBudget(10, 10),
    )
    with pytest.raises(sparse.SparseZipError, match="Encoded range"):
        encoded.read(0, 2, category="central_directory")


def test_zip_member_allowlist_crc_decompression_and_local_header_checks() -> None:
    sample_id = _id()
    member = f"{sample_id}/labelbinary.tif"
    payload = _zip_payload([(member, b"zero-mask-fixture")])
    reader, _session = _reader(payload)
    directory = sparse.read_zip_directory(reader)
    locations = sparse.resolve_selected_members(
        {reader.spec.name: directory}, [sample_id]
    )
    archive_name, entry = locations[sample_id]
    assert archive_name == reader.spec.name
    assert sparse.fetch_label_member(
        reader,
        entry,
        sample_id=sample_id,
        maximum_uncompressed_bytes=1024,
    ) == b"zero-mask-fixture"
    with pytest.raises(sparse.SparseZipError, match="Only exact selected"):
        sparse.fetch_label_member(
            reader,
            entry,
            sample_id="ang20200101t000000_r0_c0_w512_h512",
            maximum_uncompressed_bytes=1024,
        )
    with pytest.raises(sparse.SparseZipError, match="exactly once"):
        sparse.resolve_selected_members(
            {reader.spec.name: directory}, [_id(row=512)]
        )
    with pytest.raises(sparse.SparseZipError, match="inconsistent"):
        sparse.fetch_label_member(
            reader,
            replace(entry, compression=0),
            sample_id=sample_id,
            maximum_uncompressed_bytes=1024,
        )

    stored = bytearray(_zip_payload([(member, b"abcdef")], compression=zipfile.ZIP_STORED))
    stored_reader, _ = _reader(bytes(stored))
    stored_entry = sparse.read_zip_directory(stored_reader).entries[0]
    name_length, extra_length = struct.unpack_from(
        "<HH", stored, stored_entry.local_header_offset + 26
    )
    data_offset = stored_entry.local_header_offset + 30 + name_length + extra_length
    stored[data_offset] ^= 0x01
    corrupt_reader, _ = _reader(bytes(stored))
    with pytest.raises(sparse.SparseZipError, match="CRC-32"):
        sparse.fetch_label_member(
            corrupt_reader,
            stored_entry,
            sample_id=sample_id,
            maximum_uncompressed_bytes=1024,
        )


def test_bounded_deflate_rejects_forged_declared_size_zip_bomb() -> None:
    sample_id = _id()
    member = f"{sample_id}/labelbinary.tif"
    archive = bytearray(_zip_payload([(member, b"0" * 4096)]))
    reader, _ = _reader(bytes(archive))
    entry = sparse.read_zip_directory(reader).entries[0]
    # Forge both the supplied central entry and local header to declare one byte;
    # the compressed stream still expands to 4096 bytes.
    struct.pack_into("<L", archive, entry.local_header_offset + 22, 1)
    forged_entry = replace(entry, uncompressed_size=1)
    forged_reader, _ = _reader(bytes(archive))
    with pytest.raises(sparse.SparseZipError, match="expands beyond"):
        sparse.fetch_label_member(
            forged_reader,
            forged_entry,
            sample_id=sample_id,
            maximum_uncompressed_bytes=1024,
        )


def test_duplicate_and_traversal_central_members_are_rejected() -> None:
    sample_id = _id()
    member = f"{sample_id}/labelbinary.tif"
    with pytest.warns(UserWarning):
        duplicate = _zip_payload([(member, b"a"), (member, b"b")])
    duplicate_reader, _ = _reader(duplicate)
    with pytest.raises(sparse.SparseZipError, match="Duplicate ZIP member"):
        sparse.read_zip_directory(duplicate_reader)

    traversal = _zip_payload([("../labelbinary.tif", b"a")])
    traversal_reader, _ = _reader(traversal)
    with pytest.raises(sparse.SparseZipError, match="traversal"):
        sparse.read_zip_directory(traversal_reader)


class _VirtualRangeSession:
    def __init__(
        self, *, total: int, url: str, segments: list[tuple[int, bytes]]
    ) -> None:
        self.total = total
        self.url = url
        self.segments = segments
        self.calls: list[tuple[int, int]] = []

    def get(self, url: str, **kwargs: object) -> _SparseResponse:
        assert url == self.url
        raw = str(kwargs["headers"]["Range"]).removeprefix("bytes=")
        start, end = (int(value) for value in raw.split("-"))
        result = bytearray(end - start + 1)
        for segment_start, segment in self.segments:
            overlap_start = max(start, segment_start)
            overlap_end = min(end + 1, segment_start + len(segment))
            if overlap_start < overlap_end:
                result[overlap_start - start : overlap_end - start] = segment[
                    overlap_start - segment_start : overlap_end - segment_start
                ]
        self.calls.append((start, end))
        return _SparseResponse(
            bytes(result), url=url, start=start, end=end, total=self.total
        )


def test_zip64_locator_uses_64bit_central_offset_and_size() -> None:
    total = 5_000_000_000
    central_offset = 1_000
    zip64_offset = 2_000
    name = b"folder/labelbinary.tif"
    central = struct.pack(
        "<4s6H3L5H2L",
        sparse.CENTRAL_SIGNATURE,
        45,
        20,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        len(name),
        0,
        0,
        0,
        0,
        0,
        500,
    ) + name
    zip64 = struct.pack(
        "<4sQ2H2L4Q",
        sparse.ZIP64_EOCD_SIGNATURE,
        44,
        45,
        45,
        0,
        0,
        1,
        1,
        len(central),
        central_offset,
    )
    locator = struct.pack(
        "<4sLQL", sparse.ZIP64_LOCATOR_SIGNATURE, 0, zip64_offset, 1
    )
    eocd = struct.pack(
        "<4s4H2LH",
        sparse.EOCD_SIGNATURE,
        0,
        0,
        0xFFFF,
        0xFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0,
    )
    spec = sparse.ArchiveSpec(
        name="STARCOP_train_easy.zip",
        size=total,
        declared_md5="0" * 32,
        url=sparse.archive_url("STARCOP_train_easy.zip"),
    )
    session = _VirtualRangeSession(
        total=total,
        url=spec.url,
        segments=[
            (central_offset, central),
            (zip64_offset, zip64),
            (total - 42, locator + eocd),
        ],
    )
    reader = sparse.RangeReader(
        spec=spec,
        session=session,
        budget=sparse.RangeBudget(1_000_000, 1_000_000),
    )
    directory = sparse.read_zip_directory(reader)
    assert directory.zip64 is True
    assert directory.central_offset == central_offset
    assert directory.central_size == len(central)
    assert directory.entries[0].local_header_offset == 500
    assert (zip64_offset, zip64_offset + 55) in session.calls


def _geotiff(*, value: int = 0, crs: str = "EPSG:32611") -> bytes:
    data = np.full((512, 512), value, dtype=np.uint8)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=512,
            height=512,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=from_origin(500_000, 4_100_000, 30, 30),
        ) as dataset:
            dataset.write(data, 1)
        return memory.read()


def test_zero_mask_georeferencing_and_spatial_filter_components() -> None:
    sample_id = _id()
    member = f"{sample_id}/labelbinary.tif"
    decoded = audit.decode_zero_mask(
        _geotiff(),
        sample_id=sample_id,
        archive_name="STARCOP_train_easy.zip",
        label_member=member,
    )
    assert decoded["zero_mask_confirmed"] is True
    assert decoded["coordinate_resolved"] is True
    assert decoded["eligible_for_target_catalog"] is False
    with pytest.raises(audit.StarcopAuditError, match="not all zero"):
        audit.decode_zero_mask(
            _geotiff(value=1),
            sample_id=sample_id,
            archive_name="STARCOP_train_easy.zip",
            label_member=member,
        )
    with pytest.raises(audit.StarcopAuditError, match="projected CRS"):
        audit.decode_zero_mask(
            _geotiff(crs="EPSG:4326"),
            sample_id=sample_id,
            archive_name="STARCOP_train_easy.zip",
            label_member=member,
        )

    def row(identifier: str, latitude: float, longitude: float) -> dict[str, object]:
        return {
            **decoded,
            "sample_id": identifier,
            "tile": identifier.split("_r", 1)[0],
            "timestamp": "2020-01-01T12:00:00Z",
            "latitude": latitude,
            "longitude": longitude,
        }

    rows = [
        row(_id(row=0), 10.0, 10.0),
        row(_id(row=512), 20.0, 20.0),
        row(_id(row=1024), 30.0, 30.0),
        row(_id(row=1536), 30.05, 30.05),
        row(_id(row=2048), 40.0, 40.0),
    ]
    filtered = audit.filter_stage_b_rows(
        rows=rows,
        all_mars_locations={"development": (0.0, 0.0), "test": (10.0, 10.0)},
        protected_mars_locations={"test": (10.0, 10.0)},
        prior_negative_coordinates={"prior": (20.0, 20.0)},
    )
    assert all(item["eligible_for_target_catalog"] is False for item in filtered)
    passing = [item for item in filtered if item["passes_frozen_spatial_filter"]]
    assert len(passing) == 3
    assert passing[0]["group_id"] == passing[1]["group_id"]
    assert passing[2]["group_id"] != passing[0]["group_id"]
