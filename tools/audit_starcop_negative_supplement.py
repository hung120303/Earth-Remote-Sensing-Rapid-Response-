"""Audit the frozen STARCOP train manifest and sparse zero-mask metadata.

The safe default is validation-only: it validates the committed protocol and
its frozen local inputs without making a network request or opening a STARCOP
manifest. Explicit modes separately run Stage A or the preregistered Stage-B
sparse byte-range path. Stage B may reconstruct only selected train-negative
``labelbinary.tif`` members; test content, full archives, source imagery, and
target catalogs remain forbidden.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import requests
from pyproj import Transformer
from rasterio.io import MemoryFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_mars_hyperspectral_transfer import (
    FORBIDDEN_MARS_COLUMNS,
    SAFE_MARS_COLUMNS,
    read_mars_observations,
)
from tools.filter_jpl_cach4_metadata_eligibility import (
    filter_rows,
    load_prior_negative_coordinates,
    numeric_summary,
    official_test_locations,
)
from tools.starcop_sparse_zip import (
    ArchiveSpec,
    RangeBudget,
    RangeReader,
    SparseZipError,
    archive_url,
    fetch_label_member,
    read_zip_directory,
    resolve_selected_members,
    validate_archive_spec,
)

EXPECTED_PROTOCOL = ROOT / "configs/mars_starcop_negative_supplement_protocol.json"
EXPECTED_PROTOCOL_SHA256 = (
    "39887a53a093d33c645315239570b1a3ca4a08ad2d54723ae5301e476f9b9eec"
)
TRAIN_MANIFEST_NAME = "train.csv"
TRAIN_MANIFEST_URL = "https://zenodo.org/api/records/7863343/files/train.csv/content"
TRAIN_MANIFEST_BYTES = 1_038_970
TRAIN_MANIFEST_MD5 = "02f56b8e9759d01a1ee039f2eeaf4ebc"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
IGNORED_ROOT = ROOT / ".research/starcop_negative_supplement"
CACHED_TRAIN_MANIFEST = IGNORED_ROOT / TRAIN_MANIFEST_NAME
SELECTED_NEGATIVES_JSONL = IGNORED_ROOT / "stage_a_selected_train_negatives.jsonl"
COMPACT_REPORT = ROOT / "reports/acquisition/starcop_negative_supplement_stage_a.json"
EXPECTED_STAGE_A_REPORT_SHA256 = (
    "a80189174bf7bbe795720c0757000733e5d020fc56f4e85b2b4801449d368747"
)
RESOLVED_STAGE_B_JSONL = IGNORED_ROOT / "stage_b_resolved_train_negatives.jsonl"
FILTERED_STAGE_B_JSONL = IGNORED_ROOT / "stage_b_filtered_train_negatives.jsonl"
RANGE_RECEIPTS_JSONL = IGNORED_ROOT / "stage_b_range_receipts.jsonl"
STAGE_B_REPORT = ROOT / "reports/acquisition/starcop_negative_supplement_stage_b.json"
STAGE_B_MARKDOWN = ROOT / "reports/acquisition/STARCOP_NEGATIVE_SUPPLEMENT_STAGE_B.md"
MINIMUM_NEGATIVE_ROWS = 1_000
MINIMUM_NEGATIVE_FLIGHTLINES = 50
MAXIMUM_SELECTED_PER_FLIGHTLINE = 4
MAXIMUM_CENTRAL_DIRECTORY_BYTES = 536_870_912
MAXIMUM_UNCOMPRESSED_LABEL_BYTES = 1_048_576
MAXIMUM_DOWNLOADED_LABEL_BYTES = 268_435_456
STAGE_B_MINIMUM_RESOLVED_ROWS = 100
STAGE_B_MINIMUM_RESOLVED_FLIGHTLINES = 50
STAGE_B_MINIMUM_ELIGIBLE_COMPONENTS = 20
EXCLUSION_RADIUS_KM = 25.0

ID_RE = re.compile(
    r"^(?P<flight>ang(?P<date>\d{8})t(?P<time>\d{6}))_"
    r"r(?P<row>-?\d+)_c(?P<column>-?\d+)_w(?P<width>\d+)_h(?P<height>\d+)$"
)
INTEGER_RE = re.compile(r"^(?:0|[1-9]\d*)$")
REQUIRED_COLUMNS = frozenset(
    {
        "id",
        "has_plume",
        "window_col_off",
        "window_row_off",
        "window_width",
        "window_height",
    }
)


class StarcopAuditError(RuntimeError):
    """Raised when a frozen STARCOP Stage-A requirement is violated."""


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    flight: str
    timestamp: str
    source_row_offset: int
    source_column_offset: int
    source_width: int
    source_height: int
    has_plume: bool

    @property
    def selection_digest(self) -> str:
        return hashlib.sha256(self.sample_id.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_lf_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256_bytes(payload)


def md5_bytes(payload: bytes) -> str:
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    count = 0
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    partial.replace(path)
    return count


def validate_protocol_path(path: Path) -> None:
    if path.resolve() != EXPECTED_PROTOCOL.resolve():
        raise ValueError("Only the exact committed STARCOP protocol is permitted")


def load_protocol(path: Path = EXPECTED_PROTOCOL) -> dict[str, Any]:
    validate_protocol_path(path)
    observed = sha256_file(path)
    if observed != EXPECTED_PROTOCOL_SHA256:
        raise StarcopAuditError("Frozen STARCOP protocol SHA-256 mismatch")
    protocol = json.loads(path.read_text(encoding="utf-8"))

    train = protocol["frozen_source_files"]["train_manifest"]
    if train != {
        "name": TRAIN_MANIFEST_NAME,
        "bytes": TRAIN_MANIFEST_BYTES,
        "md5": TRAIN_MANIFEST_MD5,
        "url": TRAIN_MANIFEST_URL,
    }:
        raise StarcopAuditError("Frozen STARCOP train-manifest contract mismatch")
    if protocol["frozen_source_files"]["forbidden_files"] != [
        "test.csv",
        "STARCOP_test.zip",
    ]:
        raise StarcopAuditError("Frozen forbidden-file contract mismatch")
    gates = protocol["stage_a_train_manifest"]["gates"]
    if gates != {
        "minimum_negative_rows_in_released_train_manifest": MINIMUM_NEGATIVE_ROWS,
        "minimum_negative_bearing_flightlines": MINIMUM_NEGATIVE_FLIGHTLINES,
        "maximum_selected_rows_per_flightline": MAXIMUM_SELECTED_PER_FLIGHTLINE,
    }:
        raise StarcopAuditError("Frozen Stage-A gates mismatch")
    outputs = protocol["outputs"]
    if outputs["ignored_root"] != ".research/starcop_negative_supplement":
        raise StarcopAuditError("Frozen ignored-root contract mismatch")
    if (
        outputs["compact_stage_a_json"]
        != "reports/acquisition/starcop_negative_supplement_stage_a.json"
    ):
        raise StarcopAuditError("Frozen Stage-A report path mismatch")
    if outputs["compact_stage_b_json"] != (
        "reports/acquisition/starcop_negative_supplement_stage_b.json"
    ):
        raise StarcopAuditError("Frozen Stage-B report path mismatch")
    if outputs["compact_stage_b_markdown"] != (
        "reports/acquisition/STARCOP_NEGATIVE_SUPPLEMENT_STAGE_B.md"
    ):
        raise StarcopAuditError("Frozen Stage-B Markdown path mismatch")
    stage_b = protocol["stage_b_sparse_zero_mask_georeferencing"]
    if stage_b["content_caps"] != {
        "maximum_central_directory_bytes_total": MAXIMUM_CENTRAL_DIRECTORY_BYTES,
        "maximum_uncompressed_label_bytes_each": MAXIMUM_UNCOMPRESSED_LABEL_BYTES,
        "maximum_downloaded_label_bytes_total": MAXIMUM_DOWNLOADED_LABEL_BYTES,
    }:
        raise StarcopAuditError("Frozen Stage-B content caps mismatch")
    if stage_b["gates"] != {
        "minimum_resolved_selected_negative_rows": STAGE_B_MINIMUM_RESOLVED_ROWS,
        "minimum_resolved_negative_flightlines": STAGE_B_MINIMUM_RESOLVED_FLIGHTLINES,
        "minimum_eligible_25km_connected_components": STAGE_B_MINIMUM_ELIGIBLE_COMPONENTS,
        "all_retained_labels_and_coordinates_valid": True,
    }:
        raise StarcopAuditError("Frozen Stage-B gates mismatch")
    return protocol


def validate_frozen_local_inputs(protocol: dict[str, Any]) -> dict[str, object]:
    receipts: dict[str, object] = {}
    for role, specification in protocol["frozen_local_inputs"].items():
        path = ROOT / specification["path"]
        if not path.is_file():
            raise StarcopAuditError(f"Frozen local input is missing: {role}")
        if "normalized_lf_sha256" in specification:
            observed = normalized_lf_sha256(path)
            expected = specification["normalized_lf_sha256"]
            hash_kind = "normalized_lf_sha256"
        else:
            observed = sha256_file(path)
            expected = specification["sha256"]
            hash_kind = "sha256"
        if observed != expected:
            raise StarcopAuditError(f"Frozen local input identity mismatch: {role}")
        receipts[role] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            hash_kind: observed,
        }
    return receipts


def validation_plan() -> dict[str, object]:
    protocol = load_protocol()
    receipts = validate_frozen_local_inputs(protocol)
    return {
        "mode": "validation_only",
        "network_executed": False,
        "starcop_manifest_opened": False,
        "test_manifest_accessed": False,
        "archive_accessed": False,
        "target_catalog_accessed": False,
        "protocol": {
            "path": EXPECTED_PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256": EXPECTED_PROTOCOL_SHA256,
        },
        "frozen_local_inputs": receipts,
        "explicit_execution_flags": [
            "--execute-train-manifest",
            "--execute-stage-b-sparse-masks",
        ],
        "authorized_url": TRAIN_MANIFEST_URL,
        "maximum_streamed_bytes": MAX_MANIFEST_BYTES,
    }


def validate_train_url(url: str) -> None:
    parsed = urlsplit(url)
    expected = urlsplit(TRAIN_MANIFEST_URL)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "zenodo.org"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != expected.path
        or parsed.query
        or parsed.fragment
    ):
        raise StarcopAuditError("Only the exact frozen Zenodo train.csv URL is permitted")


def validate_train_filename(path: Path) -> None:
    if path.name != TRAIN_MANIFEST_NAME:
        raise StarcopAuditError("Only the exact released train.csv filename is permitted")


def _looks_like_html(payload: bytes) -> bool:
    prefix = payload[:4096].lstrip().lower()
    return prefix.startswith((b"<!doctype html", b"<html")) or b"<form" in prefix


def validate_manifest_response(response: Any) -> None:
    validate_train_url(str(response.url))
    history = getattr(response, "history", [])
    if history:
        raise StarcopAuditError("Redirected train-manifest responses are forbidden")
    if int(getattr(response, "status_code", 0)) != 200:
        raise StarcopAuditError("Train-manifest response must have HTTP status 200")
    content_type = str(response.headers.get("Content-Type", "")).lower()
    if "html" in content_type:
        raise StarcopAuditError("HTML/auth response rejected")
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise StarcopAuditError("Invalid Content-Length") from exc
        if declared < 0 or declared > MAX_MANIFEST_BYTES:
            raise StarcopAuditError("Content-Length exceeds the 2 MiB cap")


def stream_manifest_payload(response: Any) -> bytes:
    validate_manifest_response(response)
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_MANIFEST_BYTES:
            raise StarcopAuditError("Streamed train manifest exceeds the 2 MiB cap")
        chunks.append(chunk)
    payload = b"".join(chunks)
    if _looks_like_html(payload):
        raise StarcopAuditError("HTML/auth payload rejected")
    if len(payload) != TRAIN_MANIFEST_BYTES:
        raise StarcopAuditError("Frozen train-manifest byte count mismatch")
    if md5_bytes(payload) != TRAIN_MANIFEST_MD5:
        raise StarcopAuditError("Frozen train-manifest MD5 mismatch")
    return payload


def acquire_train_manifest(session: Any | None = None) -> Path:
    validate_train_url(TRAIN_MANIFEST_URL)
    client = session if session is not None else requests.Session()
    response = client.get(
        TRAIN_MANIFEST_URL,
        stream=True,
        allow_redirects=False,
        timeout=(15, 60),
    )
    response.raise_for_status()
    payload = stream_manifest_payload(response)
    IGNORED_ROOT.mkdir(parents=True, exist_ok=True)
    partial = CACHED_TRAIN_MANIFEST.with_name(TRAIN_MANIFEST_NAME + ".part")
    try:
        partial.write_bytes(payload)
        if partial.stat().st_size != TRAIN_MANIFEST_BYTES:
            raise StarcopAuditError("Atomic cache byte-count verification failed")
        if md5_bytes(partial.read_bytes()) != TRAIN_MANIFEST_MD5:
            raise StarcopAuditError("Atomic cache MD5 verification failed")
        partial.replace(CACHED_TRAIN_MANIFEST)
    finally:
        if partial.exists():
            partial.unlink()
    return CACHED_TRAIN_MANIFEST


def parse_strict_bool(value: object, *, field: str = "has_plume") -> bool:
    if not isinstance(value, str):
        raise StarcopAuditError(f"{field} must be a strict CSV boolean")
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise StarcopAuditError(f"{field} must be exactly true or false")


def parse_nonnegative_int(value: object, *, field: str) -> int:
    if not isinstance(value, str) or INTEGER_RE.fullmatch(value.strip()) is None:
        raise StarcopAuditError(f"{field} must be a nonnegative base-10 integer")
    return int(value.strip())


def _parse_timestamp(date: str, time: str, *, sample_id: str) -> str:
    try:
        parsed = datetime.strptime(date + time, "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise StarcopAuditError(f"Invalid UTC in STARCOP ID: {sample_id}") from exc
    return parsed.isoformat().replace("+00:00", "Z")


def parse_manifest_payload(payload: bytes) -> list[ManifestRow]:
    if _looks_like_html(payload):
        raise StarcopAuditError("HTML/auth payload rejected")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise StarcopAuditError("train.csv is not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = reader.fieldnames
    if fields is None:
        raise StarcopAuditError("train.csv has no header")
    if len(fields) != len(set(fields)):
        raise StarcopAuditError("train.csv contains duplicate columns")
    missing = REQUIRED_COLUMNS.difference(fields)
    if missing:
        raise StarcopAuditError(f"train.csv is missing columns: {sorted(missing)}")

    rows: list[ManifestRow] = []
    seen_ids: set[str] = set()
    for line_number, raw in enumerate(reader, start=2):
        if None in raw:
            raise StarcopAuditError(f"Malformed CSV row at line {line_number}")
        sample_id = str(raw["id"]).strip()
        if sample_id in seen_ids:
            raise StarcopAuditError(f"Duplicate STARCOP ID: {sample_id}")
        match = ID_RE.fullmatch(sample_id)
        if match is None:
            raise StarcopAuditError(f"Invalid STARCOP ID at line {line_number}")
        seen_ids.add(sample_id)

        local_row_offset = parse_nonnegative_int(
            raw["window_row_off"], field="window_row_off"
        )
        local_column_offset = parse_nonnegative_int(
            raw["window_col_off"], field="window_col_off"
        )
        local_width = parse_nonnegative_int(raw["window_width"], field="window_width")
        local_height = parse_nonnegative_int(
            raw["window_height"], field="window_height"
        )
        if (local_row_offset, local_column_offset, local_width, local_height) != (
            0,
            0,
            512,
            512,
        ):
            raise StarcopAuditError(
                f"Cached chip-local window is not 0,0,512,512: {sample_id}"
            )
        source_row_offset = int(match.group("row"))
        source_column_offset = int(match.group("column"))
        source_width = int(match.group("width"))
        source_height = int(match.group("height"))
        if source_width < 1 or source_height < 1:
            raise StarcopAuditError(f"STARCOP source window is empty: {sample_id}")
        has_plume = parse_strict_bool(raw["has_plume"])
        rows.append(
            ManifestRow(
                sample_id=sample_id,
                flight=match.group("flight"),
                timestamp=_parse_timestamp(
                    match.group("date"), match.group("time"), sample_id=sample_id
                ),
                source_row_offset=source_row_offset,
                source_column_offset=source_column_offset,
                source_width=source_width,
                source_height=source_height,
                has_plume=has_plume,
            )
        )
    if not rows:
        raise StarcopAuditError("train.csv contains no data rows")
    return rows


def read_train_manifest(
    path: Path,
    *,
    expected_bytes: int = TRAIN_MANIFEST_BYTES,
    expected_md5: str = TRAIN_MANIFEST_MD5,
) -> list[ManifestRow]:
    validate_train_filename(path)
    payload = path.read_bytes()
    if len(payload) != expected_bytes:
        raise StarcopAuditError("Frozen train-manifest byte count mismatch")
    if md5_bytes(payload) != expected_md5:
        raise StarcopAuditError("Frozen train-manifest MD5 mismatch")
    return parse_manifest_payload(payload)


def select_negative_rows(
    rows: Sequence[ManifestRow],
    *,
    maximum_per_flightline: int = MAXIMUM_SELECTED_PER_FLIGHTLINE,
) -> list[ManifestRow]:
    if maximum_per_flightline < 1:
        raise ValueError("maximum_per_flightline must be positive")
    by_flight: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        if not row.has_plume:
            by_flight[row.flight].append(row)
    selected: list[ManifestRow] = []
    for flight in sorted(by_flight):
        ranked = sorted(
            by_flight[flight], key=lambda row: (row.selection_digest, row.sample_id)
        )
        selected.extend(ranked[:maximum_per_flightline])
    return selected


def selected_record(row: ManifestRow) -> dict[str, object]:
    return {
        "sample_id": row.sample_id,
        "sensor": "AVIRIS-NG",
        "tile": row.flight,
        "timestamp": row.timestamp,
        "published_split": "train",
        "label_state": "NO_PLUME",
        "has_plume": False,
        "source_row_offset": row.source_row_offset,
        "source_column_offset": row.source_column_offset,
        "source_width": row.source_width,
        "source_height": row.source_height,
        "label_member": f"{row.sample_id}/labelbinary.tif",
        "selection_sha256": row.selection_digest,
        "coordinate_resolved": False,
        "eligible_for_target_catalog": False,
        "group_id": None,
        "novel_beyond_all_mars_25km": None,
    }


def audit_rows(rows: Sequence[ManifestRow]) -> tuple[list[ManifestRow], dict[str, object]]:
    negative = [row for row in rows if not row.has_plume]
    negative_flights = {row.flight for row in negative}
    selected = select_negative_rows(rows)
    gates = {
        "minimum_negative_rows": {
            "required": MINIMUM_NEGATIVE_ROWS,
            "observed": len(negative),
            "passed": len(negative) >= MINIMUM_NEGATIVE_ROWS,
        },
        "minimum_negative_bearing_flightlines": {
            "required": MINIMUM_NEGATIVE_FLIGHTLINES,
            "observed": len(negative_flights),
            "passed": len(negative_flights) >= MINIMUM_NEGATIVE_FLIGHTLINES,
        },
        "maximum_selected_rows_per_flightline": {
            "required_maximum": MAXIMUM_SELECTED_PER_FLIGHTLINE,
            "observed_maximum": max(
                (
                    sum(selected_row.flight == flight for selected_row in selected)
                    for flight in negative_flights
                ),
                default=0,
            ),
            "passed": all(
                sum(selected_row.flight == flight for selected_row in selected)
                <= MAXIMUM_SELECTED_PER_FLIGHTLINE
                for flight in negative_flights
            ),
        },
    }
    return selected, {
        "decision": "PASS" if all(gate["passed"] for gate in gates.values()) else "FAIL",
        "counts": {
            "released_train_rows": len(rows),
            "positive_rows": len(rows) - len(negative),
            "negative_rows": len(negative),
            "all_flightlines": len({row.flight for row in rows}),
            "negative_bearing_flightlines": len(negative_flights),
            "selected_negative_rows": len(selected),
            "selected_negative_flightlines": len({row.flight for row in selected}),
        },
        "gates": gates,
    }


def execute_stage_a(session: Any | None = None) -> dict[str, object]:
    protocol = load_protocol()
    local_receipts = validate_frozen_local_inputs(protocol)
    manifest_path = acquire_train_manifest(session=session)
    rows = read_train_manifest(manifest_path)
    selected, result = audit_rows(rows)
    records = [selected_record(row) for row in selected]
    write_jsonl(SELECTED_NEGATIVES_JSONL, records)
    selected_receipt = {
        "path": SELECTED_NEGATIVES_JSONL.relative_to(ROOT).as_posix(),
        "bytes": SELECTED_NEGATIVES_JSONL.stat().st_size,
        "sha256": sha256_file(SELECTED_NEGATIVES_JSONL),
        "rows": len(records),
    }
    report = {
        "schema_version": 1,
        "scope": "STARCOP released train.csv Stage-A metadata audit only",
        **result,
        "protocol": {
            "path": EXPECTED_PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256": EXPECTED_PROTOCOL_SHA256,
        },
        "frozen_local_inputs": local_receipts,
        "source_manifest": {
            "url": TRAIN_MANIFEST_URL,
            "cache_path": manifest_path.relative_to(ROOT).as_posix(),
            "bytes": manifest_path.stat().st_size,
            "md5": md5_bytes(manifest_path.read_bytes()),
            "sha256": sha256_file(manifest_path),
        },
        "selected_rows": selected_receipt,
        "selection": {
            "ranking": "(sha256(id UTF-8), id)",
            "qplume_used_for_filtering_or_ranking": False,
            "coordinates_known_during_selection": False,
            "source_offsets_parsed_from_id": True,
            "cached_chip_local_window_required": "row=0,column=0,width=512,height=512",
        },
        "security_boundary": {
            "network_executed": True,
            "train_manifest_opened": True,
            "test_manifest_accessed": False,
            "archive_or_zip_accessed": False,
            "label_or_imagery_accessed": False,
            "target_catalog_accessed": False,
            "eligible_for_target_catalog": False,
        },
        "next_action": (
            "Stage A passed; a separate Stage-B execution is still required."
            if result["decision"] == "PASS"
            else protocol["stage_a_train_manifest"]["failure_action"]
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(COMPACT_REPORT, report)
    return report


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stage_b_archive_specs(protocol: dict[str, Any]) -> list[ArchiveSpec]:
    raw_archives = protocol["frozen_source_files"]["train_archives"]
    if len(raw_archives) != 6:
        raise StarcopAuditError("Exactly six frozen STARCOP train archives are required")
    names = [str(item["name"]) for item in raw_archives]
    if len(names) != len(set(names)):
        raise StarcopAuditError("Frozen STARCOP train archive names are not unique")
    allowed_names = frozenset(names)
    specs: list[ArchiveSpec] = []
    for item in raw_archives:
        spec = ArchiveSpec(
            name=str(item["name"]),
            size=int(item["bytes"]),
            declared_md5=str(item["md5"]),
            url=archive_url(str(item["name"])),
        )
        try:
            validate_archive_spec(spec, allowed_names)
        except SparseZipError as exc:
            raise StarcopAuditError(str(exc)) from exc
        specs.append(spec)
    return specs


def authorize_stage_b(
    *,
    stage_a_report_path: Path = COMPACT_REPORT,
    manifest_path: Path = CACHED_TRAIN_MANIFEST,
    selected_path: Path = SELECTED_NEGATIVES_JSONL,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    """Validate the committed Stage-A PASS receipt and every ignored input."""
    protocol = load_protocol()
    validate_frozen_local_inputs(protocol)
    if stage_a_report_path.resolve() != COMPACT_REPORT.resolve():
        raise StarcopAuditError("Only the committed Stage-A report path is permitted")
    if sha256_file(stage_a_report_path) != EXPECTED_STAGE_A_REPORT_SHA256:
        raise StarcopAuditError("Committed Stage-A PASS receipt SHA-256 mismatch")
    report = json.loads(stage_a_report_path.read_text(encoding="utf-8"))
    if report.get("decision") != "PASS":
        raise StarcopAuditError("Stage B requires a committed Stage-A PASS")
    if report.get("protocol") != {
        "path": EXPECTED_PROTOCOL.relative_to(ROOT).as_posix(),
        "sha256": EXPECTED_PROTOCOL_SHA256,
    }:
        raise StarcopAuditError("Stage-A receipt protocol binding mismatch")
    source_receipt = report.get("source_manifest", {})
    if manifest_path.resolve() != CACHED_TRAIN_MANIFEST.resolve():
        raise StarcopAuditError("Only the ignored hash-bound train.csv is permitted")
    if source_receipt.get("cache_path") != manifest_path.relative_to(ROOT).as_posix():
        raise StarcopAuditError("Stage-A receipt train-manifest path mismatch")
    if manifest_path.stat().st_size != TRAIN_MANIFEST_BYTES:
        raise StarcopAuditError("Stage-B train-manifest byte count mismatch")
    payload = manifest_path.read_bytes()
    if md5_bytes(payload) != TRAIN_MANIFEST_MD5:
        raise StarcopAuditError("Stage-B train-manifest MD5 mismatch")
    if source_receipt.get("sha256") != sha256_bytes(payload):
        raise StarcopAuditError("Stage-A receipt train-manifest SHA-256 mismatch")
    if source_receipt.get("bytes") != len(payload) or source_receipt.get("md5") != TRAIN_MANIFEST_MD5:
        raise StarcopAuditError("Stage-A receipt train-manifest identity mismatch")

    selected_receipt = report.get("selected_rows", {})
    if selected_path.resolve() != SELECTED_NEGATIVES_JSONL.resolve():
        raise StarcopAuditError("Only the ignored Stage-A selected catalog is permitted")
    if selected_receipt.get("path") != selected_path.relative_to(ROOT).as_posix():
        raise StarcopAuditError("Stage-A selected-catalog path mismatch")
    if selected_path.stat().st_size != int(selected_receipt.get("bytes", -1)):
        raise StarcopAuditError("Stage-A selected-catalog byte count mismatch")
    if sha256_file(selected_path) != selected_receipt.get("sha256"):
        raise StarcopAuditError("Stage-A selected-catalog SHA-256 mismatch")
    selected_records = read_jsonl(selected_path)
    if len(selected_records) != int(selected_receipt.get("rows", -1)):
        raise StarcopAuditError("Stage-A selected-catalog row count mismatch")

    manifest_rows = read_train_manifest(manifest_path)
    recomputed = [selected_record(row) for row in select_negative_rows(manifest_rows)]
    if selected_records != recomputed:
        raise StarcopAuditError("Stage-A selected catalog is not the frozen recomputation")
    identifiers = [str(row.get("sample_id", "")) for row in selected_records]
    if len(identifiers) != len(set(identifiers)):
        raise StarcopAuditError("Stage-A selected catalog contains duplicate IDs")
    for row in selected_records:
        sample_id = str(row["sample_id"])
        if row.get("label_member") != f"{sample_id}/labelbinary.tif":
            raise StarcopAuditError("Stage-A selected label-member contract mismatch")
        if row.get("published_split") != "train" or row.get("has_plume") is not False:
            raise StarcopAuditError("Stage-A selected catalog contains a non-train negative")
        if row.get("eligible_for_target_catalog") is not False:
            raise StarcopAuditError("Stage-A selected catalog prematurely authorizes targets")
    return protocol, selected_records


def decode_zero_mask(
    payload: bytes,
    *,
    sample_id: str,
    archive_name: str,
    label_member: str,
) -> dict[str, object]:
    if not payload or len(payload) > MAXIMUM_UNCOMPRESSED_LABEL_BYTES:
        raise StarcopAuditError("Selected label TIFF exceeds the frozen size cap")
    try:
        with MemoryFile(payload) as memory_file, memory_file.open() as dataset:
            if dataset.count != 1 or dataset.width != 512 or dataset.height != 512:
                raise StarcopAuditError("Selected label TIFF must be exactly 1x512x512")
            if dataset.crs is None or not dataset.crs.is_projected:
                raise StarcopAuditError("Selected label TIFF requires a projected CRS")
            transform = dataset.transform
            coefficients = (
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            )
            determinant = transform.a * transform.e - transform.b * transform.d
            if not all(math.isfinite(value) for value in coefficients) or determinant == 0:
                raise StarcopAuditError("Selected label TIFF has an invalid affine transform")
            data = dataset.read(1, masked=False)
            if not np.isfinite(data).all() or np.any(data != 0):
                raise StarcopAuditError("Selected train-negative label TIFF is not all zero")
            center_x, center_y = transform * (dataset.width / 2.0, dataset.height / 2.0)
            transformer = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
            longitude, latitude = transformer.transform(center_x, center_y, errcheck=True)
            crs = dataset.crs.to_string()
    except StarcopAuditError:
        raise
    except Exception as exc:
        raise StarcopAuditError("Selected label bytes are not a valid georeferenced TIFF") from exc
    if not (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90.0 <= latitude <= 90.0
        and -180.0 <= longitude <= 180.0
    ):
        raise StarcopAuditError("Selected label center is not finite WGS84")
    return {
        "sample_id": sample_id,
        "sensor": "AVIRIS-NG",
        "tile": sample_id.split("_r", 1)[0],
        "published_split": "train",
        "label_state": "NO_PLUME",
        "has_plume": False,
        "label_member": label_member,
        "label_archive": archive_name,
        "label_tiff_bytes": len(payload),
        "label_tiff_sha256": sha256_bytes(payload),
        "zero_mask_confirmed": True,
        "raster_width": 512,
        "raster_height": 512,
        "raster_band_count": 1,
        "raster_crs": crs,
        "raster_affine": list(coefficients),
        "longitude": longitude,
        "latitude": latitude,
        "coordinate_resolved": True,
        "eligible_for_target_catalog": False,
        "group_id": None,
        "novel_beyond_all_mars_25km": None,
    }


def filter_stage_b_rows(
    *,
    rows: list[dict[str, object]],
    all_mars_locations: dict[str, tuple[float, float]],
    protected_mars_locations: dict[str, tuple[float, float]],
    prior_negative_coordinates: dict[str, tuple[float, float]],
) -> list[dict[str, object]]:
    spatial = filter_rows(
        rows=rows,
        all_mars_locations=all_mars_locations,
        protected_mars_locations=protected_mars_locations,
        prior_negative_coordinates=prior_negative_coordinates,
        radius_km=EXCLUSION_RADIUS_KM,
    )
    for row in spatial:
        passed = bool(row["eligible_for_target_catalog"])
        row["passes_frozen_spatial_filter"] = passed
        row["eligible_for_target_catalog"] = False
        if passed:
            row["eligibility_status"] = (
                "passes_frozen_spatial_filter_target_catalog_not_authorized"
            )
    return spatial


def _write_stage_b_markdown(report: dict[str, object]) -> None:
    counts = report["counts"]
    gates = report["gates"]
    lines = [
        "# STARCOP negative supplement — Stage B",
        "",
        f"**Decision:** {report['decision']}",
        "",
        (
            "This audit used sparse HTTP byte ranges for deterministic train-negative "
            "`labelbinary.tif` members only. It did not download a full archive, inspect "
            "the test split, or query a Sentinel-2/Landsat catalog."
        ),
        "",
        "## Counts",
        "",
        f"- Resolved zero-mask rows: {counts['resolved_selected_negative_rows']}",
        f"- Resolved flightlines: {counts['resolved_negative_flightlines']}",
        f"- Rows passing the spatial filter: {counts['spatially_eligible_rows']}",
        f"- Eligible 25 km components: {counts['eligible_25km_connected_components']}",
        "",
        "## Frozen gates",
        "",
    ]
    for name, gate in gates.items():
        lines.append(f"- `{name}`: {'PASS' if gate['passed'] else 'FAIL'}")
    lines.extend(
        [
            "",
            (
                "Even on PASS, every detailed row keeps "
                "`eligible_for_target_catalog=false`; a separately committed protocol "
                "is required before any target-satellite query."
            ),
            "",
        ]
    )
    STAGE_B_MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    partial = STAGE_B_MARKDOWN.with_name(STAGE_B_MARKDOWN.name + ".part")
    partial.write_text("\n".join(lines), encoding="utf-8")
    partial.replace(STAGE_B_MARKDOWN)


def execute_stage_b_sparse_masks(session: Any | None = None) -> dict[str, object]:
    protocol, selected_records = authorize_stage_b()
    specs = stage_b_archive_specs(protocol)
    client = session if session is not None else requests.Session()
    budget = RangeBudget(
        central_limit=MAXIMUM_CENTRAL_DIRECTORY_BYTES,
        label_limit=MAXIMUM_DOWNLOADED_LABEL_BYTES,
    )
    readers = {
        spec.name: RangeReader(spec=spec, session=client, budget=budget)
        for spec in specs
    }
    try:
        directories = {
            name: read_zip_directory(reader) for name, reader in readers.items()
        }
        selected_ids = [str(row["sample_id"]) for row in selected_records]
        member_locations = resolve_selected_members(directories, selected_ids)
        resolved: list[dict[str, object]] = []
        for selected in selected_records:
            sample_id = str(selected["sample_id"])
            archive_name, entry = member_locations[sample_id]
            if entry.uncompressed_size > MAXIMUM_UNCOMPRESSED_LABEL_BYTES:
                raise StarcopAuditError(
                    f"Selected label exceeds the frozen uncompressed cap: {sample_id}"
                )
            payload = fetch_label_member(
                readers[archive_name],
                entry,
                sample_id=sample_id,
                maximum_uncompressed_bytes=MAXIMUM_UNCOMPRESSED_LABEL_BYTES,
            )
            row = dict(selected)
            row.update(
                decode_zero_mask(
                    payload,
                    sample_id=sample_id,
                    archive_name=archive_name,
                    label_member=entry.name,
                )
            )
            row["eligible_for_target_catalog"] = False
            resolved.append(row)
    except SparseZipError as exc:
        raise StarcopAuditError(str(exc)) from exc

    write_jsonl(RESOLVED_STAGE_B_JSONL, resolved)
    frozen = protocol["frozen_local_inputs"]
    safe_manifest = ROOT / frozen["safe_mars_manifest"]["path"]
    prior_report = ROOT / frozen["mars_hyperspectral_stage_b_report"]["path"]
    prior_pairs = ROOT / frozen["mars_hyperspectral_pairs"]["path"]
    prior_masks = ROOT / frozen["mars_hyperspectral_mask_catalog"]["path"]
    if SAFE_MARS_COLUMNS & FORBIDDEN_MARS_COLUMNS:
        raise AssertionError("Safe and protected MARS columns overlap")
    mars = read_mars_observations(safe_manifest)
    all_mars, protected_mars = official_test_locations(mars)
    prior_coordinates, prior_counts = load_prior_negative_coordinates(
        stage_b_report_path=prior_report,
        pair_catalog_path=prior_pairs,
        mask_catalog_path=prior_masks,
    )
    filtered = filter_stage_b_rows(
        rows=resolved,
        all_mars_locations=all_mars,
        protected_mars_locations=protected_mars,
        prior_negative_coordinates=prior_coordinates,
    )
    write_jsonl(FILTERED_STAGE_B_JSONL, filtered)
    passing = [row for row in filtered if row["passes_frozen_spatial_filter"]]
    groups = {str(row["group_id"]) for row in passing}
    novel_groups = {
        str(row["group_id"])
        for row in passing
        if row["component_novel_beyond_all_mars_25km"]
    }
    receipts = [receipt for reader in readers.values() for receipt in reader.receipts]
    write_jsonl(RANGE_RECEIPTS_JSONL, receipts)

    gates = {
        "minimum_resolved_selected_negative_rows": {
            "required": STAGE_B_MINIMUM_RESOLVED_ROWS,
            "observed": len(resolved),
            "passed": len(resolved) >= STAGE_B_MINIMUM_RESOLVED_ROWS,
        },
        "minimum_resolved_negative_flightlines": {
            "required": STAGE_B_MINIMUM_RESOLVED_FLIGHTLINES,
            "observed": len({str(row["tile"]) for row in resolved}),
            "passed": len({str(row["tile"]) for row in resolved})
            >= STAGE_B_MINIMUM_RESOLVED_FLIGHTLINES,
        },
        "minimum_eligible_25km_connected_components": {
            "required": STAGE_B_MINIMUM_ELIGIBLE_COMPONENTS,
            "observed": len(groups),
            "passed": len(groups) >= STAGE_B_MINIMUM_ELIGIBLE_COMPONENTS,
        },
        "all_retained_labels_and_coordinates_valid": {
            "required": True,
            "observed": True,
            "passed": True,
        },
    }
    decision = "PASS" if all(gate["passed"] for gate in gates.values()) else "FAIL"
    archive_receipts = []
    for spec in specs:
        reader = readers[spec.name]
        directory = directories[spec.name]
        archive_receipts.append(
            {
                "name": spec.name,
                "bytes": spec.size,
                "record_declared_md5": spec.declared_md5,
                "archive_md5_verified": False,
                "url": spec.url,
                "zip64": directory.zip64,
                "central_directory_offset": directory.central_offset,
                "central_directory_bytes": directory.central_size,
                "central_directory_entries": len(directory.entries),
                "range_requests": len(reader.receipts),
                "range_bytes": sum(int(item["bytes"]) for item in reader.receipts),
            }
        )
    counts = {
        "stage_a_selected_negative_rows": len(selected_records),
        "resolved_selected_negative_rows": len(resolved),
        "resolved_negative_flightlines": len({str(row["tile"]) for row in resolved}),
        "mars_test_protected_rows": sum(bool(row["mars_test_protected"]) for row in filtered),
        "prior_pair_duplicate_rows": sum(bool(row["prior_pair_duplicate_25km"]) for row in filtered),
        "spatially_eligible_rows": len(passing),
        "spatially_eligible_flightlines": len({str(row["tile"]) for row in passing}),
        "eligible_25km_connected_components": len(groups),
        "components_novel_beyond_all_mars_25km": len(novel_groups),
        "all_mars_representative_locations": len(all_mars),
        "official_test_representative_locations": len(protected_mars),
        **prior_counts,
    }
    report = {
        "schema_version": 1,
        "scope": "STARCOP train-negative sparse zero-mask georeferencing only",
        "decision": decision,
        "protocol": {
            "path": EXPECTED_PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256": EXPECTED_PROTOCOL_SHA256,
        },
        "stage_a_authorization": {
            "path": COMPACT_REPORT.relative_to(ROOT).as_posix(),
            "sha256": EXPECTED_STAGE_A_REPORT_SHA256,
            "decision": "PASS",
            "manifest_md5": TRAIN_MANIFEST_MD5,
            "selected_rows_sha256": sha256_file(SELECTED_NEGATIVES_JSONL),
            "selected_rows": len(selected_records),
        },
        "archives": archive_receipts,
        "byte_range_totals": {
            "requests": len(receipts),
            "central_directory_bytes": budget.central_bytes,
            "central_directory_cap": budget.central_limit,
            "downloaded_label_bytes": budget.label_bytes,
            "downloaded_label_cap": budget.label_limit,
            "full_archive_downloaded": False,
            "archive_md5_verification_claimed": False,
        },
        "counts": counts,
        "gates": gates,
        "distance_summaries_km": {
            "nearest_official_mars_test": numeric_summary(
                float(row["nearest_mars_test_km"]) for row in filtered
            ),
            "nearest_prior_negative_source": numeric_summary(
                float(row["nearest_prior_negative_pair_km"]) for row in filtered
            ),
            "nearest_any_mars_location": numeric_summary(
                float(row["nearest_any_mars_km"]) for row in filtered
            ),
        },
        "ignored_artifacts": {
            "resolved_rows": {
                "path": RESOLVED_STAGE_B_JSONL.relative_to(ROOT).as_posix(),
                "rows": len(resolved),
                "sha256": sha256_file(RESOLVED_STAGE_B_JSONL),
            },
            "filtered_rows": {
                "path": FILTERED_STAGE_B_JSONL.relative_to(ROOT).as_posix(),
                "rows": len(filtered),
                "sha256": sha256_file(FILTERED_STAGE_B_JSONL),
            },
            "range_receipts": {
                "path": RANGE_RECEIPTS_JSONL.relative_to(ROOT).as_posix(),
                "rows": len(receipts),
                "sha256": sha256_file(RANGE_RECEIPTS_JSONL),
            },
        },
        "security_boundary": {
            "network_executed": True,
            "range_requests_only": True,
            "test_manifest_or_archive_accessed": False,
            "positive_or_nonlabel_member_downloaded": False,
            "full_archive_downloaded": False,
            "target_catalog_accessed": False,
            "eligible_for_target_catalog": False,
            "protected_mars_outcome_columns_accessed": [],
        },
        "next_action": (
            "Commit a separate target-catalog protocol before any query."
            if decision == "PASS"
            else protocol["stage_b_sparse_zero_mask_georeferencing"]["failure_action"]
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(STAGE_B_REPORT, report)
    _write_stage_b_markdown(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--execute-train-manifest",
        action="store_true",
        help="Explicitly retrieve and audit only the frozen Zenodo train.csv.",
    )
    execution.add_argument(
        "--execute-stage-b-sparse-masks",
        action="store_true",
        help=(
            "After Stage-A authorization, range-read only deterministic train-negative "
            "labelbinary.tif members and run the frozen spatial filter."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute_train_manifest:
        result = execute_stage_a()
    elif args.execute_stage_b_sparse_masks:
        result = execute_stage_b_sparse_masks()
    else:
        result = validation_plan()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
