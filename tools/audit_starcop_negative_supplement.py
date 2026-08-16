"""Audit the frozen STARCOP train manifest for negative-supplement feasibility.

The safe default is validation-only: it validates the committed protocol and
its frozen local inputs without making a network request or opening a STARCOP
manifest.  The explicit execution mode may retrieve only the exact, hash-bound
Zenodo ``train.csv`` and performs Stage A only.  There is deliberately no ZIP,
test-manifest, label-raster, or target-catalog implementation in this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
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
MINIMUM_NEGATIVE_ROWS = 1_000
MINIMUM_NEGATIVE_FLIGHTLINES = 50
MAXIMUM_SELECTED_PER_FLIGHTLINE = 4

ID_RE = re.compile(
    r"^(?P<flight>ang(?P<date>\d{8})t(?P<time>\d{6}))_"
    r"r(?P<row>\d+)_c(?P<column>\d+)_w(?P<width>\d+)_h(?P<height>\d+)$"
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
    width: int
    height: int
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
        "explicit_execution_flag": "--execute-train-manifest",
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
        width = int(match.group("width"))
        height = int(match.group("height"))
        if width != 512 or height != 512:
            raise StarcopAuditError(f"STARCOP chip is not 512x512: {sample_id}")
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
                width=width,
                height=height,
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
        "width": row.width,
        "height": row.height,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-train-manifest",
        action="store_true",
        help="Explicitly retrieve and audit only the frozen Zenodo train.csv.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = execute_stage_a() if args.execute_train_manifest else validation_plan()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
