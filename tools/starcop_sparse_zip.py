"""Strict byte-range ZIP/ZIP64 reader for the frozen STARCOP label allowlist.

This module has no command-line entry point and performs no request at import.
Callers must provide an exact frozen archive specification and an explicit
session.  Only byte ranges are accepted; full-file HTTP responses fail closed.
"""

from __future__ import annotations

import binascii
import hashlib
import re
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit

ZENODO_RECORD = "7863343"
ZENODO_ARCHIVE_URL_TEMPLATE = (
    "https://zenodo.org/api/records/7863343/files/{name}/content"
)
MAX_EOCD_SEARCH = 65_557
EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"
ZIP64_EXTRA_ID = 0x0001
SUPPORTED_METHODS = frozenset({0, 8})
RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RANGE_ATTEMPTS = 5
CONTENT_RANGE_RE = re.compile(r"^bytes (?P<start>\d+)-(?P<end>\d+)/(?P<total>\d+)$")


class SparseZipError(RuntimeError):
    """Raised when an archive or HTTP range violates the frozen contract."""


@dataclass(frozen=True)
class ArchiveSpec:
    name: str
    size: int
    declared_md5: str
    url: str


@dataclass(frozen=True)
class ZipEntry:
    name: str
    flags: int
    compression: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int
    disk_start: int


@dataclass(frozen=True)
class ZipDirectory:
    entries: tuple[ZipEntry, ...]
    central_offset: int
    central_size: int
    zip64: bool


@dataclass
class RangeBudget:
    central_limit: int
    label_limit: int
    minimum_interval_seconds: float = 0.0
    central_bytes: int = 0
    label_bytes: int = 0
    last_request_monotonic: float = 0.0

    def charge(self, category: str, amount: int) -> None:
        if amount < 0:
            raise SparseZipError("Negative byte-range charge")
        if category == "central_directory":
            if self.central_bytes + amount > self.central_limit:
                raise SparseZipError("Global central-directory byte cap exceeded")
            self.central_bytes += amount
        elif category == "label_member":
            if self.label_bytes + amount > self.label_limit:
                raise SparseZipError("Global downloaded-label byte cap exceeded")
            self.label_bytes += amount
        else:
            raise SparseZipError(f"Unknown byte-range category: {category}")

    def wait_for_request_slot(self) -> None:
        if self.minimum_interval_seconds < 0:
            raise SparseZipError("Negative HTTP pacing interval")
        remaining = (
            self.last_request_monotonic
            + self.minimum_interval_seconds
            - time.monotonic()
        )
        if remaining > 0:
            time.sleep(remaining)
        self.last_request_monotonic = time.monotonic()


@dataclass
class RangeReader:
    spec: ArchiveSpec
    session: Any
    budget: RangeBudget
    receipts: list[dict[str, object]] = field(default_factory=list)

    def read(self, start: int, end: int, *, category: str) -> bytes:
        if start < 0 or end < start or end >= self.spec.size:
            raise SparseZipError(
                f"Invalid range {start}-{end} for {self.spec.name} ({self.spec.size})"
            )
        expected_length = end - start + 1
        response: Any | None = None
        for attempt in range(MAX_RANGE_ATTEMPTS):
            self.budget.charge(category, expected_length)
            self.budget.wait_for_request_slot()
            response = self.session.get(
                self.spec.url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
                stream=True,
                allow_redirects=False,
                timeout=(15, 60),
            )
            # Charges are intentionally never rolled back. A failed or malicious
            # response consumed an attempt and cannot be retried around a global cap.
            try:
                payload = validate_range_response(
                    response,
                    spec=self.spec,
                    expected_start=start,
                    expected_end=end,
                )
                break
            except SparseZipError:
                status = int(getattr(response, "status_code", 0))
                close = getattr(response, "close", None)
                if callable(close):
                    close()
                if status not in RETRYABLE_HTTP_STATUSES or attempt + 1 >= MAX_RANGE_ATTEMPTS:
                    raise
                retry_after = str(response.headers.get("Retry-After", "")).strip()
                try:
                    delay = float(retry_after) if retry_after else 2.0**attempt
                except ValueError:
                    delay = 2.0**attempt
                time.sleep(min(max(delay, 0.25), 30.0))
        else:  # pragma: no cover - the bounded loop either succeeds or raises.
            raise SparseZipError("Exhausted range attempts")
        if response is None:  # pragma: no cover - defensive initialization guard.
            raise SparseZipError("Range response was not initialized")
        self.receipts.append(
            {
                "archive": self.spec.name,
                "category": category,
                "start": start,
                "end": end,
                "bytes": len(payload),
                "content_range": response.headers["Content-Range"],
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        return payload


def archive_url(name: str) -> str:
    return ZENODO_ARCHIVE_URL_TEMPLATE.format(name=quote(name, safe=""))


def validate_archive_spec(spec: ArchiveSpec, allowed_names: frozenset[str]) -> None:
    if spec.name not in allowed_names:
        raise SparseZipError("Archive is not one of the six frozen train archives")
    if spec.name == "STARCOP_test.zip" or not spec.name.startswith("STARCOP_train_"):
        raise SparseZipError("Test or non-train STARCOP archive is forbidden")
    if spec.size <= 0:
        raise SparseZipError("Frozen archive size must be positive")
    if re.fullmatch(r"[0-9a-f]{32}", spec.declared_md5) is None:
        raise SparseZipError("Frozen archive record MD5 is malformed")
    expected_url = archive_url(spec.name)
    parsed = urlsplit(spec.url)
    if (
        spec.url != expected_url
        or parsed.scheme != "https"
        or parsed.hostname != "zenodo.org"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SparseZipError("Archive URL differs from the exact frozen Zenodo URL")


def _looks_like_html(payload: bytes) -> bool:
    prefix = payload[:4096].lstrip().lower()
    return prefix.startswith((b"<!doctype html", b"<html")) or b"<form" in prefix


def validate_range_response(
    response: Any,
    *,
    spec: ArchiveSpec,
    expected_start: int,
    expected_end: int,
) -> bytes:
    if str(response.url) != spec.url:
        raise SparseZipError("Range response URL differs from the frozen archive URL")
    if getattr(response, "history", []):
        raise SparseZipError("Redirected range responses are forbidden")
    status = int(getattr(response, "status_code", 0))
    if status != 206:
        raise SparseZipError(
            f"Archive range response must have HTTP status 206; observed {status}"
        )
    content_type = str(response.headers.get("Content-Type", "")).lower()
    if "html" in content_type:
        raise SparseZipError("HTML/auth range response rejected")
    content_encoding = str(response.headers.get("Content-Encoding", "")).strip().lower()
    if content_encoding not in {"", "identity"}:
        raise SparseZipError("Encoded range responses are forbidden")
    raw_content_range = str(response.headers.get("Content-Range", ""))
    match = CONTENT_RANGE_RE.fullmatch(raw_content_range)
    if match is None:
        raise SparseZipError("Missing or malformed Content-Range")
    observed = tuple(int(match.group(key)) for key in ("start", "end", "total"))
    if observed != (expected_start, expected_end, spec.size):
        raise SparseZipError("Content-Range differs from the requested frozen range")
    expected_length = expected_end - expected_start + 1
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) != expected_length:
                raise SparseZipError("Range Content-Length mismatch")
        except ValueError as exc:
            raise SparseZipError("Invalid range Content-Length") from exc
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > expected_length:
            raise SparseZipError("Range response exceeded requested byte count")
        chunks.append(chunk)
    payload = b"".join(chunks)
    if len(payload) != expected_length:
        raise SparseZipError("Range response was shorter than requested")
    if _looks_like_html(payload):
        raise SparseZipError("HTML/auth range payload rejected")
    return payload


def _extra_fields(extra: bytes) -> dict[int, bytes]:
    result: dict[int, bytes] = {}
    cursor = 0
    while cursor < len(extra):
        if cursor + 4 > len(extra):
            raise SparseZipError("Truncated ZIP extra-field header")
        field_id, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        if cursor + size > len(extra):
            raise SparseZipError("Truncated ZIP extra-field payload")
        if field_id in result:
            raise SparseZipError("Duplicate ZIP extra field")
        result[field_id] = extra[cursor : cursor + size]
        cursor += size
    return result


def _zip64_values(
    extra: bytes,
    *,
    uncompressed: int,
    compressed: int,
    local_offset: int,
    disk_start: int,
) -> tuple[int, int, int, int]:
    needs = (
        uncompressed == 0xFFFFFFFF,
        compressed == 0xFFFFFFFF,
        local_offset == 0xFFFFFFFF,
        disk_start == 0xFFFF,
    )
    if not any(needs):
        return uncompressed, compressed, local_offset, disk_start
    payload = _extra_fields(extra).get(ZIP64_EXTRA_ID)
    if payload is None:
        raise SparseZipError("ZIP64 sentinel lacks a ZIP64 extra field")
    cursor = 0
    values = [uncompressed, compressed, local_offset, disk_start]
    widths = [8, 8, 8, 4]
    for index, needed in enumerate(needs):
        if not needed:
            continue
        width = widths[index]
        if cursor + width > len(payload):
            raise SparseZipError("Truncated ZIP64 extra field")
        fmt = "<Q" if width == 8 else "<L"
        values[index] = struct.unpack_from(fmt, payload, cursor)[0]
        cursor += width
    return tuple(values)  # type: ignore[return-value]


def _decode_name(raw: bytes, flags: int) -> str:
    encoding = "utf-8" if flags & 0x0800 else "cp437"
    try:
        name = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise SparseZipError("Invalid ZIP member filename encoding") from exc
    if "\x00" in name or "\\" in name:
        raise SparseZipError("Unsafe ZIP member filename")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SparseZipError("ZIP member path traversal rejected")
    return name


def parse_central_directory(payload: bytes, expected_entries: int) -> tuple[ZipEntry, ...]:
    entries: list[ZipEntry] = []
    seen_names: set[str] = set()
    cursor = 0
    while cursor < len(payload):
        if cursor + 46 > len(payload) or payload[cursor : cursor + 4] != CENTRAL_SIGNATURE:
            raise SparseZipError("Invalid or unsupported central-directory record")
        values = struct.unpack_from("<4s6H3L5H2L", payload, cursor)
        (
            _signature,
            _version_made,
            _version_needed,
            flags,
            compression,
            _mtime,
            _mdate,
            crc32_value,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            comment_length,
            disk_start,
            _internal,
            _external,
            local_offset,
        ) = values
        end = cursor + 46 + name_length + extra_length + comment_length
        if end > len(payload):
            raise SparseZipError("Truncated central-directory entry")
        raw_name = payload[cursor + 46 : cursor + 46 + name_length]
        extra_start = cursor + 46 + name_length
        extra = payload[extra_start : extra_start + extra_length]
        name = _decode_name(raw_name, flags)
        if name in seen_names:
            raise SparseZipError(f"Duplicate ZIP member: {name}")
        seen_names.add(name)
        if flags & 0x0001 or flags & 0x0040:
            raise SparseZipError(f"Encrypted ZIP member rejected: {name}")
        if compression not in SUPPORTED_METHODS:
            raise SparseZipError(f"Unsupported ZIP compression method: {compression}")
        (
            uncompressed_size,
            compressed_size,
            local_offset,
            disk_start,
        ) = _zip64_values(
            extra,
            uncompressed=uncompressed_size,
            compressed=compressed_size,
            local_offset=local_offset,
            disk_start=disk_start,
        )
        if disk_start != 0:
            raise SparseZipError("Multi-disk ZIP entry rejected")
        entries.append(
            ZipEntry(
                name=name,
                flags=flags,
                compression=compression,
                crc32=crc32_value,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_offset,
                disk_start=disk_start,
            )
        )
        cursor = end
    if len(entries) != expected_entries:
        raise SparseZipError(
            f"Central-directory entry count mismatch: {len(entries)} != {expected_entries}"
        )
    return tuple(entries)


def read_zip_directory(reader: RangeReader) -> ZipDirectory:
    tail_size = min(reader.spec.size, MAX_EOCD_SEARCH)
    tail_start = reader.spec.size - tail_size
    tail = reader.read(
        tail_start, reader.spec.size - 1, category="central_directory"
    )
    relative_eocd = tail.rfind(EOCD_SIGNATURE)
    if relative_eocd < 0 or relative_eocd + 22 > len(tail):
        raise SparseZipError("ZIP end-of-central-directory record not found")
    eocd = struct.unpack_from("<4s4H2LH", tail, relative_eocd)
    (
        _signature,
        disk_number,
        central_disk,
        entries_on_disk,
        total_entries,
        central_size,
        central_offset,
        comment_length,
    ) = eocd
    if relative_eocd + 22 + comment_length != len(tail):
        raise SparseZipError("EOCD comment length or trailing bytes are inconsistent")
    if disk_number != 0 or central_disk != 0:
        raise SparseZipError("Multi-disk ZIP archive rejected")

    is_zip64 = (
        entries_on_disk == 0xFFFF
        or total_entries == 0xFFFF
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    )
    if is_zip64:
        locator_relative = relative_eocd - 20
        if locator_relative < 0 or tail[locator_relative : locator_relative + 4] != ZIP64_LOCATOR_SIGNATURE:
            raise SparseZipError("ZIP64 EOCD locator is missing")
        _locator_sig, locator_disk, zip64_offset, total_disks = struct.unpack_from(
            "<4sLQL", tail, locator_relative
        )
        if locator_disk != 0 or total_disks != 1:
            raise SparseZipError("Multi-disk ZIP64 archive rejected")
        fixed = reader.read(zip64_offset, zip64_offset + 55, category="central_directory")
        if fixed[:4] != ZIP64_EOCD_SIGNATURE:
            raise SparseZipError("ZIP64 EOCD record is missing")
        record_size = struct.unpack_from("<Q", fixed, 4)[0]
        if record_size < 44 or zip64_offset + 12 + record_size > reader.spec.size:
            raise SparseZipError("Invalid ZIP64 EOCD record size")
        if record_size > 44:
            reader.read(
                zip64_offset + 56,
                zip64_offset + 11 + record_size,
                category="central_directory",
            )
        (
            _sig,
            _size,
            _made,
            _needed,
            zip64_disk,
            zip64_central_disk,
            entries_on_disk_64,
            total_entries_64,
            central_size_64,
            central_offset_64,
        ) = struct.unpack_from("<4sQ2H2L4Q", fixed, 0)
        if zip64_disk != 0 or zip64_central_disk != 0 or entries_on_disk_64 != total_entries_64:
            raise SparseZipError("Multi-disk or inconsistent ZIP64 counts rejected")
        total_entries = total_entries_64
        central_size = central_size_64
        central_offset = central_offset_64
    elif entries_on_disk != total_entries:
        raise SparseZipError("Multi-disk or inconsistent ZIP entry counts rejected")

    if total_entries <= 0 or central_size <= 0:
        raise SparseZipError("ZIP central directory is empty")
    if central_offset + central_size > reader.spec.size:
        raise SparseZipError("ZIP central directory exceeds archive bounds")
    central = reader.read(
        central_offset,
        central_offset + central_size - 1,
        category="central_directory",
    )
    entries = parse_central_directory(central, int(total_entries))
    return ZipDirectory(
        entries=entries,
        central_offset=central_offset,
        central_size=central_size,
        zip64=is_zip64,
    )


def resolve_selected_members(
    directories: dict[str, ZipDirectory], selected_ids: list[str]
) -> dict[str, tuple[str, ZipEntry]]:
    if len(selected_ids) != len(set(selected_ids)):
        raise SparseZipError("Selected STARCOP IDs are not unique")
    matches: dict[str, list[tuple[str, ZipEntry]]] = {
        sample_id: [] for sample_id in selected_ids
    }
    for archive_name, directory in directories.items():
        if not archive_name.endswith(".zip"):
            raise SparseZipError("Frozen STARCOP archive name lacks .zip")
        archive_root = archive_name.removesuffix(".zip")
        wanted = {
            f"{archive_root}/{sample_id}/labelbinary.tif": sample_id
            for sample_id in selected_ids
        }
        for entry in directory.entries:
            sample_id = wanted.get(entry.name)
            if sample_id is not None:
                matches[sample_id].append((archive_name, entry))
    result: dict[str, tuple[str, ZipEntry]] = {}
    for sample_id, member_matches in matches.items():
        if len(member_matches) != 1:
            raise SparseZipError(
                "Selected logical label member must occur exactly once below an exact "
                f"archive root: {sample_id}/labelbinary.tif ({len(member_matches)})"
            )
        result[sample_id] = member_matches[0]
    return result


def _validate_expected_member(
    sample_id: str, archive_name: str, member_name: str
) -> None:
    archive_root = archive_name.removesuffix(".zip")
    if archive_root == archive_name:
        raise SparseZipError("Frozen STARCOP archive name lacks .zip")
    if member_name != f"{archive_root}/{sample_id}/labelbinary.tif":
        raise SparseZipError("Only exact selected {id}/labelbinary.tif members are allowed")
    if "/" in sample_id or "\\" in sample_id or sample_id in {"", ".", ".."}:
        raise SparseZipError("Unsafe selected sample ID")


def fetch_label_member(
    reader: RangeReader,
    entry: ZipEntry,
    *,
    sample_id: str,
    maximum_uncompressed_bytes: int,
) -> bytes:
    _validate_expected_member(sample_id, reader.spec.name, entry.name)
    if (
        maximum_uncompressed_bytes < 1
        or entry.uncompressed_size > maximum_uncompressed_bytes
    ):
        raise SparseZipError("Selected label exceeds the uncompressed-byte cap")
    fixed = reader.read(
        entry.local_header_offset,
        entry.local_header_offset + 29,
        category="label_member",
    )
    if fixed[:4] != LOCAL_SIGNATURE:
        raise SparseZipError("Selected member local header is missing")
    (
        _signature,
        _version,
        flags,
        compression,
        _mtime,
        _mdate,
        crc32_value,
        compressed_size,
        uncompressed_size,
        name_length,
        extra_length,
    ) = struct.unpack_from("<4s5H3L2H", fixed, 0)
    if flags & 0x0001 or flags & 0x0040:
        raise SparseZipError("Encrypted local member rejected")
    variable_length = name_length + extra_length
    data_start = entry.local_header_offset + 30 + variable_length
    if entry.compressed_size <= 0 or data_start + entry.compressed_size > reader.spec.size:
        raise SparseZipError("Selected member data range is invalid")
    variable_and_compressed = reader.read(
        entry.local_header_offset + 30,
        data_start + entry.compressed_size - 1,
        category="label_member",
    )
    variable = variable_and_compressed[:variable_length]
    compressed = variable_and_compressed[variable_length:]
    raw_name = variable[:name_length]
    extra = variable[name_length:]
    name = _decode_name(raw_name, flags)
    if name != entry.name:
        raise SparseZipError("Local and central member names disagree")
    uses_data_descriptor = bool(flags & 0x0008)
    if flags != entry.flags or compression != entry.compression:
        raise SparseZipError("Local and central ZIP headers are inconsistent")
    if uses_data_descriptor:
        # With general-purpose bit 3 set, the local CRC and sizes are permitted
        # placeholders. The authoritative central-directory values bound the
        # byte range, while the reconstructed payload is still checked against
        # the authoritative size and CRC below.
        if crc32_value not in {0, entry.crc32}:
            raise SparseZipError("Data-descriptor local CRC placeholder is invalid")
        (
            uncompressed_size,
            compressed_size,
            _unused_offset,
            _unused_disk,
        ) = _zip64_values(
            extra,
            uncompressed=uncompressed_size,
            compressed=compressed_size,
            local_offset=0,
            disk_start=0,
        )
        for local_value, central_value, role in (
            (compressed_size, entry.compressed_size, "compressed size"),
            (uncompressed_size, entry.uncompressed_size, "uncompressed size"),
        ):
            if local_value not in {0, central_value}:
                raise SparseZipError(
                    f"Data-descriptor local {role} placeholder is invalid"
                )
    else:
        (
            uncompressed_size,
            compressed_size,
            _unused_offset,
            _unused_disk,
        ) = _zip64_values(
            extra,
            uncompressed=uncompressed_size,
            compressed=compressed_size,
            local_offset=0,
            disk_start=0,
        )
        if (
            crc32_value != entry.crc32
            or compressed_size != entry.compressed_size
            or uncompressed_size != entry.uncompressed_size
        ):
            raise SparseZipError("Local and central ZIP headers are inconsistent")
    if entry.compression == 0:
        payload = compressed
    elif entry.compression == 8:
        decompressor = zlib.decompressobj(-15)
        try:
            output_limit = entry.uncompressed_size + 1
            payload = decompressor.decompress(compressed, output_limit)
            if len(payload) > entry.uncompressed_size or decompressor.unconsumed_tail:
                raise SparseZipError("Selected label expands beyond its declared size")
            payload += decompressor.flush(output_limit - len(payload))
        except zlib.error as exc:
            raise SparseZipError("Selected label DEFLATE stream is invalid") from exc
        if len(payload) > entry.uncompressed_size:
            raise SparseZipError("Selected label expands beyond its declared size")
        if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
            raise SparseZipError("Selected label DEFLATE stream has trailing/incomplete data")
    else:  # Protected by central and local validation; retained as a fail-closed guard.
        raise SparseZipError("Unsupported selected-member compression")
    validate_label_payload(payload, entry)
    return payload


def validate_label_payload(payload: bytes, entry: ZipEntry) -> None:
    """Bind a reconstructed or cached selected mask to its central entry."""
    if len(payload) != entry.uncompressed_size:
        raise SparseZipError("Selected label uncompressed-size mismatch")
    if (binascii.crc32(payload) & 0xFFFFFFFF) != entry.crc32:
        raise SparseZipError("Selected label CRC-32 mismatch")
