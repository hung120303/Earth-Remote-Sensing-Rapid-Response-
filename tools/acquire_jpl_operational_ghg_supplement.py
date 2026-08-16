"""Acquire only compact JPL operational-GHG metadata for the frozen audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


RECORD_ID = "19011045"
RECORD_API = f"https://zenodo.org/api/records/{RECORD_ID}"
ALLOWED_FILE = "multicampaign.tar"
EXPECTED_MD5 = "43ad3290fc12133ce387d6c0fce94d6d"
USER_AGENT = "ERSRR-research-audit/1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - source-published integrity identity
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_url(url: str, *, retries: int = 7) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(f"Failed to acquire {url}") from error
            retry_after = 0.0
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                try:
                    retry_after = float(error.headers.get("Retry-After", "0"))
                except (TypeError, ValueError):
                    retry_after = 0.0
            time.sleep(max(retry_after, min(60.0, 3.0 * (2**attempt))))
    raise AssertionError("unreachable")


def download_file(url: str, output: Path) -> None:
    payload = read_url(url)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    partial.write_bytes(payload)
    partial.replace(output)


def acquire(*, output_dir: Path, protocol_path: Path) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol["candidate_source"]["metadata_archive"]
    if expected["name"] != ALLOWED_FILE or expected["reported_md5"] != EXPECTED_MD5:
        raise ValueError("Frozen protocol metadata identity mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
        and path.name not in {"zenodo_record.json", ALLOWED_FILE, "manifest.json"}
    ]
    if unexpected:
        raise ValueError(f"Unexpected files in compact acquisition root: {unexpected}")

    record_path = output_dir / "zenodo_record.json"
    record_path.write_bytes(read_url(RECORD_API))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    files = [item for item in record.get("files", []) if item.get("key") == ALLOWED_FILE]
    if len(files) != 1:
        raise ValueError(f"Expected exactly one {ALLOWED_FILE} record")
    source_file = files[0]
    checksum = str(source_file.get("checksum", ""))
    if checksum != f"md5:{EXPECTED_MD5}":
        raise ValueError(f"Zenodo checksum changed: {checksum}")
    archive_path = output_dir / ALLOWED_FILE
    if not archive_path.exists() or md5_file(archive_path) != EXPECTED_MD5:
        links = source_file.get("links", {})
        url = links.get("content") or links.get("self")
        if not url:
            raise ValueError("Zenodo file metadata has no content link")
        download_file(str(url), archive_path)
    if md5_file(archive_path) != EXPECTED_MD5:
        raise ValueError("Downloaded metadata archive MD5 mismatch")

    license_data = record.get("metadata", {}).get("license")
    manifest = {
        "schema_version": 1,
        "scope": "zenodo_record_and_multicampaign_definitions_only",
        "record_id": RECORD_ID,
        "doi": record.get("doi"),
        "record_created": record.get("created"),
        "record_modified": record.get("updated"),
        "dataset_license_metadata": license_data,
        "protocol": {
            "path": protocol_path.as_posix(),
            "sha256": sha256_file(protocol_path),
        },
        "files": {
            "zenodo_record.json": {
                "bytes": record_path.stat().st_size,
                "sha256": sha256_file(record_path),
            },
            ALLOWED_FILE: {
                "bytes": archive_path.stat().st_size,
                "md5": md5_file(archive_path),
                "sha256": sha256_file(archive_path),
            },
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".research/jpl_operational_ghg_supplement"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/mars_cross_modal_negative_supplement_protocol.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = acquire(output_dir=args.output_dir, protocol_path=args.protocol)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
