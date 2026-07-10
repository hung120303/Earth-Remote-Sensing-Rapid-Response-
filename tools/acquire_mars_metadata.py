#!/usr/bin/env python3
"""Fetch and verify the pinned public MARS-S2L metadata tranche.

The full MARS-S2L repository is roughly 100 GB. This tool deliberately fetches
only the small metadata files needed to audit labels, sites, splits, and asset
contracts before authorizing the full image download. Raw files and the local
manifest remain beneath an ignored data-acquisition directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ID = "UNEP-IMEO/MARS-S2L"
REVISION = "c26b1d7e31a0c5241fa37c9140802622c215eb32"
DEFAULT_OUTPUT = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
    "publication-v1/external/MARS-S2L"
)
MANIFEST_NAME = "metadata_manifest.json"
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "ERSRR-research-metadata-fetcher/1.0"

# LFS object ids are SHA-256 digests. Non-LFS files are still size-checked and
# receive a locally computed SHA-256 in the generated manifest.
METADATA_FILES: tuple[dict[str, Any], ...] = (
    {"path": "README.md", "size": 11_414, "sha256": None},
    {"path": "location_name_mapping.json", "size": 26_398, "sha256": None},
    {
        "path": "test.csv",
        "size": 43_066_514,
        "sha256": "add125547e0e0066216070ed61a8544e76e84f062f636390be5d2ef1808dbfaa",
    },
    {
        "path": "train.csv",
        "size": 39_298_437,
        "sha256": "f163f3986d46875270014fb958c7a5931c5c661ec650ab26391b3e7bde7f9053",
    },
    {"path": "val.csv", "size": 5_934_748, "sha256": None},
    {
        "path": "validated_images_all.csv",
        "size": 94_136_263,
        "sha256": "799fa3272be6c313534c5d974894883db9f97874adb617eeaace1c8a4f9dc9b2",
    },
    {"path": "validated_images_plumes.csv", "size": 6_383_275, "sha256": None},
)


def repo_root() -> Path:
    output = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path.cwd(),
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
    if not output:
        raise RuntimeError("Could not resolve the repository root")
    return Path(output).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_url(relative_path: str) -> str:
    encoded = urllib.parse.quote(relative_path, safe="/")
    return (
        f"https://huggingface.co/datasets/{REPO_ID}/resolve/"
        f"{REVISION}/{encoded}?download=true"
    )


def checked_output_dir(root: Path, value: str) -> Path:
    acquisition_root = (
        root / "EarthRemoteSensingRapidResponse" / "Data Collection" / "s2_emit_pairs"
    ).resolve()
    output = (root / value).resolve()
    if acquisition_root not in output.parents:
        raise ValueError("output-dir must resolve beneath the ignored s2_emit_pairs directory")
    return output


def download_file(item: dict[str, Any], output_dir: Path, *, force: bool) -> dict[str, Any]:
    relative_path = str(item["path"])
    expected_size = int(item["size"])
    expected_hash = item["sha256"]
    destination = (output_dir / relative_path).resolve()
    if output_dir != destination and output_dir not in destination.parents:
        raise ValueError(f"Unsafe metadata path: {relative_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        observed_size = destination.stat().st_size
        observed_hash = sha256(destination)
        if observed_size == expected_size and (expected_hash is None or observed_hash == expected_hash):
            return {
                "path": relative_path,
                "source_url": source_url(relative_path),
                "size": observed_size,
                "sha256": observed_hash,
                "expected_sha256": expected_hash,
                "status": "reused_verified",
            }
        raise ValueError(
            f"Existing file failed verification: {relative_path}; use --force to replace it"
        )

    partial = destination.with_name(destination.name + ".part")
    if force:
        partial.unlink(missing_ok=True)
    resume_at = partial.stat().st_size if partial.exists() else 0
    if resume_at > expected_size:
        partial.unlink()
        resume_at = 0

    request = urllib.request.Request(source_url(relative_path), headers={"User-Agent": USER_AGENT})
    if resume_at:
        request.add_header("Range", f"bytes={resume_at}-")

    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if resume_at and exc.code == 416:
            response = None
        else:
            raise

    if response is not None:
        status = getattr(response, "status", response.getcode())
        append = bool(resume_at and status == 206)
        if resume_at and not append:
            resume_at = 0
        mode = "ab" if append else "wb"
        with response, partial.open(mode) as target:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                target.write(chunk)

    observed_size = partial.stat().st_size if partial.exists() else 0
    if observed_size != expected_size:
        raise ValueError(
            f"Size mismatch for {relative_path}: expected {expected_size}, got {observed_size}"
        )
    observed_hash = sha256(partial)
    if expected_hash is not None and observed_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch for {relative_path}: expected {expected_hash}, got {observed_hash}"
        )
    os.replace(partial, destination)
    return {
        "path": relative_path,
        "source_url": source_url(relative_path),
        "size": observed_size,
        "sha256": observed_hash,
        "expected_sha256": expected_hash,
        "status": "downloaded_verified",
    }


def verify_files(output_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in METADATA_FILES:
        path = output_dir / str(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing MARS-S2L metadata file: {path}")
        observed_size = path.stat().st_size
        if observed_size != int(item["size"]):
            raise ValueError(
                f"Size mismatch for {item['path']}: expected {item['size']}, got {observed_size}"
            )
        observed_hash = sha256(path)
        expected_hash = item["sha256"]
        if expected_hash is not None and observed_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {item['path']}: expected {expected_hash}, got {observed_hash}"
            )
        results.append(
            {
                "path": item["path"],
                "source_url": source_url(str(item["path"])),
                "size": observed_size,
                "sha256": observed_hash,
                "expected_sha256": expected_hash,
                "status": "verified",
            }
        )
    return results


def write_manifest(output_dir: Path, files: list[dict[str, Any]]) -> Path:
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": REPO_ID,
            "repository_url": f"https://huggingface.co/datasets/{REPO_ID}",
            "revision": REVISION,
            "license": "CC-BY-NC-SA-4.0",
            "scope": "metadata-only; no image assets",
        },
        "files": files,
        "integrity": {
            "file_count": len(files),
            "total_bytes": sum(int(item["size"]) for item in files),
            "all_expected_sizes_match": True,
            "all_declared_lfs_hashes_match": True,
        },
    }
    path = output_dir / MANIFEST_NAME
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--dry-run", action="store_true", help="List pinned files without downloading")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing files and rewrite manifest")
    parser.add_argument("--force", action="store_true", help="Replace existing files that fail verification")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.verify_only:
        parser.error("--dry-run and --verify-only are mutually exclusive")

    root = repo_root()
    try:
        output_dir = checked_output_dir(root, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        payload = {
            "ok": True,
            "dry_run": True,
            "repository": REPO_ID,
            "revision": REVISION,
            "output_dir": output_dir.relative_to(root).as_posix(),
            "file_count": len(METADATA_FILES),
            "total_bytes": sum(int(item["size"]) for item in METADATA_FILES),
            "files": list(METADATA_FILES),
        }
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            if args.verify_only:
                files = verify_files(output_dir)
            else:
                files = [download_file(item, output_dir, force=args.force) for item in METADATA_FILES]
            manifest = write_manifest(output_dir, files)
        except (FileNotFoundError, OSError, ValueError, urllib.error.URLError) as exc:
            payload = {"ok": False, "error": str(exc), "output_dir": str(output_dir)}
            print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
            return 2
        payload = {
            "ok": True,
            "dry_run": False,
            "verify_only": bool(args.verify_only),
            "repository": REPO_ID,
            "revision": REVISION,
            "output_dir": output_dir.relative_to(root).as_posix(),
            "manifest": manifest.relative_to(root).as_posix(),
            "file_count": len(files),
            "total_bytes": sum(int(item["size"]) for item in files),
            "statuses": {item["path"]: item["status"] for item in files},
        }
    print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
