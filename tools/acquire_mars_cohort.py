#!/usr/bin/env python3
"""Download or verify the frozen MARS-S2L publication cohort safely.

The remote catalog must first be created by ``tools/build_mars_cohort.py``.
Downloads are pinned, resumable, bounded to the ignored acquisition root, and
verified against LFS SHA-256 or Git blob SHA-1 identities. This tool never
places raw imagery in a Git-visible dataset directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from acquire_mars_metadata import (
    CHUNK_SIZE,
    DEFAULT_OUTPUT,
    USER_AGENT,
    checked_output_dir,
    repo_root,
    sha256,
    verify_files,
)
from acquire_mars_pilot import safe_asset_path
from build_mars_cohort import REMOTE_CATALOG

DEFAULT_WORKERS = 8
FREE_SPACE_RESERVE_BYTES = 5 * 1024**3


def file_identity(path: Path, oid_type: str, size: int) -> str:
    if oid_type == "sha256_lfs":
        digest = hashlib.sha256()
        prefix = b""
    elif oid_type == "git_blob_sha1":
        digest = hashlib.sha1()
        prefix = f"blob {size}\0".encode("ascii")
    else:
        raise ValueError(f"Unsupported remote oid type: {oid_type}")
    digest.update(prefix)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen remote catalog: {path}; run tools/build_mars_cohort.py first"
        )
    items: list[dict[str, Any]] = []
    observed: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid remote catalog JSONL at line {line_number}") from exc
            path_value = str(item["path"])
            if path_value in observed:
                raise ValueError(f"Duplicate remote catalog path: {path_value}")
            observed.add(path_value)
            if int(item["size"]) <= 0:
                raise ValueError(f"Remote catalog has a non-positive size: {path_value}")
            items.append(item)
    items.sort(key=lambda item: item["path"])
    return items


def load_manifest_asset_paths(path: Path) -> set[str]:
    """Return the exact asset set named by a frozen JSONL cohort manifest."""
    selected: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid cohort manifest JSONL at line {line_number}"
                ) from exc
            assets = record.get("assets")
            if not isinstance(assets, list) or not assets:
                raise ValueError(
                    f"Cohort manifest row {line_number} has no asset list"
                )
            for asset in assets:
                value = str(asset.get("path") or "")
                if not value:
                    raise ValueError(
                        f"Cohort manifest row {line_number} has an empty asset path"
                    )
                selected.add(value)
    if not selected:
        raise ValueError("Cohort manifest selects no assets")
    return selected


def partial_bytes(metadata_dir: Path, item: dict[str, Any]) -> int:
    destination = safe_asset_path(metadata_dir, item["path"])
    partial = destination.with_name(destination.name + ".part")
    if destination.is_file():
        return int(item["size"])
    if partial.is_file():
        return min(partial.stat().st_size, int(item["size"]))
    return 0


def inventory(metadata_dir: Path, items: list[dict[str, Any]]) -> dict[str, Any]:
    complete_size_matches = 0
    partial_count = 0
    present_bytes = 0
    for item in items:
        destination = safe_asset_path(metadata_dir, item["path"])
        partial = destination.with_name(destination.name + ".part")
        if destination.is_file() and destination.stat().st_size == int(item["size"]):
            complete_size_matches += 1
            present_bytes += int(item["size"])
        elif partial.is_file():
            partial_count += 1
            present_bytes += min(partial.stat().st_size, int(item["size"]))
    total_bytes = sum(int(item["size"]) for item in items)
    return {
        "selected_asset_count": len(items),
        "selected_total_bytes": total_bytes,
        "complete_size_match_count": complete_size_matches,
        "partial_count": partial_count,
        "present_or_partial_bytes": present_bytes,
        "remaining_bytes": total_bytes - present_bytes,
    }


def verify_one(metadata_dir: Path, item: dict[str, Any]) -> dict[str, Any]:
    destination = safe_asset_path(metadata_dir, item["path"])
    if not destination.is_file():
        return {"path": item["path"], "status": "missing"}
    observed_size = destination.stat().st_size
    if observed_size != int(item["size"]):
        return {
            "path": item["path"],
            "status": "size_mismatch",
            "expected_size": int(item["size"]),
            "observed_size": observed_size,
        }
    observed_oid = file_identity(destination, item["remote_oid_type"], observed_size)
    if observed_oid != item["remote_oid"]:
        return {
            "path": item["path"],
            "status": "hash_mismatch",
            "expected_oid": item["remote_oid"],
            "observed_oid": observed_oid,
        }
    return {
        "path": item["path"],
        "status": "verified",
        "size": observed_size,
        "remote_oid": observed_oid,
        "remote_oid_type": item["remote_oid_type"],
    }


def download_response(url: str, resume_at: int) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Connection": "close"}
    )
    if resume_at:
        request.add_header("Range", f"bytes={resume_at}-")
    for attempt in range(5):
        try:
            return urllib.request.urlopen(request, timeout=120)
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (
                429,
                500,
                502,
                503,
                504,
            ):
                raise
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable download retry state")


def acquire_one(metadata_dir: Path, item: dict[str, Any], *, force: bool) -> dict[str, Any]:
    destination = safe_asset_path(metadata_dir, item["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        checked = verify_one(metadata_dir, item)
        if checked["status"] == "verified":
            checked["status"] = "reused_verified"
            return checked
        raise ValueError(
            f"Existing asset failed verification ({checked['status']}): {item['path']}; "
            "use --force to replace it"
        )
    partial = destination.with_name(destination.name + ".part")
    if force:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    expected_size = int(item["size"])
    resume_at = partial.stat().st_size if partial.is_file() else 0
    if resume_at > expected_size:
        partial.unlink()
        resume_at = 0

    response = download_response(item["source_url"], resume_at)
    try:
        status = getattr(response, "status", response.getcode())
        append = bool(resume_at and status == 206)
        if resume_at and not append:
            resume_at = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as target:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                target.write(chunk)
    finally:
        # CPython normally closes HTTPResponse objects immediately, but the
        # Windows SSL stack can otherwise accumulate completed CLOSE_WAIT
        # sockets during tens of thousands of small-file requests.
        response.close()

    observed_size = partial.stat().st_size if partial.is_file() else 0
    if observed_size != expected_size:
        raise ValueError(
            f"Downloaded size mismatch for {item['path']}: expected {expected_size}, got {observed_size}"
        )
    observed_oid = file_identity(partial, item["remote_oid_type"], observed_size)
    if observed_oid != item["remote_oid"]:
        raise ValueError(
            f"Downloaded hash mismatch for {item['path']}: expected {item['remote_oid']}, got {observed_oid}"
        )
    os.replace(partial, destination)
    return {
        "path": item["path"],
        "status": "downloaded_verified",
        "size": observed_size,
        "remote_oid": observed_oid,
        "remote_oid_type": item["remote_oid_type"],
    }


def parallel_map(
    function: Any, items: list[dict[str, Any]], *, workers: int, progress_label: str
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, item): item for item in items}
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 100 == 0:
                gc.collect()
            if completed % 1000 == 0 or completed == len(items):
                print(
                    f"{progress_label}: {completed:,}/{len(items):,}",
                    file=sys.stderr,
                    flush=True,
                )
    results.sort(key=lambda item: item["path"])
    return results


def status_counts(results: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(item["status"] for item in results)
    return dict(sorted(counts.items()))


def write_receipt(
    root: Path,
    value: str,
    payload: dict[str, Any],
    catalog_path: Path,
) -> Path:
    path = (root / value).resolve()
    if root not in path.parents:
        raise ValueError("Receipt output must resolve beneath the repository root")
    receipt = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "verify_only" if payload.get("verify_only") else "acquire_and_verify",
        "catalog": {
            "path": catalog_path.relative_to(root).as_posix(),
            "sha256": sha256(catalog_path),
            "asset_count": payload["selected_asset_count"],
            "total_bytes": payload["selected_total_bytes"],
        },
        "result": payload,
        "verification": {
            "lfs": "SHA-256 over file bytes",
            "regular_git_file": "Git blob SHA-1 over blob header and file bytes",
            "size_checked": True,
            "all_selected_assets_verified": bool(payload["ok"]),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument(
        "--catalog-file",
        default=REMOTE_CATALOG,
        help="Catalog path relative to the ignored MARS directory",
    )
    parser.add_argument(
        "--manifest-file",
        help="Optional cohort JSONL path relative to the ignored MARS directory; acquire only assets named by it",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--max-assets",
        type=int,
        help="Operate on only the first N catalog paths for a smoke test; never use for a full claim",
    )
    parser.add_argument(
        "--start-asset",
        type=int,
        default=0,
        help="Zero-based catalog offset for a bounded transfer smoke test; never use for a full claim",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--receipt", help="Write a compact verification receipt beneath the repository root")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_assets is not None and args.max_assets <= 0:
        parser.error("--max-assets must be positive")
    if args.start_asset < 0:
        parser.error("--start-asset must be non-negative")
    if args.dry_run and args.verify_only:
        parser.error("--dry-run and --verify-only are mutually exclusive")
    root = repo_root()
    try:
        metadata_dir = checked_output_dir(root, args.metadata_dir)
        verify_files(metadata_dir)
        catalog_path = safe_asset_path(metadata_dir, args.catalog_file)
        catalog = load_catalog(catalog_path)
        full_catalog_total = len(catalog)
        manifest_path: Path | None = None
        if args.manifest_file:
            manifest_path = safe_asset_path(metadata_dir, args.manifest_file)
            selected_paths = load_manifest_asset_paths(manifest_path)
            by_path = {str(item["path"]): item for item in catalog}
            missing = selected_paths - set(by_path)
            if missing:
                raise ValueError(
                    f"Cohort manifest names {len(missing)} assets absent from the remote catalog"
                )
            catalog = [by_path[path] for path in sorted(selected_paths)]
        catalog_total = len(catalog)
        if args.start_asset >= catalog_total:
            raise ValueError(
                f"--start-asset {args.start_asset:,} is outside the {catalog_total:,}-asset catalog"
            )
        catalog = catalog[args.start_asset :]
        if args.max_assets is not None:
            catalog = catalog[: args.max_assets]
        state = inventory(metadata_dir, catalog)
        state["catalog_asset_count"] = catalog_total
        state["catalog_start_asset"] = args.start_asset
        state["partial_scope"] = (
            len(catalog) != catalog_total or catalog_total != full_catalog_total
        )
        state["source_catalog_asset_count"] = full_catalog_total
        state["metadata_dir"] = metadata_dir.relative_to(root).as_posix()
        state["remote_catalog"] = catalog_path.relative_to(root).as_posix()
        if manifest_path is not None:
            state["manifest_filter"] = {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": sha256(manifest_path),
            }
        if args.dry_run:
            print(json.dumps({"ok": True, "dry_run": True, **state}, indent=None if args.compact else 2, sort_keys=True))
            return 0
        if args.verify_only:
            results = parallel_map(
                lambda item: verify_one(metadata_dir, item),
                catalog,
                workers=args.workers,
                progress_label="Verified",
            )
            counts = status_counts(results)
            ok = counts.get("verified", 0) == len(catalog)
            payload = {
                "ok": ok,
                "dry_run": False,
                "verify_only": True,
                **state,
                "statuses": counts,
            }
            if args.receipt:
                receipt = write_receipt(root, args.receipt, payload, catalog_path)
                payload["receipt"] = receipt.relative_to(root).as_posix()
            print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
            return 0 if ok else 3

        disk = shutil.disk_usage(metadata_dir)
        if disk.free < int(state["remaining_bytes"]) + FREE_SPACE_RESERVE_BYTES:
            raise ValueError(
                f"Insufficient free space: need {int(state['remaining_bytes']) + FREE_SPACE_RESERVE_BYTES:,} "
                f"bytes including reserve, have {disk.free:,}"
            )
        results = parallel_map(
            lambda item: acquire_one(metadata_dir, item, force=args.force),
            catalog,
            workers=args.workers,
            progress_label="Acquired",
        )
        counts = status_counts(results)
        verified_count = counts.get("reused_verified", 0) + counts.get("downloaded_verified", 0)
        final_state = inventory(metadata_dir, catalog)
        final_state["catalog_asset_count"] = catalog_total
        final_state["catalog_start_asset"] = args.start_asset
        final_state["partial_scope"] = (
            len(catalog) != catalog_total or catalog_total != full_catalog_total
        )
        final_state["source_catalog_asset_count"] = full_catalog_total
        final_state["metadata_dir"] = metadata_dir.relative_to(root).as_posix()
        final_state["remote_catalog"] = catalog_path.relative_to(root).as_posix()
        if manifest_path is not None:
            final_state["manifest_filter"] = {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": sha256(manifest_path),
            }
        payload = {
            "ok": verified_count == len(catalog),
            "dry_run": False,
            "verify_only": False,
            **final_state,
            "statuses": counts,
        }
        if args.receipt:
            receipt = write_receipt(root, args.receipt, payload, catalog_path)
            payload["receipt"] = receipt.relative_to(root).as_posix()
        print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
        return 0 if payload["ok"] else 3
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        urllib.error.URLError,
    ) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=None if args.compact else 2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
