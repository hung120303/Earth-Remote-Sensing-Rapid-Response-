#!/usr/bin/env python3
"""Accelerate the frozen MARS development download with Hugging Face Xet.

This is a transfer accelerator, not a replacement for the publication
verifier. It selects only assets named by a frozen cohort manifest, pins the
catalog's immutable Hub revision, and verifies each completed file against the
catalog identity. Run ``tools/acquire_mars_cohort.py --verify-only`` afterward
to produce the training receipt over the complete development cohort.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from acquire_mars_metadata import DEFAULT_OUTPUT, checked_output_dir, repo_root, sha256, verify_files
from acquire_mars_pilot import safe_asset_path
from acquire_mars_cohort import (
    FREE_SPACE_RESERVE_BYTES,
    incomplete_items,
    load_catalog,
    load_manifest_asset_paths,
    parallel_map,
    verify_one,
)
REPO_ID = "UNEP-IMEO/MARS-S2L"
REPO_TYPE = "dataset"
ASSET_REVISION = "c26b1d7e31a0c5241fa37c9140802622c215eb32"
REMOTE_CATALOG = "paper_v3_mixed_remote_catalog.jsonl"
DEFAULT_MANIFEST = "paper_v3_development_samples.jsonl"
DEFAULT_WORKLIST = "paper_v3_development_missing_worklist.json"
REVISION_PATTERN = re.compile(r"/resolve/([0-9a-f]{40})/")


def select_manifest_catalog(
    catalog: list[dict[str, Any]], selected_paths: set[str]
) -> list[dict[str, Any]]:
    by_path = {str(item["path"]): item for item in catalog}
    absent = selected_paths - set(by_path)
    if absent:
        raise ValueError(
            f"Cohort manifest names {len(absent)} assets absent from the remote catalog"
        )
    return [by_path[path] for path in sorted(selected_paths)]


def catalog_revision(items: list[dict[str, Any]]) -> str:
    revisions: set[str] = set()
    for item in items:
        match = REVISION_PATTERN.search(str(item.get("source_url", "")))
        if match is None:
            raise ValueError(f"Catalog asset has no immutable revision URL: {item['path']}")
        revisions.add(match.group(1))
    if revisions != {ASSET_REVISION}:
        raise ValueError(f"Unexpected catalog asset revisions: {sorted(revisions)}")
    return ASSET_REVISION


def write_worklist(
    path: Path,
    missing: list[dict[str, Any]],
    *,
    catalog_sha256: str,
    manifest_sha256: str,
) -> None:
    payload = {
        "schema_version": 1,
        "catalog_sha256": catalog_sha256,
        "manifest_sha256": manifest_sha256,
        "missing_paths": [str(item["path"]) for item in missing],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_worklist(
    path: Path,
    selected: list[dict[str, Any]],
    *,
    catalog_sha256: str,
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported Xet worklist schema")
    if payload.get("catalog_sha256") != catalog_sha256:
        raise ValueError("Xet worklist does not match the frozen catalog")
    if payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Xet worklist does not match the development manifest")
    paths = payload.get("missing_paths")
    if not isinstance(paths, list) or len(paths) != len(set(paths)):
        raise ValueError("Xet worklist paths must be a unique list")
    return select_manifest_catalog(selected, set(map(str, paths)))


def download_verified(
    metadata_dir: Path, item: dict[str, Any], *, revision: str
) -> dict[str, Any]:
    # Import lazily so catalog/unit tests remain usable in lightweight Python
    # environments that do not install the transfer client.
    from huggingface_hub import hf_hub_download

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            hf_hub_download(
                REPO_ID,
                filename=str(item["path"]),
                repo_type=REPO_TYPE,
                revision=revision,
                local_dir=metadata_dir,
                force_download=False,
            )
            checked = verify_one(metadata_dir, item)
            if checked["status"] != "verified":
                raise ValueError(
                    f"Xet transfer failed catalog verification ({checked['status']}): "
                    f"{item['path']}"
                )
            destination = safe_asset_path(metadata_dir, str(item["path"]))
            destination.with_name(destination.name + ".part").unlink(missing_ok=True)
            checked["status"] = "downloaded_xet_verified"
            return checked
        except Exception as exc:  # the final error retains the concrete cause
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def seed_tree_cache(metadata_dir: Path, revision: str) -> None:
    """Cache one immutable repo tree without selecting any payload files."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        REPO_ID,
        repo_type=REPO_TYPE,
        revision=revision,
        local_dir=metadata_dir,
        allow_patterns="__ersrr_tree_cache_only__",
        max_workers=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--catalog-file", default=REMOTE_CATALOG)
    parser.add_argument("--manifest-file", default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-assets", type=int)
    parser.add_argument(
        "--worklist-file",
        default=DEFAULT_WORKLIST,
        help="Ignored missing-asset inventory relative to the MARS directory",
    )
    parser.add_argument(
        "--refresh-worklist",
        action="store_true",
        help="Rescan the full cohort and replace the ignored worklist",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_assets is not None and args.max_assets <= 0:
        parser.error("--max-assets must be positive")

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    root = repo_root()
    try:
        metadata_dir = checked_output_dir(root, args.metadata_dir)
        verify_files(metadata_dir)
        catalog_path = safe_asset_path(metadata_dir, args.catalog_file)
        manifest_path = safe_asset_path(metadata_dir, args.manifest_file)
        worklist_path = safe_asset_path(metadata_dir, args.worklist_file)
        catalog = load_catalog(catalog_path)
        selected = select_manifest_catalog(
            catalog, load_manifest_asset_paths(manifest_path)
        )
        revision = catalog_revision(selected)
        catalog_hash = sha256(catalog_path)
        manifest_hash = sha256(manifest_path)
        if worklist_path.is_file() and not args.refresh_worklist:
            candidates = load_worklist(
                worklist_path,
                selected,
                catalog_sha256=catalog_hash,
                manifest_sha256=manifest_hash,
            )
            worklist_reused = True
        else:
            candidates = incomplete_items(metadata_dir, selected)
            write_worklist(
                worklist_path,
                candidates,
                catalog_sha256=catalog_hash,
                manifest_sha256=manifest_hash,
            )
            worklist_reused = False
        # A worklist is a point-in-time missing inventory. Do not repeat tens of
        # thousands of slow NTFS metadata calls under WSL: Hub downloads safely
        # reuse any entry completed by an interrupted run, and every scheduled
        # entry is catalog-verified below.
        missing = candidates
        if args.max_assets is not None:
            missing = missing[: args.max_assets]
        total_bytes = sum(int(item["size"]) for item in selected)
        missing_bytes = sum(int(item["size"]) for item in missing)
        state = {
            "ok": True,
            "dry_run": bool(args.dry_run),
            "repository": REPO_ID,
            "revision": revision,
            "catalog": {
                "path": catalog_path.relative_to(root).as_posix(),
                "sha256": catalog_hash,
            },
            "manifest": {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": manifest_hash,
            },
            "worklist": {
                "path": worklist_path.relative_to(root).as_posix(),
                "reused": worklist_reused,
                "candidate_asset_count": len(candidates),
            },
            "selected_asset_count": len(selected),
            "selected_total_bytes": total_bytes,
            "missing_asset_count": len(missing),
            "missing_bytes": missing_bytes,
            "missing_xet_asset_count": sum(
                item.get("xet_hash") is not None for item in missing
            ),
            "bounded_smoke": args.max_assets is not None,
        }
        if args.dry_run or not missing:
            print(
                json.dumps(state, indent=None if args.compact else 2, sort_keys=True)
            )
            return 0

        disk = shutil.disk_usage(metadata_dir)
        if disk.free < missing_bytes + FREE_SPACE_RESERVE_BYTES:
            raise ValueError(
                f"Insufficient free space: need {missing_bytes + FREE_SPACE_RESERVE_BYTES:,} "
                f"bytes including reserve, have {disk.free:,}"
            )
        seed_tree_cache(metadata_dir, revision)
        results = parallel_map(
            lambda item: download_verified(metadata_dir, item, revision=revision),
            missing,
            workers=args.workers,
            progress_label="Xet acquired and catalog-verified",
        )
        completed_paths = {str(result["path"]) for result in results}
        remaining_candidates = [
            item for item in candidates if str(item["path"]) not in completed_paths
        ]
        write_worklist(
            worklist_path,
            remaining_candidates,
            catalog_sha256=catalog_hash,
            manifest_sha256=manifest_hash,
        )
        state["downloaded_and_verified_count"] = len(results)
        state["remaining_asset_count"] = len(remaining_candidates)
        state["ok"] = state["remaining_asset_count"] == 0 or args.max_assets is not None
        print(json.dumps(state, indent=None if args.compact else 2, sort_keys=True))
        return 0 if state["ok"] else 3
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
