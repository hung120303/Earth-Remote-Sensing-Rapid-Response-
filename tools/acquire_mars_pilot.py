#!/usr/bin/env python3
"""Download a deterministic MARS-S2L raster-contract pilot from pinned metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from acquire_mars_metadata import (
    CHUNK_SIZE,
    DEFAULT_OUTPUT,
    REPO_ID,
    REVISION,
    USER_AGENT,
    checked_output_dir,
    repo_root,
    sha256,
    source_url,
    verify_files,
)

PILOT_MANIFEST = "contract_pilot_manifest.json"
DEFAULT_SAMPLES_PER_CLASS = 3
DEFAULT_SEED = 42
MIN_CLEAR_PCT = 80.0


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected boolean text, got {value!r}")


def recommended_s2(row: dict[str, str]) -> bool:
    try:
        clear_pct = float(row["percentage_clear"])
    except ValueError:
        return False
    return (
        row["satellite"].startswith("S2")
        and "MSIL1C" in row["tile"]
        and row["observability"] == "clear"
        and clear_pct >= MIN_CLEAR_PCT
        and bool(row["background_image_tile"].strip())
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def stable_rank(row: dict[str, str], split: str, label: str, seed: int) -> str:
    value = f"{seed}\0{split}\0{label}\0{row['id_loc_image']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_rows(
    metadata_dir: Path, *, samples_per_class: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_rows = {split: read_rows(metadata_dir / f"{split}.csv") for split in ("train", "val", "test")}
    train_validation_locations = {
        row["id_location"] for split in ("train", "val") for row in split_rows[split]
    }
    selected: list[dict[str, Any]] = []
    availability: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        availability[split] = {}
        for label, expected in (("positive", True), ("negative", False)):
            candidates = [
                row
                for row in split_rows[split]
                if recommended_s2(row)
                and parse_bool(row["isplume"]) is expected
                and (split != "test" or row["id_location"] not in train_validation_locations)
            ]
            candidates.sort(key=lambda row: stable_rank(row, split, label, seed))
            if len(candidates) < samples_per_class:
                raise ValueError(
                    f"Only {len(candidates)} candidates for {split}/{label}; need {samples_per_class}"
                )
            availability[split][label] = len(candidates)
            for row in candidates[:samples_per_class]:
                assets = [
                    {"role": "image", "path": row["s2path"]},
                    {"role": "cloud_mask", "path": row["cloudmaskpath"]},
                ]
                if expected:
                    assets.extend(
                        [
                            {"role": "plume_mask", "path": row["plumepath"]},
                            {"role": "methane_enhancement", "path": row["ch4path"]},
                        ]
                    )
                if any(not item["path"].strip() for item in assets):
                    raise ValueError(f"Selected row has a missing required asset: {row['id_loc_image']}")
                selected.append(
                    {
                        "id_loc_image": row["id_loc_image"],
                        "split": split,
                        "label": label,
                        "id_location": row["id_location"],
                        "unseen_test_location": split == "test",
                        "satellite": row["satellite"],
                        "scene_id": row["tile"],
                        "scene_datetime": row["tile_date"],
                        "background_scene_id": row["background_image_tile"],
                        "observability": row["observability"],
                        "percentage_clear": float(row["percentage_clear"]),
                        "country": row["country"],
                        "longitude": float(row["lon"]),
                        "latitude": float(row["lat"]),
                        "crs": row["crs"],
                        "width": int(row["width"]),
                        "height": int(row["height"]),
                        "assets": assets,
                    }
                )
    return selected, availability


def safe_asset_path(metadata_dir: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe MARS-S2L asset path: {relative_path}")
    base = metadata_dir.resolve()
    # Resolving a non-existent >260-character final path on Windows can
    # intermittently introduce an extended-path prefix that compares unequal
    # to its base. The path is already lexically constrained above, so perform
    # a normalized common-path check and separately reject existing symlinks.
    destination = Path(os.path.abspath(os.path.join(str(base), *relative.parts)))
    if os.path.normcase(os.path.commonpath([str(base), str(destination)])) != os.path.normcase(
        str(base)
    ):
        raise ValueError(f"Asset path escapes metadata directory: {relative_path}")
    parent = destination.parent
    while parent != base:
        if parent.exists() and parent.is_symlink():
            raise ValueError(f"Asset path traverses a symlink: {relative_path}")
        parent = parent.parent
    return destination


def fetch_asset(metadata_dir: Path, relative_path: str, *, force: bool) -> dict[str, Any]:
    destination = safe_asset_path(metadata_dir, relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        return {
            "path": relative_path,
            "source_url": source_url(relative_path),
            "size": destination.stat().st_size,
            "sha256": sha256(destination),
            "status": "reused",
        }
    partial = destination.with_name(destination.name + ".part")
    if force:
        partial.unlink(missing_ok=True)
    request = urllib.request.Request(source_url(relative_path), headers={"User-Agent": USER_AGENT})
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to download {relative_path}: HTTP {exc.code}") from exc
    with response, partial.open("wb") as target:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            target.write(chunk)
    if not partial.is_file() or partial.stat().st_size == 0:
        raise ValueError(f"Downloaded empty asset: {relative_path}")
    os.replace(partial, destination)
    return {
        "path": relative_path,
        "source_url": source_url(relative_path),
        "size": destination.stat().st_size,
        "sha256": sha256(destination),
        "status": "downloaded",
    }


def write_manifest(
    metadata_dir: Path,
    samples: list[dict[str, Any]],
    availability: dict[str, Any],
    assets: list[dict[str, Any]],
    *,
    seed: int,
    samples_per_class: int,
) -> Path:
    identity = hashlib.sha256()
    for sample in sorted(samples, key=lambda item: item["id_loc_image"]):
        identity.update(sample["id_loc_image"].encode("utf-8") + b"\0")
    for asset in sorted(assets, key=lambda item: item["path"]):
        identity.update(asset["path"].encode("utf-8") + b"\0")
        identity.update(asset["sha256"].encode("ascii") + b"\0")
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": REPO_ID,
            "revision": REVISION,
            "license": "CC-BY-NC-SA-4.0",
        },
        "selection": {
            "seed": seed,
            "samples_per_class_per_split": samples_per_class,
            "sensor": "Sentinel-2 MSI",
            "product_level": "L1C",
            "observability": "clear",
            "minimum_clear_percentage": MIN_CLEAR_PCT,
            "requires_background_reference": True,
            "test_locations": "not present in train or validation",
            "candidate_availability": availability,
        },
        "integrity": {
            "sample_count": len(samples),
            "asset_count": len(assets),
            "total_bytes": sum(int(item["size"]) for item in assets),
            "pilot_identity_sha256": identity.hexdigest(),
        },
        "samples": samples,
        "assets": assets,
    }
    destination = metadata_dir / PILOT_MANIFEST
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def verify_manifest(metadata_dir: Path) -> dict[str, Any]:
    manifest_path = metadata_dir / PILOT_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source", {}).get("revision") != REVISION:
        raise ValueError("Pilot manifest revision does not match the pinned repository revision")
    checked: list[dict[str, Any]] = []
    for item in manifest["assets"]:
        destination = safe_asset_path(metadata_dir, item["path"])
        if not destination.is_file():
            raise FileNotFoundError(f"Missing pilot asset: {item['path']}")
        observed_size = destination.stat().st_size
        observed_hash = sha256(destination)
        if observed_size != item["size"] or observed_hash != item["sha256"]:
            raise ValueError(f"Pilot asset failed verification: {item['path']}")
        checked.append({**item, "status": "verified"})
    return {**manifest, "assets": checked}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.samples_per_class <= 0:
        parser.error("--samples-per-class must be positive")
    if args.dry_run and args.verify_only:
        parser.error("--dry-run and --verify-only are mutually exclusive")
    root = repo_root()
    try:
        metadata_dir = checked_output_dir(root, args.metadata_dir)
        verify_files(metadata_dir)
        if args.verify_only:
            manifest = verify_manifest(metadata_dir)
        else:
            samples, availability = select_rows(
                metadata_dir, samples_per_class=args.samples_per_class, seed=args.seed
            )
            paths = sorted({asset["path"] for sample in samples for asset in sample["assets"]})
            if args.dry_run:
                payload = {
                    "ok": True,
                    "dry_run": True,
                    "samples": samples,
                    "asset_count": len(paths),
                    "availability": availability,
                }
                print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
                return 0
            assets = [fetch_asset(metadata_dir, path, force=args.force) for path in paths]
            manifest_path = write_manifest(
                metadata_dir,
                samples,
                availability,
                assets,
                seed=args.seed,
                samples_per_class=args.samples_per_class,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=None if args.compact else 2))
        return 2
    payload = {
        "ok": True,
        "dry_run": False,
        "verify_only": bool(args.verify_only),
        "sample_count": manifest["integrity"]["sample_count"],
        "asset_count": manifest["integrity"]["asset_count"],
        "total_bytes": manifest["integrity"]["total_bytes"],
        "pilot_identity_sha256": manifest["integrity"]["pilot_identity_sha256"],
        "manifest": (metadata_dir / PILOT_MANIFEST).relative_to(root).as_posix(),
    }
    print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
