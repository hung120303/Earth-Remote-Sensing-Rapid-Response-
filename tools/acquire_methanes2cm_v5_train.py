#!/usr/bin/env python3
"""Download pinned MethaneS2CM archives and extract only v5 training imagery."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

# Xet's multi-file worker can leave large Windows-mounted downloads in
# CLOSE-WAIT after a transient CDN disconnect.  The standard Hub HTTP transport
# is slower but resumable and substantially more reliable for this acquisition.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

import h5py
import numpy as np
import tifffile
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_v3 import tracked_dirty, write_json  # noqa: E402

REPO_ID = "H1deaki/MethaneS2CM"
REVISION = "ee9a96d4994ca6bc45725c1e92d7a06258131eaf"
DATA_ROOT = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/"
    "external/MethaneS2CM"
)
SPLIT = "l2a_location_split_32x32"
DEFAULT_REPORT = Path("reports/experiments/methanes2cm_v5_train_acquisition.json")
DEFAULT_PACKED_TRAIN = DATA_ROOT / SPLIT / "v5_train_packed.h5"
ARCHIVES = {
    "dataset_part_001.tar.gz": (3_188_752_433, "dfaf1ddd5e94e0966d5e1d7f1770f2bab2b284292a80a525fc1846b941adb9b2"),
    "dataset_part_002.tar.gz": (3_097_937_508, "42da8fe076cd63366cdeed1501c41204a47117bb0cbf3b1afa884f2d1014ed93"),
    "dataset_part_003.tar.gz": (2_969_718_054, "314d6cf3fdb712c85464dd47d2722d4d9dd6003138816e839bfc112adafdaadd"),
    "dataset_part_004.tar.gz": (3_058_837_101, "a6cdf28faec67c1aee12682730b92b05f1f5cfcd2a3ca7275e525a6fe8e54315"),
    "dataset_part_005.tar.gz": (3_194_654_524, "7695c215431c60f511058ee5070f9e9286e675c967359d04e8b9169762d8d77e"),
    "dataset_part_006.tar.gz": (540_685_275, "9f4eb6575eb020358c39ab14da64f276319141176c59b7199a1f86b4b94234f3"),
}
ASSET_COLUMNS = {
    "s2_path": ("s2.tif", "target"),
    "s2_pre_path": ("s2_pre.tif", "reference90"),
    "s2_pre_pre_path": ("s2_pre_pre.tif", "reference365"),
    "plume_mask_path": ("plume.tif", "mask"),
}


def checked_destination(split_dir: Path, relative: str, expected_name: str) -> Path:
    value = PurePosixPath(relative)
    if (
        value.is_absolute()
        or ".." in value.parts
        or len(value.parts) != 2
        or value.name != expected_name
        or not value.parts[0].isdigit()
    ):
        raise ValueError(f"Unsafe MethaneS2CM CSV asset path: {relative!r}")
    # The exact two-part/digit/name contract above already excludes traversal.
    # Keep this lexical: resolving 320k nonexistent paths across /mnt/c is both
    # unnecessary and orders of magnitude slower than archive validation.
    base = Path(os.path.abspath(split_dir))
    return base / value.parts[0] / value.parts[1]


def expected_members(
    split_dir: Path, csv_path: Path
) -> tuple[dict[str, tuple[int, str]], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    rows.sort(key=lambda row: int(row["id"]))
    result: dict[str, tuple[int, str]] = {}
    for index, row in enumerate(rows):
        identifier = str(row["id"])
        if not identifier.isdigit():
            raise ValueError(f"Invalid MethaneS2CM id: {identifier!r}")
        for column, (filename, dataset) in ASSET_COLUMNS.items():
            relative = str(row[column])
            checked_destination(split_dir, relative, filename)
            member = f"{SPLIT}/{relative}"
            if member in result:
                raise ValueError(f"Duplicate expected archive member: {member}")
            result[member] = (index, dataset)
    return result, rows


def download_one(output: Path, filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            revision=REVISION,
            repo_type="dataset",
            local_dir=output,
        )
    )


def verify_archive(path: Path, expected: tuple[int, str]) -> None:
    size, identity = expected
    if path.stat().st_size != size or sha256(path) != identity:
        raise ValueError(f"Pinned archive identity mismatch: {path.name}")


def validate_packed(path: Path, expected_samples: int) -> dict[str, Any]:
    with h5py.File(path, "r") as source:
        required = {
            "sample_id": (expected_samples,),
            "label": (expected_samples,),
            "latitude": (expected_samples,),
            "longitude": (expected_samples,),
            "target": (expected_samples, 12, 32, 32),
            "reference90": (expected_samples, 12, 32, 32),
            "reference365": (expected_samples, 12, 32, 32),
            "mask": (expected_samples, 32, 32),
        }
        for name, shape in required.items():
            if name not in source or source[name].shape != shape:
                raise ValueError(f"Packed MethaneS2CM dataset {name!r} has the wrong shape")
        if str(source.attrs.get("source_revision")) != REVISION:
            raise ValueError("Packed MethaneS2CM source revision mismatch")
        mismatches = 0
        positives = 0
        for start in range(0, expected_samples, 2048):
            stop = min(start + 2048, expected_samples)
            labels = source["label"][start:stop].astype(bool)
            masks = np.any(source["mask"][start:stop] > 0, axis=(1, 2))
            mismatches += int(np.count_nonzero(labels != masks))
            positives += int(np.count_nonzero(labels))
        if mismatches:
            raise ValueError(f"Packed MethaneS2CM has {mismatches} label/mask disagreements")
    return {
        "samples": expected_samples,
        "positives": positives,
        "packed_bytes": path.stat().st_size,
        "packed_sha256": sha256(path),
    }


def pack_selected(
    archives: list[Path],
    expected: dict[str, tuple[int, str]],
    rows: list[dict[str, str]],
    destination: Path,
) -> dict[str, Any]:
    if destination.exists():
        reused = validate_packed(destination, len(rows))
        return {
            "expected_files": len(expected),
            "archive_members_seen": len(expected),
            "files_decoded": 0,
            "packed_file_reused": True,
            **reused,
        }
    seen: set[str] = set()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(temporary, "w") as packed:
        packed.attrs["source_dataset"] = REPO_ID
        packed.attrs["source_revision"] = REVISION
        packed.attrs["source_split"] = f"{SPLIT}/train.csv"
        packed.create_dataset(
            "sample_id", data=np.asarray([int(row["id"]) for row in rows], dtype=np.int64)
        )
        packed.create_dataset(
            "label", data=np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
        )
        packed.create_dataset(
            "latitude", data=np.asarray([float(row["latitude"]) for row in rows])
        )
        packed.create_dataset(
            "longitude", data=np.asarray([float(row["longitude"]) for row in rows])
        )
        for name in ("target", "reference90", "reference365"):
            packed.create_dataset(
                name,
                shape=(len(rows), 12, 32, 32),
                dtype=np.uint16,
                chunks=(1, 12, 32, 32),
                compression="lzf",
                shuffle=True,
            )
        packed.create_dataset(
            "mask",
            shape=(len(rows), 32, 32),
            dtype=np.uint8,
            chunks=(1, 32, 32),
            compression="lzf",
            shuffle=True,
        )
        for archive_path in archives:
            print(f"Scanning {archive_path.name} for sealed training members", flush=True)
            with tarfile.open(archive_path, mode="r|gz") as archive:
                for member in archive:
                    target = expected.get(member.name)
                    if target is None:
                        continue
                    if member.name in seen:
                        raise ValueError(
                            f"Duplicate training member across archives: {member.name}"
                        )
                    seen.add(member.name)
                    if not member.isfile() or member.size <= 0:
                        raise ValueError(
                            f"Expected regular nonempty archive member: {member.name}"
                        )
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(f"Could not read archive member: {member.name}")
                    with source:
                        values = np.asarray(tifffile.imread(io.BytesIO(source.read())))
                    index, dataset = target
                    expected_shape = (32, 32) if dataset == "mask" else (12, 32, 32)
                    expected_dtype = np.uint8 if dataset == "mask" else np.uint16
                    if values.shape != expected_shape or values.dtype != expected_dtype:
                        raise ValueError(
                            f"Unexpected {dataset} raster contract in {member.name}: "
                            f"{values.shape} {values.dtype}"
                        )
                    if dataset == "mask" and not set(
                        int(value) for value in np.unique(values)
                    ).issubset({0, 1}):
                        raise ValueError(f"Nonbinary plume mask in {member.name}")
                    packed[dataset][index] = values
            print(f"Found {len(seen):,}/{len(expected):,} required members", flush=True)
        missing = set(expected) - seen
        if missing:
            raise ValueError(f"Archives lack {len(missing):,} expected training assets")
        packed.flush()
    os.replace(temporary, destination)
    validation = validate_packed(destination, len(rows))
    return {
        "expected_files": len(expected),
        "archive_members_seen": len(seen),
        "files_decoded": len(seen),
        "packed_file_reused": False,
        **validation,
    }


def ignored(root: Path, path: Path) -> bool:
    relative = path.resolve().relative_to(root.resolve())
    return (
        subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
            cwd=root,
            check=False,
        ).returncode
        == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DATA_ROOT.as_posix())
    parser.add_argument("--packed-train", default=DEFAULT_PACKED_TRAIN.as_posix())
    parser.add_argument("--report", default=DEFAULT_REPORT.as_posix())
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    root = repo_root()
    if tracked_dirty(root):
        raise RuntimeError("Refusing acquisition from a dirty tracked worktree")
    output = (root / args.output).resolve()
    packed_train = (root / args.packed_train).resolve()
    report_path = (root / args.report).resolve()
    split_dir = output / SPLIT
    expected, train_rows = expected_members(split_dir, split_dir / "train.csv")
    if not ignored(root, output / next(iter(ARCHIVES))):
        raise ValueError("MethaneS2CM bulk output is not ignored by Git")

    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(download_one, output, filename): filename for filename in ARCHIVES
        }
        for future in as_completed(futures):
            filename = futures[future]
            paths[filename] = future.result()
            print(f"Downloaded {filename}", flush=True)
    for filename, expected_identity in ARCHIVES.items():
        verify_archive(paths[filename], expected_identity)
        print(f"Verified {filename}", flush=True)

    extraction = pack_selected(
        [paths[filename] for filename in sorted(ARCHIVES)],
        expected,
        train_rows,
        packed_train,
    )
    test_csv = split_dir / "test.csv"
    with test_csv.open("r", encoding="utf-8", newline="") as source:
        test_rows = list(csv.DictReader(source))
    extracted_test_directories = sum(
        (split_dir / str(row["id"])).is_dir() for row in test_rows
    )
    if extracted_test_directories:
        raise RuntimeError("Training acquisition violated the sealed location-test boundary")

    report = {
        "schema_version": 1,
        "scope": "methanes2cm_v5_location_training_acquisition",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "license": "CC-BY-NC-4.0",
        },
        "archives": [
            {
                "name": filename,
                "bytes": ARCHIVES[filename][0],
                "sha256": ARCHIVES[filename][1],
                "tracked": False,
            }
            for filename in sorted(ARCHIVES)
        ],
        "extraction": {
            "source_split": f"{SPLIT}/train.csv",
            **extraction,
            "packed_path": packed_train.relative_to(root).as_posix(),
            "tracked": False,
        },
        "sealed_test": {
            "sample_directories_present_after_training_extraction": extracted_test_directories,
            "images_opened": False,
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": Path(__file__).resolve().relative_to(root).as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "tracked_worktree_dirty_at_start": False,
        },
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
