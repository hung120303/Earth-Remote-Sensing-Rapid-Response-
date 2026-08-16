"""Acquire only train-split MARS-Hyperspectral masks and compact metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


DEFAULT_REPOSITORY = "UNEP-IMEO/MARS-Hyperspectral"
DEFAULT_REVISION = "74b3d3132d135fee1761df1dadb7d662a4b5245b"
TRAIN_SPLIT_FILES = {
    "EMIT": "EMIT/train_t_v4a.csv",
    "EnMAP": "EnMAP/train_s_v4a.csv",
    "PRISMA": "PRISMA/train_s_v4a.csv",
}
ALLOWED_BASENAMES = {"info.json", "plumemask.tif"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_train_folders(metadata_root: Path) -> dict[str, str]:
    folders: dict[str, str] = {}
    for sensor, relative_csv in TRAIN_SPLIT_FILES.items():
        path = metadata_root / relative_csv
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"id", "folder_name"}
            if not required <= set(reader.fieldnames or []):
                raise ValueError(f"Missing train folder columns in {path}")
            for row in reader:
                sample_id = row["id"].strip()
                folder = row["folder_name"].strip()
                relative = PurePosixPath(sensor) / folder
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe source folder: {relative}")
                if sample_id in folders:
                    raise ValueError(f"Duplicate train sample ID: {sample_id}")
                folders[sample_id] = relative.as_posix()
    return folders


def allowed_patterns(train_folders: dict[str, str]) -> list[str]:
    return sorted(
        f"{folder}/{basename}"
        for folder in train_folders.values()
        for basename in sorted(ALLOWED_BASENAMES)
    )


def validate_download(
    *, output_dir: Path, expected_patterns: list[str]
) -> dict[str, object]:
    expected = set(expected_patterns)
    found: dict[str, Path] = {}
    unexpected: list[str] = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir).as_posix()
        if relative.startswith(".cache/") or relative == "train_label_manifest.json":
            continue
        if path.name not in ALLOWED_BASENAMES or relative not in expected:
            unexpected.append(relative)
            continue
        found[relative] = path
    if unexpected:
        raise ValueError(f"Unexpected downloaded files: {unexpected[:10]}")
    mask_paths = sorted(path for path in found if path.endswith("/plumemask.tif"))
    info_paths = sorted(path for path in found if path.endswith("/info.json"))
    missing_masks = sorted(path for path in expected if path.endswith("/plumemask.tif") and path not in found)
    missing_info = sorted(path for path in expected if path.endswith("/info.json") and path not in found)
    return {
        "expected_files": len(expected),
        "found_files": len(found),
        "mask_files": len(mask_paths),
        "info_files": len(info_paths),
        "missing_mask_files": missing_masks,
        "missing_info_files": missing_info,
        "bytes": sum(path.stat().st_size for path in found.values()),
        "sha256_by_path": {relative: sha256_file(found[relative]) for relative in sorted(found)},
    }


def acquire(
    *,
    metadata_root: Path,
    output_dir: Path,
    repository: str,
    revision: str,
    max_workers: int,
) -> dict[str, object]:
    from huggingface_hub import snapshot_download

    train_folders = read_train_folders(metadata_root)
    patterns = allowed_patterns(train_folders)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
        local_dir=output_dir,
        max_workers=max_workers,
    )
    validation = validate_download(output_dir=output_dir, expected_patterns=patterns)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "repository": repository,
        "revision": revision,
        "license": "CC-BY-NC-SA-4.0",
        "scope": "train_split_info_and_authoritative_masks_only",
        "train_samples": len(train_folders),
        **validation,
    }
    (output_dir / "train_label_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path(".research/mars_hyperspectral_transfer"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".research/mars_hyperspectral_transfer/train_labels"),
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-workers", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_workers < 1 or args.max_workers > 32:
        raise ValueError("max-workers must be in [1, 32]")
    report = acquire(
        metadata_root=args.metadata_root,
        output_dir=args.output_dir,
        repository=args.repository,
        revision=args.revision,
        max_workers=args.max_workers,
    )
    summary = {key: value for key, value in report.items() if key != "sha256_by_path"}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
