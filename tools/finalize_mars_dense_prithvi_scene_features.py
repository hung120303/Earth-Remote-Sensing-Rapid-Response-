#!/usr/bin/env python3
"""Validate and receipt a completed dense scene cache after a DrvFs stat race."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from extract_mars_dense_prithvi_scene_features import feature_names  # noqa: E402
from train_mars_paper_residual import iter_development_manifest  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_dense_prithvi_scene_feature_protocol.json")
GENERATOR_GIT_COMMIT = "8e3d0807650fc21215c19f43a9d702e692930212"


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_bytes(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    extractor = (ROOT / protocol["extractor"]["path"]).resolve()
    if sha256(extractor) != protocol["extractor"]["sha256"]:
        raise ValueError("Generator extractor differs from the frozen protocol")
    if git("cat-file", "-t", GENERATOR_GIT_COMMIT) != "commit":
        raise ValueError("Recorded generator commit is not available")
    generator_blob = git_bytes(
        "show",
        f"{GENERATOR_GIT_COMMIT}:{protocol['extractor']['path']}",
    )
    if hashlib.sha256(generator_blob).hexdigest() != protocol["extractor"]["sha256"]:
        raise ValueError("Generator commit does not contain the frozen extractor")

    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen input mismatch during finalization: {name}")
        if not path.exists():
            raise FileNotFoundError(f"Frozen input is unavailable: {name}")

    feature_path = (ROOT / protocol["outputs"]["features"]).resolve()
    metadata_path = (ROOT / protocol["outputs"]["metadata"]).resolve()
    receipt_path = (ROOT / protocol["outputs"]["receipt"]).resolve()
    if receipt_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing receipt: {receipt_path}")
    if not feature_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Completed feature and metadata caches are both required")

    matrix = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    expected_names = np.asarray(feature_names())
    if matrix.shape != (44363, expected_names.size) or matrix.dtype != np.float16:
        raise ValueError("Completed dense scene cache geometry differs from the protocol")
    for start in range(0, matrix.shape[0], 4096):
        if not np.isfinite(matrix[start : start + 4096]).all():
            raise ValueError(f"Dense scene cache has non-finite rows at {start}")

    with np.load(metadata_path, allow_pickle=False) as values:
        metadata = {name: values[name] for name in values.files}
    required = {
        "feature_names",
        "sample_ids",
        "labels",
        "sensors",
        "groups",
        "folds",
        "features_sha256",
        "adapter_sha256",
        "manifest_sha256",
        "fold_protocol_sha256",
        "sample_id_sha256",
        "input_contract",
    }
    if set(metadata) != required:
        raise ValueError(
            f"Dense scene metadata schema differs: missing={sorted(required - set(metadata))}, "
            f"extra={sorted(set(metadata) - required)}"
        )
    if not np.array_equal(metadata["feature_names"], expected_names):
        raise ValueError("Dense scene feature names differ from the frozen schema")
    sample_ids = metadata["sample_ids"].astype(str)
    if sample_ids.shape != (matrix.shape[0],) or np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("Dense scene sample identifiers are incomplete or duplicated")
    feature_hash = sha256(feature_path)
    if str(metadata["features_sha256"].item()) != feature_hash:
        raise ValueError("Dense scene feature hash differs from its metadata")
    if str(metadata["adapter_sha256"].item()) != protocol["inputs"]["adapter"]["sha256"]:
        raise ValueError("Dense scene metadata binds a different adapter")
    if str(metadata["manifest_sha256"].item()) != protocol["inputs"]["manifest"]["sha256"]:
        raise ValueError("Dense scene metadata binds a different manifest")
    if (
        str(metadata["fold_protocol_sha256"].item())
        != protocol["inputs"]["fold_protocol"]["sha256"]
    ):
        raise ValueError("Dense scene metadata binds a different fold protocol")
    sample_hash = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
    if str(metadata["sample_id_sha256"].item()) != sample_hash:
        raise ValueError("Dense scene sample identifier hash differs from metadata")

    records = list(
        iter_development_manifest(
            (ROOT / protocol["inputs"]["manifest"]["path"]).resolve()
        )
    )
    manifest_ids = np.asarray([str(record["sample_id"]) for record in records])
    if not np.array_equal(sample_ids, manifest_ids):
        raise ValueError("Dense scene rows do not align to manifest order")
    fold_protocol = json.loads(
        (ROOT / protocol["inputs"]["fold_protocol"]["path"]).read_text(encoding="utf-8")
    )
    group_to_fold = {
        str(row["group_id"]): int(row["fold"]) for row in fold_protocol["assignments"]
    }
    expected_groups = np.asarray([str(record["group_id"]) for record in records])
    expected_folds = np.asarray(
        [group_to_fold[str(record["group_id"])] for record in records], dtype=np.uint8
    )
    if not np.array_equal(metadata["groups"].astype(str), expected_groups):
        raise ValueError("Dense scene physical-site groups differ from the manifest")
    if not np.array_equal(metadata["folds"], expected_folds):
        raise ValueError("Dense scene folds differ from the frozen site protocol")

    receipt = {
        "schema_version": 1,
        "scope": "development-only frozen dense-Prithvi representation cache",
        "status": "validated after transient DrvFs post-rename stat visibility failure",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "fold_counts": {
            str(int(fold)): int(np.count_nonzero(expected_folds == fold))
            for fold in np.unique(expected_folds)
        },
        "features": {
            "path": feature_path.relative_to(ROOT).as_posix(),
            "bytes": feature_path.stat().st_size,
            "sha256": feature_hash,
            "dtype": str(matrix.dtype),
            "shape": list(matrix.shape),
        },
        "metadata": {
            "path": metadata_path.relative_to(ROOT).as_posix(),
            "bytes": metadata_path.stat().st_size,
            "sha256": sha256(metadata_path),
        },
        "sample_id_sha256": sample_hash,
        "adapter_sha256": protocol["inputs"]["adapter"]["sha256"],
        "protocol_sha256": sha256(protocol_path),
        "generator_extractor_sha256": protocol["extractor"]["sha256"],
        "generator_git_commit": GENERATOR_GIT_COMMIT,
        "finalizer_sha256": sha256(Path(__file__).resolve()),
        "finalizer_git_commit": git("rev-parse", "HEAD"),
        "external_inputs_accessed": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, receipt_path)
    print(json.dumps({"ok": True, **receipt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
