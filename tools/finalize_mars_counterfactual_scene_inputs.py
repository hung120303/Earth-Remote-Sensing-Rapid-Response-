#!/usr/bin/env python3
"""Validate and receipt the counterfactual cache after a DrvFs stat race."""

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
from extract_mars_counterfactual_scene_inputs import CHANNEL_NAMES  # noqa: E402
from mars_paper_model import SENSOR_NAMES  # noqa: E402
from train_mars_paper_residual import iter_development_manifest  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_counterfactual_scene_inputs_protocol.json")
GENERATOR_GIT_COMMIT = "7f339caa"


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
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if git("cat-file", "-t", GENERATOR_GIT_COMMIT) != "commit":
        raise ValueError("Recorded counterfactual generator commit is unavailable")
    for key, path in (
        ("extractor", protocol["extractor"]["path"]),
        ("protocol", args.protocol),
    ):
        committed = git_bytes("show", f"{GENERATOR_GIT_COMMIT}:{path}")
        current = (ROOT / path).read_bytes()
        if hashlib.sha256(committed).digest() != hashlib.sha256(current).digest():
            raise ValueError(f"Current {key} differs from the generator commit")
    extractor_path = (ROOT / protocol["extractor"]["path"]).resolve()
    if sha256(extractor_path) != protocol["extractor"]["sha256"]:
        raise ValueError("Counterfactual extractor differs from its frozen protocol")

    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Frozen input is unavailable: {name}")
        if path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen counterfactual input hash mismatch: {name}")

    image_path = (ROOT / protocol["outputs"]["images"]).resolve()
    metadata_path = (ROOT / protocol["outputs"]["metadata"]).resolve()
    receipt_path = (ROOT / protocol["outputs"]["receipt"]).resolve()
    if receipt_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing receipt: {receipt_path}")
    if not image_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Both completed counterfactual caches are required")

    images = np.load(image_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (17745, len(CHANNEL_NAMES), 64, 64)
    if images.shape != expected_shape or images.dtype != np.float16:
        raise ValueError("Completed counterfactual image cache has the wrong schema")
    finite_min = np.inf
    finite_max = -np.inf
    for start in range(0, images.shape[0], 256):
        chunk = np.asarray(images[start : start + 256])
        if not np.isfinite(chunk).all():
            raise ValueError(f"Counterfactual cache has non-finite values near row {start}")
        finite_min = min(finite_min, float(chunk.min()))
        finite_max = max(finite_max, float(chunk.max()))

    with np.load(metadata_path, allow_pickle=False) as values:
        metadata = {name: values[name] for name in values.files}
    required = {
        "channel_names",
        "labels",
        "sensors",
        "sample_ids",
        "groups",
        "folds",
        "images_sha256",
        "manifest_sha256",
        "fold_protocol_sha256",
        "released_checkpoint_sha256",
        "protocol_sha256",
    }
    if set(metadata) != required:
        raise ValueError(
            f"Counterfactual metadata schema differs: missing={sorted(required - set(metadata))}, "
            f"extra={sorted(set(metadata) - required)}"
        )
    if not np.array_equal(metadata["channel_names"].astype(str), np.asarray(CHANNEL_NAMES)):
        raise ValueError("Counterfactual channel names differ from the frozen schema")
    image_hash = sha256(image_path)
    if str(metadata["images_sha256"].item()) != image_hash:
        raise ValueError("Counterfactual image hash differs from metadata")
    for key, input_name in (
        ("manifest_sha256", "manifest"),
        ("fold_protocol_sha256", "fold_protocol"),
        ("released_checkpoint_sha256", "released_checkpoint"),
    ):
        if str(metadata[key].item()) != protocol["inputs"][input_name]["sha256"]:
            raise ValueError(f"Counterfactual metadata binds a different {input_name}")
    if str(metadata["protocol_sha256"].item()) != sha256(protocol_path):
        raise ValueError("Counterfactual metadata binds a different protocol")

    fold_protocol = json.loads(
        (ROOT / protocol["inputs"]["fold_protocol"]["path"]).read_text(encoding="utf-8")
    )
    group_to_fold = {
        str(item["group_id"]): int(item["fold"]) for item in fold_protocol["assignments"]
    }
    records = [
        record
        for record in iter_development_manifest(
            (ROOT / protocol["inputs"]["manifest"]["path"]).resolve()
        )
        if group_to_fold[str(record["group_id"])] in set(map(int, protocol["folds"]))
    ]
    expected_ids = np.asarray([str(record["sample_id"]) for record in records])
    expected_groups = np.asarray([str(record["group_id"]) for record in records])
    expected_folds = np.asarray(
        [group_to_fold[str(record["group_id"])] for record in records], dtype=np.uint8
    )
    expected_labels = np.asarray(
        [int(record["label_state"] == "PLUME") for record in records], dtype=np.uint8
    )
    expected_sensors = np.asarray(
        [SENSOR_NAMES.index(str(record["sensor_family"])) for record in records],
        dtype=np.uint8,
    )
    for key, expected in (
        ("sample_ids", expected_ids),
        ("groups", expected_groups),
        ("folds", expected_folds),
        ("labels", expected_labels),
        ("sensors", expected_sensors),
    ):
        observed = metadata[key].astype(str) if expected.dtype.kind == "U" else metadata[key]
        if not np.array_equal(observed, expected):
            raise ValueError(f"Counterfactual {key} do not match manifest order")
    if np.unique(expected_ids).size != expected_ids.size:
        raise ValueError("Counterfactual sample identities are duplicated")

    receipt = {
        "schema_version": 1,
        "scope": "development folds 3/4 label-free counterfactual feature extraction",
        "status": "validated after transient DrvFS post-rename stat visibility failure",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(images.shape[0]),
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "finite_range": [finite_min, finite_max],
        "fold_counts": {
            str(int(fold)): int(np.count_nonzero(expected_folds == fold))
            for fold in np.unique(expected_folds)
        },
        "label_counts": {
            "NO_PLUME": int(np.count_nonzero(expected_labels == 0)),
            "PLUME": int(np.count_nonzero(expected_labels == 1)),
        },
        "outputs": {
            "images": {
                "path": image_path.relative_to(ROOT).as_posix(),
                "bytes": image_path.stat().st_size,
                "sha256": image_hash,
                "tracked": False,
            },
            "metadata": {
                "path": metadata_path.relative_to(ROOT).as_posix(),
                "bytes": metadata_path.stat().st_size,
                "sha256": sha256(metadata_path),
                "tracked": False,
            },
        },
        "protocol_sha256": sha256(protocol_path),
        "generator_extractor_sha256": protocol["extractor"]["sha256"],
        "generator_git_commit": git("rev-parse", GENERATOR_GIT_COMMIT),
        "finalizer_sha256": sha256(Path(__file__).resolve()),
        "finalizer_git_commit": git("rev-parse", "HEAD"),
        "external_inputs_accessed": False,
        "invariants": [
            "All 17,745 selected rows and every float16 value were validated.",
            "Every sample, site, label, sensor, and fold matches the frozen manifest order.",
            "Only development folds 3 and 4 are present.",
            "No fold 0/1/2, exact-paper, or fresh-external input was accessed.",
        ],
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
