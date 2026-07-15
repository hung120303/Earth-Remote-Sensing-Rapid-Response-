#!/usr/bin/env python3
"""Verify and merge label-independent paper-cohort Prithvi CLS shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from extract_mars_scene_features import atomic_savez  # noqa: E402

DEFAULT_OUTPUT = Path("outputs/mars_paper_prithvi_cls_features.npz")
DEFAULT_RECEIPT = Path("reports/acquisition/mars_paper_prithvi_cls_features.json")
EXPECTED_ROWS = 43_524
EXPECTED_FEATURES = 768
ARRAY_KEYS = ("features", "sample_ids", "groups", "sensors")
SCALAR_KEYS = (
    "shard_count",
    "total_available_rows",
    "foundation_revision",
    "checkpoint_sha256",
    "foundation_receipt_sha256",
    "sealed_manifest_sha256",
    "test_acquisition_receipt_sha256",
    "input_contract",
    "nir_transfer_contract",
    "missing_reference_datetime_policy",
)


def scalar(cache: np.lib.npyio.NpzFile, key: str) -> object:
    value = cache[key]
    if value.shape != ():
        raise ValueError(f"Expected scalar field {key}")
    return value.item()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT.as_posix())
    args = parser.parse_args()
    if len(args.inputs) != 5:
        parser.error("exactly five paper feature shards are required")
    root = repo_root()
    shards = []
    reference_names: np.ndarray | None = None
    reference_scalars: dict[str, object] | None = None
    all_ids: set[str] = set()
    for value in args.inputs:
        path = (root / value).resolve()
        digest = sha256(path)
        with np.load(path, allow_pickle=False) as cache:
            if "labels" in cache.files:
                raise ValueError(f"Paper feature shard contains labels: {path}")
            required = set(
                (
                    *ARRAY_KEYS,
                    *SCALAR_KEYS,
                    "feature_names",
                    "shard_index",
                    "shard_start",
                    "shard_end",
                    "missing_reference_datetime_rows",
                )
            )
            if missing := required - set(cache.files):
                raise ValueError(f"Paper shard lacks keys: {sorted(missing)}")
            arrays = {key: cache[key] for key in ARRAY_KEYS}
            rows = arrays["sample_ids"].shape[0]
            if any(array.shape[0] != rows for array in arrays.values()):
                raise ValueError(f"Paper shard rows do not align: {path}")
            names = cache["feature_names"].astype(str)
            if (
                arrays["features"].shape != (rows, EXPECTED_FEATURES)
                or names.size != EXPECTED_FEATURES
            ):
                raise ValueError(f"Paper shard feature schema differs: {path}")
            scalars = {key: scalar(cache, key) for key in SCALAR_KEYS}
            if reference_names is None:
                reference_names = names
                reference_scalars = scalars
            elif not np.array_equal(reference_names, names) or reference_scalars != scalars:
                raise ValueError(f"Paper shard schema or provenance differs: {path}")
            ids = arrays["sample_ids"].astype(str)
            if len(set(ids.tolist())) != ids.size or all_ids.intersection(ids.tolist()):
                raise ValueError(f"Duplicate paper sample IDs in shard: {path}")
            all_ids.update(ids.tolist())
            shards.append(
                {
                    "index": int(scalar(cache, "shard_index")),
                    "start": int(scalar(cache, "shard_start")),
                    "end": int(scalar(cache, "shard_end")),
                    "missing": int(scalar(cache, "missing_reference_datetime_rows")),
                    "arrays": arrays,
                    "path": path,
                    "sha256": digest,
                }
            )
    shards.sort(key=lambda item: item["index"])
    if [item["index"] for item in shards] != list(range(5)):
        raise ValueError("Paper shards do not cover indices 0..4 exactly once")
    cursor = 0
    for item in shards:
        if item["start"] != cursor or item["end"] <= item["start"]:
            raise ValueError("Paper shard ranges are not contiguous and ordered")
        cursor = item["end"]
    if cursor != EXPECTED_ROWS or len(all_ids) != EXPECTED_ROWS:
        raise ValueError("Paper shards do not cover all 43,524 available rows")
    assert reference_names is not None and reference_scalars is not None
    if int(reference_scalars["total_available_rows"]) != EXPECTED_ROWS:
        raise ValueError("Paper shard declared total differs from exact available cohort")
    output = (root / args.output).resolve()
    atomic_savez(
        output,
        **{
            key: np.concatenate([item["arrays"][key] for item in shards])
            for key in ARRAY_KEYS
        },
        feature_names=reference_names,
        **{key: np.asarray(value) for key, value in reference_scalars.items()},
        missing_reference_datetime_rows=np.asarray(sum(item["missing"] for item in shards)),
        source_shard_sha256=np.asarray([item["sha256"] for item in shards]),
    )
    report = {
        "schema_version": 1,
        "scope": "ignored label-independent Prithvi CLS cache for exact available paper rows",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": EXPECTED_ROWS,
        "features": EXPECTED_FEATURES,
        "label_independent_output": True,
        "missing_reference_datetime_rows": sum(item["missing"] for item in shards),
        "output": str(output.relative_to(root)),
        "sha256": sha256(output),
        "provenance": reference_scalars,
        "shards": [
            {
                "index": item["index"],
                "start": item["start"],
                "end": item["end"],
                "rows": item["end"] - item["start"],
                "path": str(item["path"].relative_to(root)),
                "sha256": item["sha256"],
            }
            for item in shards
        ],
    }
    receipt = (root / args.receipt).resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix(receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
