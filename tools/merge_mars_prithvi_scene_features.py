#!/usr/bin/env python3
"""Verify and merge restartable per-fold Prithvi feature shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from extract_mars_scene_features import atomic_savez  # noqa: E402

DEFAULT_OUTPUT = Path("outputs/mars_prithvi_eo_2_tiny_tl_features_all_folds.npz")
ARRAY_KEYS = ("features", "labels", "sensors", "sample_ids", "groups", "folds")
SCALAR_KEYS = (
    "foundation_revision",
    "checkpoint_sha256",
    "foundation_receipt_sha256",
    "manifest_sha256",
    "protocol_sha256",
    "input_contract",
    "nir_transfer_contract",
    "missing_reference_datetime_policy",
)


def scalar(cache: np.lib.npyio.NpzFile, key: str) -> str:
    value = cache[key]
    if value.shape != ():
        raise ValueError(f"Expected scalar provenance field {key}")
    return str(value.item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Per-fold .npz shards")
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--expected-rows", type=int, default=44_363)
    args = parser.parse_args()
    if len(args.inputs) != 5:
        parser.error("exactly five frozen fold shards are required")
    root = repo_root()
    paths = [(root / value).resolve() for value in args.inputs]
    shards: list[dict[str, object]] = []
    reference_names: np.ndarray | None = None
    reference_scalars: dict[str, str] | None = None
    all_ids: set[str] = set()
    for path in paths:
        digest = sha256(path)
        with np.load(path, allow_pickle=False) as cache:
            missing = set((*ARRAY_KEYS, *SCALAR_KEYS, "feature_names")) - set(cache.files)
            if missing:
                raise ValueError(f"Shard {path} lacks keys: {sorted(missing)}")
            arrays = {key: cache[key] for key in ARRAY_KEYS}
            rows = arrays["labels"].shape[0]
            if any(value.shape[0] != rows for value in arrays.values()):
                raise ValueError(f"Row arrays do not align in shard {path}")
            folds = np.unique(arrays["folds"])
            if folds.size != 1 or int(folds[0]) not in range(5):
                raise ValueError(f"Shard {path} must contain exactly one fold")
            names = cache["feature_names"].astype(str)
            scalars = {key: scalar(cache, key) for key in SCALAR_KEYS}
            if reference_names is None:
                reference_names = names
                reference_scalars = scalars
            elif not np.array_equal(reference_names, names) or reference_scalars != scalars:
                raise ValueError(f"Schema or provenance differs in shard {path}")
            ids = arrays["sample_ids"].astype(str)
            if len(set(ids.tolist())) != ids.size or all_ids.intersection(ids.tolist()):
                raise ValueError(f"Duplicate sample IDs found in shard {path}")
            all_ids.update(ids.tolist())
            shards.append(
                {
                    "fold": int(folds[0]),
                    "arrays": arrays,
                    "missing": int(cache["missing_reference_datetime_rows"].item()),
                    "path": path,
                    "sha256": digest,
                }
            )
    shards.sort(key=lambda value: int(value["fold"]))
    observed_folds = [int(value["fold"]) for value in shards]
    if observed_folds != list(range(5)):
        raise ValueError(f"Expected folds 0..4 once each, got {observed_folds}")
    total_rows = sum(int(value["arrays"]["labels"].shape[0]) for value in shards)  # type: ignore[index]
    if total_rows != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows, got {total_rows}")
    assert reference_names is not None and reference_scalars is not None
    output = (root / args.output).resolve()
    atomic_savez(
        output,
        **{
            key: np.concatenate([value["arrays"][key] for value in shards])  # type: ignore[index]
            for key in ARRAY_KEYS
        },
        feature_names=reference_names,
        **{key: np.asarray(value) for key, value in reference_scalars.items()},
        missing_reference_datetime_rows=np.asarray(
            sum(int(value["missing"]) for value in shards)
        ),
        source_shard_folds=np.asarray(observed_folds, dtype=np.uint8),
        source_shard_sha256=np.asarray([str(value["sha256"]) for value in shards]),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "rows": total_rows,
                "features": int(reference_names.size),
                "missing_reference_datetime_rows": sum(
                    int(value["missing"]) for value in shards
                ),
                "output": args.output,
                "sha256": sha256(output),
                "shards": [
                    {
                        "fold": int(value["fold"]),
                        "path": str(Path(value["path"]).relative_to(root)),
                        "sha256": value["sha256"],
                    }
                    for value in shards
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
