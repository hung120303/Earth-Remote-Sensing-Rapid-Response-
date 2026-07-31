#!/usr/bin/env python3
"""Extract and finalize the one-shot DOFA-v2 fold-2 feature cache."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_dofa_v2_base import CHECKPOINT_SHA256  # noqa: E402
from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from extract_mars_dofa_v2_scene_features import (  # noqa: E402
    FEATURE_WIDTH,
    MARS_TO_DOFA_MULTIPLIER,
    feature_names,
)

DEFAULT_PROTOCOL = Path("configs/mars_dofa_v2_fold2_protocol.json")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def verify_static_contract(protocol: dict[str, Any]) -> None:
    if sha256(Path(__file__).resolve()) != protocol["extractor"]["sha256"]:
        raise ValueError("Frozen fold-2 extractor hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    for name, contract in protocol["inputs"].items():
        if "sha256" not in contract:
            continue
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen fold-2 input mismatch: {name}")


def validate_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as cache:
        features = cache["features"]
        names = cache["feature_names"].astype(str)
        labels = cache["labels"].astype(np.uint8)
        sensors = cache["sensors"].astype(np.uint8)
        sample_ids = cache["sample_ids"].astype(str)
        groups = cache["groups"].astype(str)
        folds = cache["folds"].astype(np.uint8)
        if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
            raise ValueError("Fold-2 DOFA feature shape differs")
        if names.tolist() != feature_names():
            raise ValueError("Fold-2 DOFA feature names differ")
        row_count = features.shape[0]
        for values, label in (
            (labels, "labels"),
            (sensors, "sensors"),
            (sample_ids, "sample IDs"),
            (groups, "groups"),
            (folds, "folds"),
        ):
            if values.shape != (row_count,):
                raise ValueError(f"Fold-2 DOFA {label} do not align")
        if np.unique(folds).tolist() != [2]:
            raise ValueError("Fold-2 cache contains a non-fold-2 row")
        if len(np.unique(sample_ids)) != row_count:
            raise ValueError("Fold-2 cache sample IDs are not unique")
        if str(cache["checkpoint_sha256"]) != CHECKPOINT_SHA256:
            raise ValueError("Fold-2 checkpoint identity differs")
        if float(cache["mars_to_dofa_multiplier"]) != MARS_TO_DOFA_MULTIPLIER:
            raise ValueError("Fold-2 radiometric contract differs")
        finite = np.isfinite(features)
        if not finite.all():
            raise ValueError("Fold-2 DOFA cache contains non-finite values")
        return {
            "rows": int(row_count),
            "feature_count": int(features.shape[1]),
            "positive_rows": int(labels.sum()),
            "negative_rows": int(row_count - labels.sum()),
            "sensors": {
                "Sentinel-2": int((sensors == 0).sum()),
                "Landsat": int((sensors == 1).sum()),
            },
            "physical_groups": int(len(np.unique(groups))),
            "minimum": float(features.min()),
            "maximum": float(features.max()),
            "mean": float(features.astype(np.float64).mean()),
            "standard_deviation": float(features.astype(np.float64).std()),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    root = repo_root()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verify_static_contract(protocol)
    if not protocol["development_authorization"]["all_promotion_gates_pass"]:
        raise ValueError("Development result did not authorize fold-2 extraction")

    output = (root / protocol["fold2_cache"]["features"]).resolve()
    receipt_path = (root / protocol["fold2_cache"]["receipt"]).resolve()
    if receipt_path.exists():
        raise FileExistsError("Refusing to repeat finalized fold-2 extraction")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str((root / protocol["base_extractor"]["path"]).resolve()),
        "--folds",
        "2",
        "--batch-size",
        str(protocol["runtime"]["batch_size"]),
        "--workers",
        str(protocol["runtime"]["workers"]),
        "--output",
        protocol["fold2_cache"]["features"],
    ]
    resumed_existing_cache = output.exists()
    if not resumed_existing_cache:
        subprocess.run(command, cwd=root, check=True)
    stats = validate_cache(output)
    receipt = {
        "schema_version": 1,
        "scope": "one-shot MARS fold-2 DOFA-v2 scene-feature extraction",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "folds": [2],
        "fit_labels_accessed": False,
        "feature_extraction_uses_labels": False,
        "resumed_existing_unfinalized_cache": resumed_existing_cache,
        "protocol_sha256": sha256(protocol_path),
        "extractor_sha256": sha256(Path(__file__).resolve()),
        "base_extractor_sha256": sha256(
            (root / protocol["base_extractor"]["path"]).resolve()
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "command": command[1:],
        "features": {
            "path": protocol["fold2_cache"]["features"],
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        },
        "statistics": stats,
    }
    write_json(receipt_path, receipt)
    print(json.dumps({"ok": True, **receipt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
