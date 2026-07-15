#!/usr/bin/env python3
"""Extract a label-free, hash-pinned scene-feature cache for exact-paper replay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from diagnose_mars_scene_stacker_paper_cache import aligned_indices  # noqa: E402

DEFAULT_SOURCE = Path("outputs/mars_paper_test_v3_diagnostic_cache.npz")
DEFAULT_SOURCE_SHA256 = "1624fddc0222f8ffc5137f557c7fc3e465d53b335c82cc8014711baa35bb94a1"
DEFAULT_OUTPUT = Path("outputs/mars_paper_scene_features_label_free.npz")
DEFAULT_RECEIPT = Path("reports/acquisition/mars_paper_scene_features_label_free.json")
OUTPUT_FIELDS = (
    "sample_ids",
    "groups",
    "base_features",
    "base_feature_names",
    "current_v3_scores",
)
FORBIDDEN_TOKENS = ("label", "truth", "test_only", "pixel", "baseline")


def extract_payload(cache: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Copy only label-free arrays and align the frozen v3 score to available rows."""
    sample_ids = np.asarray(cache["available_ids"]).astype(str)
    groups = np.asarray(cache["available_groups"]).astype(str)
    base = np.asarray(cache["available_base_features"]).astype(np.float32)
    names = np.asarray(cache["base_feature_names"]).astype(str)
    aligned_ids = np.asarray(cache["aligned_sample_ids"]).astype(str)
    current = np.asarray(cache["candidate_scores"]).astype(np.float64)
    indices = aligned_indices(aligned_ids, sample_ids)
    if not (
        sample_ids.ndim == groups.ndim == 1
        and base.ndim == 2
        and names.ndim == 1
        and sample_ids.size == groups.size == base.shape[0]
        and names.size == base.shape[1]
        and current.shape == aligned_ids.shape
    ):
        raise ValueError("Paper scene feature source arrays do not align")
    if len(set(sample_ids.tolist())) != sample_ids.size or not np.isfinite(base).all():
        raise ValueError("Paper scene feature rows are invalid")
    payload = {
        "sample_ids": sample_ids,
        "groups": groups,
        "base_features": base,
        "base_feature_names": names,
        "current_v3_scores": current[indices],
    }
    if tuple(payload) != OUTPUT_FIELDS or any(
        token in name.lower() for name in payload for token in FORBIDDEN_TOKENS
    ):
        raise RuntimeError("Label-free output schema violated")
    return payload


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE.as_posix())
    parser.add_argument("--source-sha256", default=DEFAULT_SOURCE_SHA256)
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT.as_posix())
    args = parser.parse_args()
    root = repo_root()
    source = (root / args.source).resolve()
    output = (root / args.output).resolve()
    receipt = (root / args.receipt).resolve()
    if sha256(source) != args.source_sha256:
        raise ValueError("Frozen paper diagnostic source hash mismatch")
    # NPZ members are lazy: outcome arrays are not decompressed or accessed.
    with np.load(source, allow_pickle=False) as cache:
        payload = extract_payload(cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, output)
    with np.load(output, allow_pickle=False) as check:
        if tuple(check.files) != OUTPUT_FIELDS:
            raise RuntimeError("Written label-free cache schema differs")
        if any(token in name.lower() for name in check.files for token in FORBIDDEN_TOKENS):
            raise RuntimeError("Written cache contains a forbidden outcome field")
    report = {
        "schema_version": 1,
        "scope": "derived exact-paper scene inputs with no outcome or pixel-truth arrays",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(payload["sample_ids"].size),
        "groups": int(np.unique(payload["groups"]).size),
        "base_features": int(payload["base_features"].shape[1]),
        "output_fields": list(OUTPUT_FIELDS),
        "forbidden_tokens": list(FORBIDDEN_TOKENS),
        "source_sha256": args.source_sha256,
        "output_sha256": sha256(output),
        "script_sha256": sha256(Path(__file__).resolve()),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    }
    write_json(receipt, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
