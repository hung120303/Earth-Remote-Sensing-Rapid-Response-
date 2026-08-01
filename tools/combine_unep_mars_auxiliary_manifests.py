#!/usr/bin/env python3
"""Combine append-only UNEP auxiliary model manifests with strict identity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def repo_root() -> Path:
    value = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    return Path(value).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine_manifests(
    manifests: Iterable[Iterable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for rows in manifests:
        for row in rows:
            sample_id = str(row["sample_id"])
            if row.get("research_role") != "auxiliary_training":
                raise ValueError(f"Non-auxiliary row reached combine: {sample_id}")
            if row.get("label_state") != "PLUME":
                raise ValueError(f"Non-plume row reached combine: {sample_id}")
            if row.get("sensor_family") != "Sentinel-2":
                raise ValueError(f"Unexpected sensor reached combine: {sample_id}")
            existing = by_id.get(sample_id)
            if existing is not None and existing != row:
                raise ValueError(f"Conflicting duplicate sample identity: {sample_id}")
            by_id[sample_id] = row
    rows = sorted(by_id.values(), key=lambda row: str(row["sample_id"]))
    for row in rows:
        if row.get("physical_location_id") != row.get("group_id"):
            raise ValueError(f"Physical group mismatch: {row['sample_id']}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    root = repo_root()
    inputs = [(root / value).resolve() for value in args.input]
    rows_by_input = [read_jsonl(path) for path in inputs]
    rows = combine_manifests(rows_by_input)
    output = (root / args.output).resolve()
    write_jsonl(output, rows)
    receipt = {
        "schema_version": 1,
        "status": "append_only_auxiliary_model_manifest_combined",
        "inputs": [
            {
                "path": path.relative_to(root).as_posix(),
                "rows": len(input_rows),
                "sha256": sha256(path),
            }
            for path, input_rows in zip(inputs, rows_by_input)
        ],
        "output": {
            "path": output.relative_to(root).as_posix(),
            "rows": len(rows),
            "groups": len({str(row["group_id"]) for row in rows}),
            "rows_by_satellite": dict(
                sorted(Counter(str(row["satellite"]) for row in rows).items())
            ),
            "sha256": sha256(output),
            "bytes": output.stat().st_size,
        },
    }
    receipt_path = (root / args.receipt).resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
    print(json.dumps(receipt["output"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
