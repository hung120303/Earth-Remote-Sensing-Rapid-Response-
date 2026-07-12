#!/usr/bin/env python3
"""Audit embedded versus frozen-manifest MARS image band declarations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import (  # noqa: E402
    iter_manifest,
    role_paths,
    safe_asset_path,
    validate_image_band_order,
)

from acquire_mars_metadata import DEFAULT_OUTPUT, checked_output_dir, repo_root, sha256  # noqa: E402
from build_mars_v3_strict_cohort import V3_STRICT_SAMPLES  # noqa: E402
from build_mars_v3_training_cohort import V3_SAMPLES  # noqa: E402

DEFAULT_JSON = Path("reports/acquisition/mars_s2l_band_description_audit.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/MARS_S2L_BAND_DESCRIPTION_AUDIT.md")


def safe_output(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Audit output must resolve beneath the repository root")
    return result


def inspect_record(metadata_dir: Path, record: dict[str, Any]) -> dict[str, str]:
    image = safe_asset_path(metadata_dir, role_paths(record)["image"])
    with rasterio.open(image) as source:
        descriptions = tuple(source.descriptions)
    authority = validate_image_band_order(record, descriptions)
    return {
        "sample_id": str(record["sample_id"]),
        "research_role": str(record["research_role"]),
        "country": str(record.get("country") or "unknown"),
        "image": image.relative_to(metadata_dir).as_posix(),
        "band_order_authority": authority,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    counts = report["results"]["band_order_authority"]
    fallback = report["results"]["frozen_manifest_fallback"]
    lines = [
        "# MARS-S2L image band-description audit",
        "",
        f"- Images inspected: {report['cohort']['images']:,}",
        f"- Embedded 12-band descriptions: {counts.get('embedded_descriptions', 0):,}",
        f"- All descriptions absent; exact frozen manifest declaration used: {counts.get('frozen_manifest_declaration', 0):,}",
        f"- Contract failures: {report['results']['contract_failures']:,}",
        "",
        "The fallback accepts only the producer omission where all 12 TIFF descriptions are absent and the hash-bound manifest declares the exact expected order. Partial, mixed, or conflicting labels remain fatal.",
    ]
    if fallback:
        lines.extend(["", "## Manifest-fallback samples", ""])
        lines.extend(
            f"- `{item['sample_id']}` — {item['research_role']} / {item['country']}"
            for item in fallback
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    root = repo_root()
    metadata_dir = checked_output_dir(root, args.metadata_dir)
    manifests = [metadata_dir / V3_SAMPLES, metadata_dir / V3_STRICT_SAMPLES]
    records: list[dict[str, Any]] = []
    observed: set[str] = set()
    for manifest in manifests:
        for record in iter_manifest(manifest):
            sample_id = str(record["sample_id"])
            if sample_id in observed:
                raise ValueError(f"Duplicate sample across audit manifests: {sample_id}")
            observed.add(sample_id)
            records.append(record)

    failures: list[dict[str, str]] = []

    def checked(record: dict[str, Any]) -> dict[str, str]:
        try:
            return inspect_record(metadata_dir, record)
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(
                {"sample_id": str(record["sample_id"]), "error": str(exc)}
            )
            return {"band_order_authority": "contract_failure"}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(checked, records))
    counts = Counter(item["band_order_authority"] for item in results)
    fallback = sorted(
        (
            item
            for item in results
            if item["band_order_authority"] == "frozen_manifest_declaration"
        ),
        key=lambda item: item["sample_id"],
    )
    report = {
        "schema_version": 1,
        "scope": "full_v3_training_validation_and_strict_image_header_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "images": len(records),
            "manifests": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256(path),
                }
                for path in manifests
            ],
        },
        "results": {
            "band_order_authority": dict(sorted(counts.items())),
            "contract_failures": len(failures),
            "failures": failures,
            "frozen_manifest_fallback": fallback,
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_tracked_worktree_dirty_at_start": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"],
                    cwd=root,
                    text=True,
                ).strip()
            ),
            "script": "tools/audit_mars_band_descriptions.py",
            "script_sha256": sha256(Path(__file__)),
            "adapter_sha256": sha256(MODEL_ROOT / "mars_s2l_adapter.py"),
            "rasterio": rasterio.__version__,
        },
    }
    output_json = safe_output(root, args.output_json)
    output_markdown = safe_output(root, args.output_markdown)
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
