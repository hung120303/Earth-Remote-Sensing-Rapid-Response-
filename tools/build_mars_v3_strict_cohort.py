#!/usr/bin/env python3
"""Freeze the minimum full strict-spatial MARS-S2L evaluation corpus."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from acquire_mars_metadata import (
    DEFAULT_OUTPUT,
    REPO_ID,
    REVISION,
    checked_output_dir,
    repo_root,
    sha256,
)
from build_mars_cohort import COHORT_MANIFEST
from build_mars_dev_cohort import DEV_CATALOG
from build_mars_protocol import ASSIGNMENTS_NAME, DEFAULT_PROTOCOL

V3_STRICT_SAMPLES = "publication_v3_strict_samples.jsonl"
V3_STRICT_CATALOG = "publication_v3_strict_remote_catalog.jsonl"
DEFAULT_JSON = Path("reports/acquisition/mars_s2l_v3_strict_cohort.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/MARS_S2L_V3_STRICT_COHORT.md")
DEV_RECEIPT = Path("reports/acquisition/mars_s2l_development_download.json")
SELECTED_ROLE = "strict_spatial_test"
EXCLUDED_ASSET_ROLE = "methane_enhancement"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def safe_report(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Report must resolve beneath the repository root")
    return result


def tracked_dirty(root: Path) -> bool:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
    )
    return bool(status.strip())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    transfer = report["transfer"]
    lines = [
        "# MARS-S2L full strict-spatial v3 evaluation transfer",
        "",
        "Frozen prevalence-representative strict-spatial evaluation corpus. It includes every official-test scene whose 25 km group contains no official-training scene; methane-enhancement rasters are excluded because detection evaluation does not consume them.",
        "",
        f"- Samples: {report['samples']['total']:,} ({report['samples']['positives']:,} plume / {report['samples']['negatives']:,} reviewed no-plume)",
        f"- Frozen 25 km groups: {report['samples']['groups']:,}",
        f"- Assets: {transfer['selected_asset_count']:,}; exact size: {transfer['selected_total_bytes']:,} bytes ({transfer['selected_total_bytes'] / 1024**3:.3f} GiB)",
        f"- Already verified in the development tranche: {transfer['reusable_verified_assets']:,} assets / {transfer['reusable_verified_bytes']:,} bytes",
        f"- Remaining transfer: {transfer['remaining_assets']:,} assets / {transfer['remaining_bytes']:,} bytes ({transfer['remaining_bytes'] / 1024**3:.3f} GiB)",
        f"- Sample manifest SHA-256: `{report['identities']['sample_manifest_sha256']}`",
        f"- Asset catalog SHA-256: `{report['identities']['asset_catalog_sha256']}`",
        "",
        "```powershell",
        "python tools/acquire_mars_cohort.py --catalog-file publication_v3_strict_remote_catalog.jsonl --dry-run",
        "python tools/acquire_mars_cohort.py --catalog-file publication_v3_strict_remote_catalog.jsonl --receipt reports/acquisition/mars_s2l_v3_strict_download.json",
        "python tools/acquire_mars_cohort.py --catalog-file publication_v3_strict_remote_catalog.jsonl --verify-only --receipt reports/acquisition/mars_s2l_v3_strict_download.json",
        "```",
        "",
        f"Raw files remain under `{report['destination']}` and are ignored by Git.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    metadata_dir = checked_output_dir(root, args.metadata_dir)
    cohort_path = metadata_dir / COHORT_MANIFEST
    assignments_path = metadata_dir / ASSIGNMENTS_NAME
    dev_catalog_path = metadata_dir / DEV_CATALOG
    protocol_path = root / DEFAULT_PROTOCOL
    for required in (
        cohort_path,
        assignments_path,
        dev_catalog_path,
        protocol_path,
        root / DEV_RECEIPT,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(cohort_path) != protocol["data"]["cohort_manifest_sha256"]:
        raise ValueError("Frozen cohort identity differs from the publication protocol")
    if sha256(assignments_path) != protocol["assignments"]["sha256"]:
        raise ValueError("Protocol assignment identity mismatch")
    receipt = json.loads((root / DEV_RECEIPT).read_text(encoding="utf-8"))
    if not receipt["verification"]["all_selected_assets_verified"]:
        raise ValueError("Development asset receipt is not fully verified")
    if receipt["catalog"]["sha256"] != sha256(dev_catalog_path):
        raise ValueError("Development catalog identity differs from its verification receipt")

    assignments = {
        str(row["sample_id"]): str(row["research_role"])
        for row in iter_jsonl(assignments_path)
    }
    selected: list[dict[str, Any]] = []
    catalog: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(cohort_path):
        if assignments.get(str(row["sample_id"])) != SELECTED_ROLE:
            continue
        retained = [
            asset for asset in row["assets"] if asset["role"] != EXCLUDED_ASSET_ROLE
        ]
        enriched = dict(row)
        enriched["research_role"] = SELECTED_ROLE
        enriched["assets"] = retained
        selected.append(enriched)
        for asset in retained:
            catalog.setdefault(str(asset["path"]), asset)

    expected = protocol["assignments"]["counts_by_role"][SELECTED_ROLE]
    labels = Counter(str(row["label_state"]) for row in selected)
    groups = {str(row["group_id"]) for row in selected}
    observed = {
        "rows": len(selected),
        "positive": labels["PLUME"],
        "negative": labels["NO_PLUME"],
        "groups": len(groups),
    }
    for key, value in observed.items():
        if int(value) != int(expected[key]):
            raise ValueError(f"Strict {key} mismatch: expected {expected[key]}, got {value}")

    sample_path = metadata_dir / V3_STRICT_SAMPLES
    catalog_path = metadata_dir / V3_STRICT_CATALOG
    write_jsonl(sample_path, sorted(selected, key=lambda row: str(row["sample_id"])))
    write_jsonl(catalog_path, [catalog[path] for path in sorted(catalog)])

    reusable_paths = {str(row["path"]) for row in iter_jsonl(dev_catalog_path)} & set(catalog)
    reusable_bytes = sum(int(catalog[path]["size"]) for path in reusable_paths)
    total_bytes = sum(int(asset["size"]) for asset in catalog.values())
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "minimum_full_mars_s2l_v3_strict_spatial_evaluation_corpus",
        "source": {
            "repository": REPO_ID,
            "revision": REVISION,
            "cohort_manifest_sha256": sha256(cohort_path),
            "protocol_sha256": sha256(protocol_path),
            "protocol_assignments_sha256": sha256(assignments_path),
            "development_verification_receipt_sha256": sha256(root / DEV_RECEIPT),
        },
        "selection": {
            "research_role": SELECTED_ROLE,
            "retained_asset_roles": ["image", "cloud_mask", "plume_mask"],
            "excluded_asset_roles": [EXCLUDED_ASSET_ROLE],
            "class_enriched": False,
            "paper_claim_scope": True,
        },
        "samples": {
            "total": len(selected),
            "positives": labels["PLUME"],
            "negatives": labels["NO_PLUME"],
            "groups": len(groups),
        },
        "transfer": {
            "selected_asset_count": len(catalog),
            "selected_total_bytes": total_bytes,
            "reusable_verified_assets": len(reusable_paths),
            "reusable_verified_bytes": reusable_bytes,
            "remaining_assets": len(catalog) - len(reusable_paths),
            "remaining_bytes": total_bytes - reusable_bytes,
        },
        "destination": metadata_dir.relative_to(root).as_posix(),
        "identities": {
            "sample_manifest": sample_path.relative_to(root).as_posix(),
            "sample_manifest_sha256": sha256(sample_path),
            "asset_catalog": catalog_path.relative_to(root).as_posix(),
            "asset_catalog_sha256": sha256(catalog_path),
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": "tools/build_mars_v3_strict_cohort.py",
            "script_sha256": sha256(Path(__file__)),
        },
    }
    output_json = safe_report(root, args.output_json)
    output_markdown = safe_report(root, args.output_markdown)
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
