#!/usr/bin/env python3
"""Build a group-diverse MARS-S2L development tranche from frozen roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from acquire_mars_metadata import DEFAULT_OUTPUT, REPO_ID, REVISION, checked_output_dir, repo_root, sha256
from build_mars_cohort import COHORT_MANIFEST, REMOTE_CATALOG
from build_mars_protocol import ASSIGNMENTS_NAME, DEFAULT_PROTOCOL

DEV_SAMPLES = "publication_dev_samples.jsonl"
DEV_CATALOG = "publication_dev_remote_catalog.jsonl"
DEFAULT_JSON = Path("reports/acquisition/mars_s2l_development_cohort.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/MARS_S2L_DEVELOPMENT_COHORT.md")
SELECTION_SEED = 20260711
TARGETS: dict[tuple[str, str], int] = {
    ("internal_training", "PLUME"): 256,
    ("internal_training", "NO_PLUME"): 512,
    ("internal_validation", "PLUME"): 128,
    ("internal_validation", "NO_PLUME"): 256,
    ("strict_spatial_test", "PLUME"): 67,
    ("strict_spatial_test", "NO_PLUME"): 512,
}


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def tracked_dirty(root: Path) -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    )
    return bool(output.strip())


def safe_output(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Development-cohort report must resolve beneath the repository root")
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path.name} at line {line_number}") from exc
    return rows


def stable_rank(value: str, role: str, label: str) -> str:
    payload = f"{SELECTION_SEED}\0{role}\0{label}\0{value}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def group_round_robin(
    candidates: list[dict[str, Any]], role: str, label: str, target: int
) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_group[row["group_id"]].append(row)
    for group, values in by_group.items():
        values.sort(key=lambda row: stable_rank(row["sample_id"], role, label))
    ordered_groups = sorted(
        by_group,
        key=lambda group: stable_rank(group, role, label),
    )
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < target:
        added = False
        for group in ordered_groups:
            values = by_group[group]
            if depth < len(values):
                selected.append(values[depth])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != target:
        raise ValueError(
            f"Only selected {len(selected)} of {target} requested {role}/{label} samples"
        )
    return selected


def select_samples(
    cohort: list[dict[str, Any]], assignments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    role_by_id = {row["sample_id"]: row["research_role"] for row in assignments}
    if len(role_by_id) != len(assignments):
        raise ValueError("Protocol assignments contain duplicate sample IDs")
    eligible: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cohort:
        role = role_by_id.get(row["sample_id"])
        key = (role, row["label_state"])
        if key in TARGETS:
            enriched = dict(row)
            enriched["research_role"] = role
            eligible[key].append(enriched)
    selected: list[dict[str, Any]] = []
    for (role, label), target in TARGETS.items():
        selected.extend(group_round_robin(eligible[(role, label)], role, label, target))
    selected.sort(key=lambda row: (row["research_role"], row["label_state"], row["sample_id"]))
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError("Development cohort contains duplicate samples")
    return selected


def selected_asset_paths(rows: list[dict[str, Any]]) -> set[str]:
    paths = {asset["path"] for row in rows for asset in row["assets"]}
    expected = sum(4 if row["label_state"] == "PLUME" else 2 for row in rows)
    if len(paths) != expected:
        raise ValueError(
            f"Development assets are unexpectedly shared: {len(paths)} unique vs {expected} references"
        )
    return paths


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({(row["research_role"], row["label_state"]) for row in rows})
    by_role_label = {
        f"{role}:{label}": {
            "rows": sum(
                row["research_role"] == role and row["label_state"] == label for row in rows
            ),
            "groups": len(
                {
                    row["group_id"]
                    for row in rows
                    if row["research_role"] == role and row["label_state"] == label
                }
            ),
            "locations": len(
                {
                    row["physical_location_id"]
                    for row in rows
                    if row["research_role"] == role and row["label_state"] == label
                }
            ),
        }
        for role, label in keys
    }
    return {
        "sample_count": len(rows),
        "positive_samples": sum(row["label_state"] == "PLUME" for row in rows),
        "negative_samples": sum(row["label_state"] == "NO_PLUME" for row in rows),
        "groups": len({row["group_id"] for row in rows}),
        "locations": len({row["physical_location_id"] for row in rows}),
        "by_role_label": by_role_label,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    cohort = report["cohort"]
    assets = report["assets"]
    lines = [
        "# MARS-S2L development cohort",
        "",
        f"- Samples: {cohort['sample_count']:,} ({cohort['positive_samples']:,} plume / {cohort['negative_samples']:,} no plume)",
        f"- Groups: {cohort['groups']:,}; locations: {cohort['locations']:,}",
        f"- Assets: {assets['count']:,}; exact size: {assets['total_bytes']:,} bytes ({assets['binary_gib']:.3f} GiB)",
        f"- Sample-manifest SHA-256: `{report['identities']['sample_manifest_sha256']}`",
        f"- Asset-catalog SHA-256: `{report['identities']['asset_catalog_sha256']}`",
        "",
        "| Role | Label | Rows | Groups | Locations |",
        "|---|---|---:|---:|---:|",
    ]
    for key, item in cohort["by_role_label"].items():
        role, label = key.split(":", 1)
        lines.append(
            f"| {role} | {label} | {item['rows']:,} | {item['groups']:,} | {item['locations']:,} |"
        )
    lines.extend(
        [
            "",
            "This is a group-diverse development tranche for baseline and pipeline iteration. It preserves the frozen internal train/validation and strict spatial test roles, but it is deliberately class-enriched and is not the paper's prevalence-representative final cohort.",
            "",
            "Download and verify only this tranche with:",
            "",
            "```bash",
            f"python tools/acquire_mars_cohort.py --catalog-file {DEV_CATALOG}",
            f"python tools/acquire_mars_cohort.py --catalog-file {DEV_CATALOG} --verify-only",
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.verify_only and args.dry_run:
        parser.error("--verify-only and --dry-run are mutually exclusive")
    root = repo_root()
    try:
        metadata_dir = checked_output_dir(root, args.metadata_dir)
        cohort_path = metadata_dir / COHORT_MANIFEST
        assignments_path = metadata_dir / ASSIGNMENTS_NAME
        source_catalog_path = metadata_dir / REMOTE_CATALOG
        protocol_path = root / DEFAULT_PROTOCOL
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if sha256(cohort_path) != protocol["data"]["cohort_manifest_sha256"]:
            raise ValueError("Frozen cohort identity does not match the publication protocol")
        if sha256(assignments_path) != protocol["assignments"]["sha256"]:
            raise ValueError("Assignment identity does not match the publication protocol")
        cohort = read_jsonl(cohort_path)
        assignments = read_jsonl(assignments_path)
        selected = select_samples(cohort, assignments)
        paths = selected_asset_paths(selected)
        source_catalog = read_jsonl(source_catalog_path)
        catalog_by_path = {item["path"]: item for item in source_catalog}
        missing = paths - set(catalog_by_path)
        if missing:
            raise ValueError(f"Remote catalog is missing {len(missing)} selected assets")
        dev_catalog = [catalog_by_path[path] for path in sorted(paths)]
        dev_samples_path = metadata_dir / DEV_SAMPLES
        dev_catalog_path = metadata_dir / DEV_CATALOG
        expected_sample_identity = hashlib.sha256(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in selected
            ).encode("utf-8")
        ).hexdigest()
        expected_catalog_identity = hashlib.sha256(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in dev_catalog
            ).encode("utf-8")
        ).hexdigest()
        if args.verify_only:
            if sha256(dev_samples_path) != expected_sample_identity:
                raise ValueError("Development sample manifest failed deterministic verification")
            if sha256(dev_catalog_path) != expected_catalog_identity:
                raise ValueError("Development asset catalog failed deterministic verification")
            payload = {
                "ok": True,
                "verify_only": True,
                "sample_count": len(selected),
                "asset_count": len(dev_catalog),
                "sample_manifest_sha256": expected_sample_identity,
                "asset_catalog_sha256": expected_catalog_identity,
            }
            print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
            return 0
        summary = summarize_rows(selected)
        total_bytes = sum(int(item["size"]) for item in dev_catalog)
        if args.dry_run:
            payload = {
                "ok": True,
                "dry_run": True,
                "cohort": summary,
                "asset_count": len(dev_catalog),
                "total_bytes": total_bytes,
                "binary_gib": round(total_bytes / 1024**3, 3),
                "sample_manifest_sha256": expected_sample_identity,
                "asset_catalog_sha256": expected_catalog_identity,
            }
            print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
            return 0
        output_json = safe_output(root, args.output_json)
        output_markdown = safe_output(root, args.output_markdown)
        write_jsonl(dev_samples_path, selected)
        write_jsonl(dev_catalog_path, dev_catalog)
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "repository": REPO_ID,
                "revision": REVISION,
                "protocol_sha256": sha256(protocol_path),
                "assignment_sha256": protocol["assignments"]["sha256"],
            },
            "selection": {
                "seed": SELECTION_SEED,
                "method": "stable hash rank with round-robin sampling across frozen 25 km groups",
                "targets": {f"{role}:{label}": value for (role, label), value in TARGETS.items()},
                "class_enriched": True,
                "paper_claim_scope": False,
            },
            "cohort": summary,
            "assets": {
                "count": len(dev_catalog),
                "total_bytes": total_bytes,
                "decimal_gb": round(total_bytes / 1_000_000_000, 3),
                "binary_gib": round(total_bytes / 1024**3, 3),
            },
            "local_ignored_artifacts": {
                "sample_manifest": dev_samples_path.relative_to(root).as_posix(),
                "asset_catalog": dev_catalog_path.relative_to(root).as_posix(),
            },
            "identities": {
                "sample_manifest_sha256": sha256(dev_samples_path),
                "asset_catalog_sha256": sha256(dev_catalog_path),
            },
            "provenance": {
                "git_commit": git_commit(root),
                "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
                "script": "tools/build_mars_dev_cohort.py",
                "script_sha256": sha256(Path(__file__)),
                "python": sys.version.split()[0],
            },
        }
        write_json(output_json, report)
        write_markdown(output_markdown, report)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=None if args.compact else 2))
        return 2
    payload = {
        "ok": True,
        "dry_run": False,
        "sample_count": report["cohort"]["sample_count"],
        "asset_count": report["assets"]["count"],
        "total_bytes": report["assets"]["total_bytes"],
        "binary_gib": report["assets"]["binary_gib"],
        "output_json": output_json.relative_to(root).as_posix(),
        "output_markdown": output_markdown.relative_to(root).as_posix(),
    }
    print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
