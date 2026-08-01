#!/usr/bin/env python3
"""Reconcile a growing UNEP catalog against immutable prior group roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREVIOUS = Path(".research/unep_mars_post2024/eligible_manifest.jsonl")
DEFAULT_CURRENT = Path(
    ".research/unep_mars_post2024_refresh_20260801/eligible_manifest.jsonl"
)
DEFAULT_OUTPUT = Path(
    ".research/unep_mars_post2024_refresh_20260801/eligible_manifest_reconciled.jsonl"
)
DEFAULT_JSON = Path(
    ".research/unep_mars_post2024_refresh_20260801/role_reconciliation.json"
)
DEFAULT_MARKDOWN = Path(
    ".research/unep_mars_post2024_refresh_20260801/ROLE_RECONCILIATION.md"
)
QUARANTINE_ROLE = "quarantined_role_conflict"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    identities = [str(row["sample_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"Duplicate sample identity in {path}")
    for row in rows:
        if not row.get("group_id") or not row.get("research_role"):
            raise ValueError(f"Missing group/role contract in {path}")
    return rows


def reconcile_roles(
    previous: list[dict[str, Any]], current: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior_by_id = {str(row["sample_id"]): row for row in previous}
    current_ids = {str(row["sample_id"]) for row in current}
    missing_previous = sorted(set(prior_by_id) - current_ids)
    if missing_previous:
        raise ValueError(
            f"Refresh omitted {len(missing_previous)} prior identities; append-only audit failed"
        )

    by_current_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in current:
        by_current_group[str(row["group_id"])].append(row)

    reconciled: list[dict[str, Any]] = []
    component_audit: list[dict[str, Any]] = []
    for computed_group, members in by_current_group.items():
        overlaps = [
            prior_by_id[str(row["sample_id"])]
            for row in members
            if str(row["sample_id"]) in prior_by_id
        ]
        prior_groups = sorted({str(row["group_id"]) for row in overlaps})
        prior_roles = sorted({str(row["research_role"]) for row in overlaps})
        if not overlaps:
            stable_group = computed_group
            stable_role = str(members[0]["research_role"])
            decision = "new_component_uses_prospective_hash"
        elif len(prior_groups) == 1 and len(prior_roles) == 1:
            stable_group = prior_groups[0]
            stable_role = prior_roles[0]
            decision = "inherit_immutable_prior_group_and_role"
        else:
            stable_group = computed_group
            stable_role = QUARANTINE_ROLE
            decision = "quarantine_component_merging_prior_groups_or_roles"

        computed_roles = sorted({str(row["research_role"]) for row in members})
        for row in members:
            updated = dict(row)
            updated["refresh_computed_group_id"] = str(row["group_id"])
            updated["refresh_computed_research_role"] = str(row["research_role"])
            updated["group_id"] = stable_group
            updated["research_role"] = stable_role
            updated["role_reconciliation"] = decision
            reconciled.append(updated)
        component_audit.append(
            {
                "computed_group_id": computed_group,
                "computed_roles": computed_roles,
                "members": len(members),
                "prior_members": len(overlaps),
                "new_members": len(members) - len(overlaps),
                "prior_group_ids": prior_groups,
                "prior_roles": prior_roles,
                "stable_group_id": stable_group,
                "stable_role": stable_role,
                "decision": decision,
            }
        )

    reconciled.sort(key=lambda row: str(row["sample_id"]))
    old_identity_changes = 0
    for row in reconciled:
        prior = prior_by_id.get(str(row["sample_id"]))
        if prior is None:
            continue
        if (
            str(row["group_id"]) != str(prior["group_id"])
            or str(row["research_role"]) != str(prior["research_role"])
        ):
            old_identity_changes += 1
    if old_identity_changes:
        raise ValueError(
            f"Reconciliation changed {old_identity_changes} immutable prior identities"
        )

    summary = {
        "previous_rows": len(previous),
        "current_rows": len(current),
        "overlap_rows": len(previous),
        "new_rows": len(current) - len(previous),
        "old_identity_group_or_role_changes_after_reconciliation": old_identity_changes,
        "by_role": dict(sorted(Counter(str(row["research_role"]) for row in reconciled).items())),
        "groups_by_role": {
            role: len({str(row["group_id"]) for row in reconciled if row["research_role"] == role})
            for role in sorted({str(row["research_role"]) for row in reconciled})
        },
        "component_count": len(component_audit),
        "inherited_components": sum(
            row["decision"] == "inherit_immutable_prior_group_and_role"
            for row in component_audit
        ),
        "new_components": sum(
            row["decision"] == "new_component_uses_prospective_hash"
            for row in component_audit
        ),
        "quarantined_components": sum(
            row["stable_role"] == QUARANTINE_ROLE for row in component_audit
        ),
        "components": component_audit,
    }
    return reconciled, summary


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# UNEP refresh role reconciliation",
        "",
        "Catalog growth is reconciled against the immutable prior manifest before acquisition. Existing identities retain their prior group and role; new members of one prior component inherit it; ambiguous merges are quarantined.",
        "",
        f"- Prior/current/new rows: **{summary['previous_rows']} / {summary['current_rows']} / {summary['new_rows']}**",
        f"- Prior identity changes after reconciliation: **{summary['old_identity_group_or_role_changes_after_reconciliation']}**",
        f"- Reconciled roles: **{json.dumps(summary['by_role'], sort_keys=True)}**",
        f"- Quarantined components: **{summary['quarantined_components']}**",
        f"- Output SHA-256: `{report['output']['sha256']}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", default=DEFAULT_PREVIOUS.as_posix())
    parser.add_argument("--current", default=DEFAULT_CURRENT.as_posix())
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    previous_path = (ROOT / args.previous).resolve()
    current_path = (ROOT / args.current).resolve()
    output_path = (ROOT / args.output).resolve()
    json_path = (ROOT / args.output_json).resolve()
    markdown_path = (ROOT / args.output_markdown).resolve()

    rows, summary = reconcile_roles(
        read_jsonl(previous_path), read_jsonl(current_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output_path)
    report = {
        "schema_version": 1,
        "status": "append_only_roles_reconciled",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "previous": {"path": args.previous, "sha256": sha256(previous_path)},
            "current": {"path": args.current, "sha256": sha256(current_path)},
        },
        "output": {
            "path": args.output,
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        },
        "summary": summary,
        "sealed_external_accessed": False,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_json, json_path)
    write_markdown(markdown_path, report)
    compact_summary = {
        key: value for key, value in summary.items() if key != "components"
    }
    print(json.dumps({"ok": True, **compact_summary}, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
