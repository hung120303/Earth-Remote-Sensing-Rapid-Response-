from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from reconcile_unep_mars_refresh_roles import (  # noqa: E402
    QUARANTINE_ROLE,
    reconcile_roles,
)


def row(sample: str, group: str, role: str) -> dict[str, str]:
    return {"sample_id": sample, "group_id": group, "research_role": role}


def test_existing_component_and_new_members_inherit_prior_contract() -> None:
    previous = [row("old", "stable", "auxiliary_training")]
    current = [
        row("old", "recomputed", "development"),
        row("new", "recomputed", "development"),
    ]
    reconciled, summary = reconcile_roles(previous, current)
    assert {(item["sample_id"], item["group_id"], item["research_role"]) for item in reconciled} == {
        ("old", "stable", "auxiliary_training"),
        ("new", "stable", "auxiliary_training"),
    }
    assert summary["old_identity_group_or_role_changes_after_reconciliation"] == 0


def test_brand_new_component_keeps_prospective_assignment() -> None:
    reconciled, _ = reconcile_roles(
        [row("old", "stable", "development")],
        [
            row("old", "stable", "development"),
            row("new", "new_group", "auxiliary_training"),
        ],
    )
    selected = next(item for item in reconciled if item["sample_id"] == "new")
    assert selected["group_id"] == "new_group"
    assert selected["research_role"] == "auxiliary_training"


def test_component_merging_prior_groups_is_quarantined_and_rejected() -> None:
    previous = [
        row("a", "group_a", "auxiliary_training"),
        row("b", "group_b", "development"),
    ]
    current = [
        row("a", "merged", "auxiliary_training"),
        row("b", "merged", "auxiliary_training"),
    ]
    with pytest.raises(ValueError, match="immutable prior identities"):
        reconcile_roles(previous, current)


def test_refresh_must_be_append_only() -> None:
    with pytest.raises(ValueError, match="append-only"):
        reconcile_roles(
            [row("a", "group_a", "auxiliary_training")],
            [row("b", "group_b", "auxiliary_training")],
        )
