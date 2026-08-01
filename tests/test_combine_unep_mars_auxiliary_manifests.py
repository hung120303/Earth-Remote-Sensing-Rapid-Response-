from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from combine_unep_mars_auxiliary_manifests import combine_manifests  # noqa: E402


def row(sample_id: str, group_id: str = "g1") -> dict:
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "physical_location_id": group_id,
        "research_role": "auxiliary_training",
        "label_state": "PLUME",
        "sensor_family": "Sentinel-2",
    }


def test_combines_append_only_rows_in_identity_order() -> None:
    assert [value["sample_id"] for value in combine_manifests([[row("b")], [row("a")]])] == ["a", "b"]


def test_accepts_identical_duplicate_but_rejects_conflict() -> None:
    assert len(combine_manifests([[row("a")], [row("a")]])) == 1
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        combine_manifests([[row("a")], [row("a", "g2")]])


@pytest.mark.parametrize("field,value", [("research_role", "development"), ("label_state", "NO_PLUME"), ("sensor_family", "Landsat")])
def test_rejects_out_of_contract_rows(field: str, value: str) -> None:
    invalid = row("a")
    invalid[field] = value
    with pytest.raises(ValueError):
        combine_manifests([[invalid]])
