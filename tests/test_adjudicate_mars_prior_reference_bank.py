from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.adjudicate_mars_prior_reference_bank import audit_selection_manifest


def row(
    sample_id: str,
    *,
    sensor: str = "Sentinel-2",
    selected: list[str] | None = None,
) -> dict[str, object]:
    selected = selected or []
    return {
        "exact_grid_candidates": len(selected),
        "fallback_to_original_only": not selected,
        "fold": 3,
        "grid_key": "S2:32SKA" if sensor == "Sentinel-2" else "Landsat:190030",
        "original_reference_distance": 0.1,
        "physical_location_id": "site",
        "recent_pool_candidates": len(selected),
        "sample_id": sample_id,
        "selected_distances": [0.1] * len(selected),
        "selected_sample_ids": selected,
        "selected_target_scene_ids": [f"scene-{value}" for value in selected],
        "sensor_family": sensor,
        "strictly_prior_clear_candidates": len(selected),
        "target_datetime": "2020-01-01 00:00:00+00:00",
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8"
    )


def test_manifest_audit_counts_references_and_fallback(tmp_path: Path) -> None:
    path = tmp_path / "selection.jsonl"
    write_rows(
        path,
        [
            row("a", selected=["p1", "p2", "p3", "p4", "p5"]),
            row("b"),
            row("c", sensor="Landsat"),
        ],
    )
    counts = audit_selection_manifest(
        path,
        allowed_folds={3, 4},
        maximum_selected_references=5,
        maximum_recent_pool=10,
    )
    assert counts == {
        "rows": 3,
        "sentinel_rows": 2,
        "sentinel_rows_with_selected_reference": 1,
        "sentinel_rows_with_five_selected_references": 1,
        "landsat_rows": 1,
        "fallback_rows": 2,
    }


def test_manifest_audit_rejects_schema_drift(tmp_path: Path) -> None:
    path = tmp_path / "selection.jsonl"
    unsafe = row("a")
    unsafe["label"] = 1
    write_rows(path, [unsafe])
    with pytest.raises(ValueError, match="Unsafe manifest schema"):
        audit_selection_manifest(
            path,
            allowed_folds={3, 4},
            maximum_selected_references=5,
            maximum_recent_pool=10,
        )


def test_manifest_audit_rejects_landsat_alternate_reference(tmp_path: Path) -> None:
    path = tmp_path / "selection.jsonl"
    write_rows(path, [row("a", sensor="Landsat", selected=["p1"])])
    with pytest.raises(ValueError, match="Landsat alternate reference"):
        audit_selection_manifest(
            path,
            allowed_folds={3, 4},
            maximum_selected_references=5,
            maximum_recent_pool=10,
        )
