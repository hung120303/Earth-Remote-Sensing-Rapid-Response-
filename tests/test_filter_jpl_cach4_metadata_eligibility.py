from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import tools.filter_jpl_cach4_metadata_eligibility as eligibility


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _negative(sample_id: str, latitude: float, longitude: float) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "sensor": "AVIRIS-NG",
        "tile": "flight",
        "timestamp": "2018-01-01T00:00:00Z",
        "latitude": latitude,
        "longitude": longitude,
        "label_state": "NO_PLUME",
        "published_split": "train",
        "coordinate_resolved": True,
        "eligible_for_target_catalog": False,
    }


def test_exclusion_threshold_is_inclusive_at_exactly_25km() -> None:
    assert eligibility.within_exclusion_radius(25.0, 25.0) is True
    assert eligibility.within_exclusion_radius(
        math.nextafter(25.0, math.inf), 25.0
    ) is False


def test_eligible_group_ids_are_stable_transitive_components() -> None:
    rows = [
        _negative("c", 0.0, 0.40),
        _negative("a", 0.0, 0.0),
        _negative("far", 5.0, 5.0),
        _negative("b", 0.0, 0.20),
    ]
    forward = eligibility.filter_rows(
        rows=rows,
        all_mars_locations={"mars": (20.0, 20.0)},
        protected_mars_locations={"test": (30.0, 30.0)},
        prior_negative_coordinates={"prior": (-30.0, -30.0)},
        radius_km=25.0,
    )
    reverse = eligibility.filter_rows(
        rows=list(reversed(rows)),
        all_mars_locations={"mars": (20.0, 20.0)},
        protected_mars_locations={"test": (30.0, 30.0)},
        prior_negative_coordinates={"prior": (-30.0, -30.0)},
        radius_km=25.0,
    )
    forward_groups = {str(row["sample_id"]): row["group_id"] for row in forward}
    reverse_groups = {str(row["sample_id"]): row["group_id"] for row in reverse}
    assert forward_groups == reverse_groups
    assert forward_groups["a"] == forward_groups["b"] == forward_groups["c"]
    assert forward_groups["far"] != forward_groups["a"]


def test_prior_pair_coordinates_are_bound_to_reported_identities(
    tmp_path: Path,
) -> None:
    pair_path = tmp_path / "pairs.jsonl"
    mask_path = tmp_path / "masks.jsonl"
    report_path = tmp_path / "report.json"
    _jsonl(
        pair_path,
        [
            {"sample_id": "negative", "label_state": "NO_PLUME"},
            {"sample_id": "positive", "label_state": "PLUME"},
        ],
    )
    _jsonl(
        mask_path,
        [
            {
                "sample_id": "negative",
                "label_state": "NO_PLUME",
                "latitude": 1.0,
                "longitude": 2.0,
            },
            {
                "sample_id": "positive",
                "label_state": "PLUME",
                "latitude": 3.0,
                "longitude": 4.0,
            },
        ],
    )
    report_path.write_text(
        json.dumps(
            {
                "ignored_pair_catalog": {
                    "path": pair_path.as_posix(),
                    "sha256": eligibility.sha256_file(pair_path),
                },
                "inputs": {
                    "mask_catalog": {
                        "path": mask_path.as_posix(),
                        "sha256": eligibility.sha256_file(mask_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    coordinates, counts = eligibility.load_prior_negative_coordinates(
        stage_b_report_path=report_path,
        pair_catalog_path=pair_path,
        mask_catalog_path=mask_path,
    )
    assert coordinates == {"negative": (1.0, 2.0)}
    assert counts == {"negative_pair_rows": 1, "unique_negative_source_samples": 1}

    pair_path.write_text(pair_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        eligibility.load_prior_negative_coordinates(
            stage_b_report_path=report_path,
            pair_catalog_path=pair_path,
            mask_catalog_path=mask_path,
        )


def test_recorded_relative_path_accepts_equivalent_absolute_repository_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "ignored" / "artifact.jsonl"
    artifact.parent.mkdir()
    artifact.write_text('{"sample_id":"a"}\n', encoding="utf-8")
    monkeypatch.setattr(eligibility, "ROOT", tmp_path)
    eligibility._require_recorded_file(
        path=artifact.resolve(),
        recorded={
            "path": "ignored/artifact.jsonl",
            "sha256": eligibility.sha256_file(artifact),
        },
        role="Fixture",
    )


def test_pair_receipt_normalization_is_stable_across_crlf(tmp_path: Path) -> None:
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    payload = b'{"sample_id":"a"}\n{"sample_id":"b"}\n'
    lf.write_bytes(payload)
    crlf.write_bytes(payload.replace(b"\n", b"\r\n"))
    assert eligibility.sha256_file(lf) != eligibility.sha256_file(crlf)
    assert eligibility.normalized_jsonl_sha256(lf) == (
        eligibility.normalized_jsonl_sha256(crlf)
    )


def test_only_committed_filter_protocol_path_is_accepted(tmp_path: Path) -> None:
    alternative = tmp_path / "protocol.json"
    alternative.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="committed frozen CACH4 filter protocol"):
        eligibility.validate_filter_protocol_path(alternative)
    eligibility.validate_filter_protocol_path(eligibility.EXPECTED_FILTER_PROTOCOL)


def test_location_gate_counts_components_not_rows_or_flightlines() -> None:
    assert eligibility.conservative_component_gate(
        component_count=19, required_components=20
    ) is False
    assert eligibility.conservative_component_gate(
        component_count=20, required_components=20
    ) is True


def test_component_novelty_requires_every_member_to_be_novel() -> None:
    rows = [_negative("near", 0.0, 0.20), _negative("farther", 0.0, 0.40)]
    filtered = eligibility.filter_rows(
        rows=rows,
        all_mars_locations={"mars": (0.0, 0.0)},
        protected_mars_locations={"test": (30.0, 30.0)},
        prior_negative_coordinates={"prior": (-30.0, -30.0)},
        radius_km=25.0,
    )
    assert filtered[0]["group_id"] == filtered[1]["group_id"]
    assert [row["novel_beyond_all_mars_25km"] for row in filtered] == [False, True]
    assert all(
        row["component_novel_beyond_all_mars_25km"] is False
        for row in filtered
    )
