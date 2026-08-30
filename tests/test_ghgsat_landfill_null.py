from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

import tools.audit_ghgsat_landfill_null as audit


class Response:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        declared: int | None = None,
        url: str = "https://zenodo.org/exact.csv",
    ) -> None:
        self.payload = payload
        self.status_code = status
        self.url = url
        self.history: list[object] = []
        self.closed = False
        self.headers = {"Content-Length": str(len(payload) if declared is None else declared)}

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self.payload[index : index + 3] for index in range(0, len(self.payload), 3)]

    def close(self) -> None:
        self.closed = True


class Session:
    def __init__(self, response: Response) -> None:
        self.response = response
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def get(self, *args: object, **kwargs: object) -> Response:
        self.calls.append((args, kwargs))
        return self.response


def protocol() -> dict[str, object]:
    return copy.deepcopy(audit.load_protocol())


def miniature_protocol(
    *, positive_observations: int = 1, null_observations: int = 1, positive_rows: int = 1, sites: int = 1
) -> dict[str, object]:
    value = protocol()
    value["population_reconciliation_gates"] = {
        "exact_clear_sky_observations": positive_observations + null_observations,
        "exact_positive_observations": positive_observations,
        "exact_null_observations": null_observations,
        "exact_positive_plume_rows": positive_rows,
        "exact_distinct_sites": sites,
        "exact_years": [2021],
        "all_released_rows_and_observations_valid": True,
    }
    return value


def row(
    *,
    site: str = "site-a",
    obs: str = "obs-a",
    state: str = "positive",
    sat: int = 1,
    date: str = "2021-01-02T03:04:05Z",
    lat: float = 1.0,
    lon: float = 2.0,
) -> dict[str, str]:
    values = {
        "site_ID": site,
        "lat": str(lat if state == "positive" else 0.0),
        "lon": str(lon if state == "positive" else 0.0),
        "year": "2021",
        "month": "1",
        "day": "2",
        "hour": "3",
        "minute": "4",
        "second": "5",
        "Q_t_per_hr": "1.5" if state == "positive" else "0.0",
        "Q_error_t_per_hr": "0.1" if state == "positive" else "",
        "wind_speed_m_per_s": "2.0" if state == "positive" else "0.0",
        "sat_ID": str(sat),
        "obs_ID": obs,
        "date": date,
        "IME_kg": "3.0" if state == "positive" else "",
        "intermediate_results_L_m": "4.0" if state == "positive" else "",
        "intermediate_results_effective_wind_speed_m_per_s": "5.0" if state == "positive" else "",
        "conversion_ch4_ppb_to_molm2": "6.0" if state == "positive" else "",
        "manually_pinned_sources": "pin" if state == "positive" else "",
        "plume_tif_file_name": "plume.tif" if state == "positive" else "",
    }
    return values


def write_csv(path: Path, rows: list[dict[str, str]], *, fields: list[str] | None = None) -> None:
    ordered = fields or list(protocol()["csv_contract"]["expected_columns_in_order"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parsed_rows(tmp_path: Path, rows: list[dict[str, str]], *, value: dict[str, object] | None = None) -> list[dict[str, object]]:
    path = tmp_path / "fixture.csv"
    write_csv(path, rows)
    return audit.parse_csv_rows(path, value or miniature_protocol())


def observation(site: str, obs: str, *, state: str = "null", sat: int = 1, lon: float = 0.0) -> dict[str, object]:
    return {
        "site_ID": site,
        "obs_ID": obs,
        "date": "2021-01-02T03:04:05Z",
        "sat_ID": sat,
        "year": 2021,
        "observation_state": state,
        "representative_latitude": 0.0 if state == "null" else None,
        "representative_longitude": lon if state == "null" else None,
        "eligible_for_target_catalog": False,
    }


def test_safe_default_has_no_network_or_csv_access(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(audit, "load_protocol", lambda: {"ok": True})
    monkeypatch.setattr(audit, "validate_frozen_local_inputs", lambda value: {"ok": True})
    monkeypatch.setattr(
        audit,
        "validation_plan",
        lambda: {"mode": "validation_only", "network_executed": False, "ghgsat_csv_opened": False},
    )
    monkeypatch.setattr(audit.requests, "Session", lambda: (_ for _ in ()).throw(AssertionError("network")))
    monkeypatch.setattr(audit, "audit_verified_cache", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CSV")))
    assert audit.main([]) == 0
    output = capsys.readouterr().out
    assert '"network_executed": false' in output
    assert '"ghgsat_csv_opened": false' in output
    assert not audit.build_parser().parse_args([]).download_and_audit


def test_protocol_hash_mismatch_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "protocol.json"
    path.write_bytes(audit.EXPECTED_PROTOCOL.read_bytes() + b" ")
    monkeypatch.setattr(audit, "EXPECTED_PROTOCOL", path)
    with pytest.raises(audit.GHGSatAuditError, match="SHA-256 mismatch"):
        audit.load_protocol(path)


def test_frozen_input_hash_mismatch_rejected(tmp_path: Path) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"wrong")
    value = {"frozen_local_inputs": {"fixture": {"path": "input", "bytes": 5, "sha256": "0" * 64}}}
    with pytest.raises(audit.GHGSatAuditError, match="hash mismatch"):
        audit.validate_frozen_local_inputs(value, root=tmp_path)


@pytest.mark.parametrize("failure", ["cap", "bytes", "md5", "http"])
def test_download_cap_exact_count_md5_part_and_atomic_behavior(tmp_path: Path, failure: str) -> None:
    payload = b"abcdef"
    value = protocol()
    value["authoritative_source"]["csv"] = {
        "name": "fixture.csv",
        "url": "https://zenodo.org/exact.csv",
        "bytes": len(payload),
        "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        "maximum_bytes": 8,
    }
    status = 200
    if failure == "cap":
        payload = b"012345678"
    elif failure == "bytes":
        value["authoritative_source"]["csv"]["bytes"] = len(payload) + 1
    elif failure == "md5":
        value["authoritative_source"]["csv"]["md5"] = "0" * 32
    elif failure == "http":
        status = 503
    response = Response(payload, status=status)
    if failure == "cap":
        response.headers.pop("Content-Length")
    session = Session(response)
    destination = tmp_path / "fixture.csv"
    value["outputs"]["ignored_csv"] = "fixture.csv"
    destination.write_bytes(b"old-cache")
    downloader = audit.FrozenCSVDownloader(session)
    old_root = audit.ROOT
    audit.ROOT = tmp_path
    try:
        with pytest.raises(audit.GHGSatAuditError):
            downloader.download(protocol=value, destination=destination)
    finally:
        audit.ROOT = old_root
    assert destination.read_bytes() == b"old-cache"
    assert not (tmp_path / "fixture.csv.part").exists()
    assert downloader.bytes_received == len(payload)
    assert session.calls[0][0] == ("https://zenodo.org/exact.csv",)
    assert session.calls[0][1]["headers"] == {
        "Accept": "*/*",
        "User-Agent": "ERSRR-research-metadata-audit/1.0",
    }
    assert response.closed


def test_download_success_atomically_installs_only_verified_payload(tmp_path: Path) -> None:
    payload = b"abcdef"
    value = protocol()
    value["authoritative_source"]["csv"].update(
        bytes=len(payload), md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(), maximum_bytes=8
    )
    value["outputs"]["ignored_csv"] = "fixture.csv"
    destination = tmp_path / "fixture.csv"
    old_root = audit.ROOT
    audit.ROOT = tmp_path
    try:
        receipt = audit.FrozenCSVDownloader(Session(Response(payload))).download(protocol=value, destination=destination)
    finally:
        audit.ROOT = old_root
    assert destination.read_bytes() == payload
    assert receipt["atomic_install"] is True
    assert not destination.with_name("fixture.csv.part").exists()


def test_download_rejects_redirect_outside_authorized_zenodo_host(tmp_path: Path) -> None:
    payload = b"abcdef"
    value = protocol()
    value["authoritative_source"]["csv"].update(
        bytes=len(payload),
        md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        maximum_bytes=8,
    )
    value["outputs"]["ignored_csv"] = "fixture.csv"
    destination = tmp_path / "fixture.csv"
    old_root = audit.ROOT
    audit.ROOT = tmp_path
    try:
        with pytest.raises(audit.GHGSatAuditError, match="authorized Zenodo host"):
            audit.FrozenCSVDownloader(
                Session(Response(payload, url="https://example.invalid/file.csv"))
            ).download(protocol=value, destination=destination)
    finally:
        audit.ROOT = old_root
    assert not destination.exists()
    assert not destination.with_name("fixture.csv.part").exists()


def test_exact_header_and_strict_field_validation(tmp_path: Path) -> None:
    value = miniature_protocol()
    fields = list(value["csv_contract"]["expected_columns_in_order"])
    bad_header = tmp_path / "bad-header.csv"
    write_csv(bad_header, [row()], fields=list(reversed(fields)))
    with pytest.raises(audit.GHGSatAuditError, match="header"):
        audit.parse_csv_rows(bad_header, value)
    cases = [
        ("lat", "nan", "finite"),
        ("sat_ID", "1.0", "integer"),
        ("sat_ID", "6", "C1-C5"),
        ("date", "2021-01-02T03:04:06Z", "calendar"),
    ]
    for field, replacement, message in cases:
        bad = row()
        bad[field] = replacement
        path = tmp_path / f"bad-{field}.csv"
        write_csv(path, [bad])
        with pytest.raises(audit.GHGSatAuditError, match=message):
            audit.parse_csv_rows(path, value)

    invalid_null = row(state="null")
    invalid_null["IME_kg"] = "0.0"
    path = tmp_path / "bad-null-measurement.csv"
    write_csv(path, [invalid_null])
    with pytest.raises(audit.GHGSatAuditError, match="positive-only measurement"):
        audit.parse_csv_rows(path, value)


@pytest.mark.parametrize(
    ("positive_observations", "null_observations", "positive_rows"),
    [(1, 1, 1), (1, 1, 2), (2, 1, 3)],
)
def test_positive_null_grouping_and_exact_count_reconciliation(
    tmp_path: Path, positive_observations: int, null_observations: int, positive_rows: int
) -> None:
    rows: list[dict[str, str]] = []
    for index in range(positive_rows):
        obs_index = index % positive_observations
        rows.append(row(obs=f"positive-{obs_index}", lat=1.0 + index * 0.001, lon=2.0))
    rows.append(row(obs="null", state="null"))
    value = miniature_protocol(
        positive_observations=positive_observations,
        null_observations=null_observations,
        positive_rows=positive_rows,
    )
    parsed = parsed_rows(tmp_path, rows, value=value)
    observations, counts = audit.validate_and_group_rows(parsed, value)
    assert counts["positive_observations"] == positive_observations
    assert counts["positive_plume_rows"] == positive_rows
    assert counts["null_observations"] == 1
    assert len(observations) == positive_observations + 1
    wrong = copy.deepcopy(value["population_reconciliation_gates"])
    wrong["exact_positive_plume_rows"] = positive_rows + 1
    with pytest.raises(audit.GHGSatAuditError, match="reconciliation"):
        audit.validate_and_group_rows(parsed, value, expectations=wrong)


def test_mixed_positive_null_observation_rejected(tmp_path: Path) -> None:
    rows = parsed_rows(tmp_path, [row(obs="mixed"), row(obs="mixed", state="null")])
    with pytest.raises(audit.GHGSatAuditError, match="mixes positive and null"):
        audit.validate_and_group_rows(rows, miniature_protocol(positive_rows=1))


def test_same_site_observation_id_cannot_change_time_or_satellite(tmp_path: Path) -> None:
    value = miniature_protocol(positive_observations=2, null_observations=0, positive_rows=2)
    first = row(obs="same")
    second = row(obs="same", date="2021-01-03T03:04:05Z")
    second["day"] = "3"
    rows = parsed_rows(tmp_path, [first, second], value=value)
    with pytest.raises(audit.GHGSatAuditError, match="inconsistent date or satellite"):
        audit.validate_and_group_rows(rows, value)


def test_null_observation_must_have_exactly_one_released_row(tmp_path: Path) -> None:
    value = miniature_protocol(positive_observations=1, null_observations=1, positive_rows=1)
    rows = parsed_rows(
        tmp_path,
        [row(obs="positive"), row(obs="null", state="null")],
        value=value,
    )
    duplicate = dict(rows[-1])
    duplicate["data_row_number"] = 3
    with pytest.raises(audit.GHGSatAuditError, match="exactly one released row"):
        audit.validate_and_group_rows([*rows, duplicate], value)


def test_missing_positive_coordinate_for_null_site_rejected(tmp_path: Path) -> None:
    value = miniature_protocol(positive_observations=1, null_observations=1, positive_rows=1, sites=2)
    rows = parsed_rows(tmp_path, [row(site="positive", obs="p"), row(site="null-only", obs="n", state="null")], value=value)
    with pytest.raises(audit.GHGSatAuditError, match="no positive coordinate"):
        audit.validate_and_group_rows(rows, value)


def test_coordinate_medoid_and_tie_break_are_deterministic() -> None:
    points = [(0.0, 1.0, 9), (0.0, -1.0, 8), (0.0, 0.0, 7)]
    assert audit.coordinate_medoid(list(reversed(points))) == (0.0, 0.0, 7)
    tied = [(0.0, 1.0, 2), (0.0, -1.0, 3)]
    assert audit.coordinate_medoid(tied) == (0.0, -1.0, 3)
    duplicate_tie = [(1.0, 1.0, 5), (1.0, 1.0, 2)]
    assert audit.coordinate_medoid(duplicate_tie) == (1.0, 1.0, 2)


def test_over_25km_within_site_span_rejected(tmp_path: Path) -> None:
    value = miniature_protocol(positive_observations=2, null_observations=1, positive_rows=2)
    rows = parsed_rows(
        tmp_path,
        [row(obs="p1", lat=0.0, lon=1.0), row(obs="p2", lat=0.0, lon=1.3), row(obs="n", state="null")],
        value=value,
    )
    with pytest.raises(audit.GHGSatAuditError, match="span exceeds"):
        audit.validate_and_group_rows(rows, value)


def test_c1_c2_only_deterministic_four_per_site_selection() -> None:
    value = protocol()
    rows = [observation("a", f"obs-{index}", sat=(index % 5) + 1) for index in range(15)]
    selected = audit.select_morning_nulls(list(reversed(rows)), value)
    candidates = [item for item in rows if item["sat_ID"] in {1, 2}]
    expected = sorted(candidates, key=audit.selection_rank)[:4]
    assert [item["obs_ID"] for item in selected] == [item["obs_ID"] for item in expected]
    assert {item["sat_ID"] for item in selected} <= {1, 2}


def test_strict_protected_radius_exclusions_and_transitive_components() -> None:
    degrees_25km = math.degrees(25.0 / 6371.0088)
    rows = [
        observation("at-boundary", "a", lon=degrees_25km),
        observation("chain-a", "b", lon=1.0),
        observation("chain-b", "c", lon=1.2),
        observation("chain-c", "d", lon=1.4),
    ]
    filtered, sizes, exclusions = audit.spatial_filter(
        rows,
        protected_mars={"test": (0.0, 0.0)},
        prior_negative={},
    )
    boundary = next(item for item in filtered if item["site_ID"] == "at-boundary")
    assert boundary["nearest_official_mars_test_km"] == pytest.approx(25.0)
    assert boundary["passes_protected_distance_filter"] is False
    assert exclusions["within_or_at_25km_of_official_mars_test_representative"] == 1
    assert sorted(sizes.values()) == [3]
    chain_ids = {item["component_id"] for item in filtered if str(item["site_ID"]).startswith("chain")}
    assert len(chain_ids) == 1


@pytest.mark.parametrize(
    ("observations", "sites", "components", "expected"),
    [(55, 30, 20, False), (56, 29, 20, False), (56, 30, 19, False), (56, 30, 20, True)],
)
def test_gate_pass_fail_boundaries(observations: int, sites: int, components: int, expected: bool) -> None:
    eligible = [observation(f"site-{index % sites}", f"obs-{index}") for index in range(observations)]
    sizes = {f"component-{index}": 1 for index in range(components)}
    gates = audit.evaluate_gates(eligible, sizes, protocol())
    assert all(bool(gate["pass"]) for gate in gates.values()) is expected


def test_mars_reader_rejects_forbidden_column_request(tmp_path: Path) -> None:
    with pytest.raises(audit.GHGSatAuditError, match="Forbidden"):
        audit.read_safe_mars_points(tmp_path / "unused.csv", protocol(), requested_columns={"lat", "isplume"})


def test_compact_report_explicitly_proves_forbidden_resources_untouched() -> None:
    value = protocol()
    report = audit.build_failure_report(value, audit.GHGSatAuditError("fixture"))
    assert report["decision"] == "FAIL"
    assert report["all_released_rows_and_observations_valid"] is False
    assert report["access_boundary"]
    assert all(accessed is False for accessed in report["access_boundary"].values())
    markdown = audit._markdown(report)
    assert "Target catalog/assets" in markdown
    assert "protected outcomes" in markdown
    assert "model checkpoints" in markdown


def test_failed_cached_audit_removes_stale_jsonl_and_emits_compact_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = protocol()
    value["outputs"] = {
        "ignored_root": "ignored",
        "ignored_csv": "ignored/GHGSat_detected_plumes.csv",
        "ignored_validated_observations": "ignored/validated.jsonl",
        "ignored_selected_nulls": "ignored/selected.jsonl",
        "ignored_eligible_nulls": "ignored/eligible.jsonl",
        "compact_json": "reports/report.json",
        "compact_markdown": "reports/report.md",
    }
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    for key in (
        "ignored_validated_observations",
        "ignored_selected_nulls",
        "ignored_eligible_nulls",
    ):
        path = tmp_path / value["outputs"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n", encoding="utf-8")
    csv_path = tmp_path / value["outputs"]["ignored_csv"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(b"invalid")
    with pytest.raises(audit.GHGSatAuditError):
        audit.audit_verified_cache(value, local_receipts={})
    assert not any(
        (tmp_path / value["outputs"][key]).exists()
        for key in (
            "ignored_validated_observations",
            "ignored_selected_nulls",
            "ignored_eligible_nulls",
        )
    )
    report = json.loads((tmp_path / value["outputs"]["compact_json"]).read_text())
    assert report["decision"] == "FAIL"
    assert report["all_released_rows_and_observations_valid"] is False


def test_combined_flags_are_documented_as_download_then_single_audit() -> None:
    assert "download, verify/install, then audit once" in (audit.__doc__ or "")
    args = audit.build_parser().parse_args(["--download-and-audit", "--audit-cached"])
    assert args.download_and_audit and args.audit_cached
