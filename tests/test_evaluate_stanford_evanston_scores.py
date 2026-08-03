from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import evaluate_stanford_evanston_scores as evaluator  # noqa: E402

PROTOCOL = ROOT / "configs/stanford_evanston_outcome_evaluation_protocol.json"


def synthetic_summary(event_id: str, rate: float) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=evaluator.REQUIRED_SUMMARY_COLUMNS)
    writer.writeheader()
    writer.writerow(
        {
            "release_ID": event_id,
            "date": "2024-08-09",
            "time_UTC": "18:22:00",
            "location": "Evanston",
            "lat": "41.2757",
            "lon": "-110.9313",
            "ch4_kgh_mean": str(rate),
            "ch4_kgh_sigma": "1.0",
            "ci95_lower": "0.0",
            "ci95_upper": "2.0",
            "PredInt95_lower": "0.0",
            "PredInt95_upper": "3.0",
            "PI_within_10pct_of_mean": "True",
        }
    )
    return stream.getvalue().encode("utf-8")


def test_frozen_outcome_protocol_validates_every_local_binding() -> None:
    protocol = evaluator.load_protocol(PROTOCOL)

    assert protocol["cohort"]["expected_rows"] == 9
    assert len(protocol["sources"]["summary_files"]) == 9
    assert protocol["outcome"]["field"] == "ch4_kgh_mean"
    assert protocol["sources"]["repository_version"] == 17
    assert protocol["sources"]["license"] == "CC BY 4.0"


def test_protocol_thresholds_match_evaluator_contracts_exactly() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    for name, contract in evaluator.MODEL_CONTRACTS.items():
        frozen = protocol["models"][name]
        assert frozen["threshold"] == contract["threshold"]
        assert frozen["comparator"] == contract["comparator"]
        assert frozen["score_field"] == contract["score"]
        assert frozen["decision_field"] == contract["decision"]


def test_cli_exposes_no_source_score_or_output_path_override() -> None:
    for forbidden in (
        "--scores",
        "--score-receipt",
        "--source",
        "--joined",
        "--output-json",
        "--output-markdown",
    ):
        with pytest.raises(SystemExit):
            evaluator.parse_args([forbidden, "arbitrary"])


def test_dry_run_validates_scores_without_opening_network(monkeypatch, capsys) -> None:
    def forbidden_network(*args, **kwargs):
        raise AssertionError("dry run attempted network outcome access")

    monkeypatch.setattr(evaluator.urllib.request, "urlopen", forbidden_network)

    assert evaluator.main(["--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 9
    assert payload["source_files_opened"] == 0
    assert payload["scores_sha256"] == (
        "89e887309ba15ed85ba38a24fee653ca830c1a0a1d8a9c0c075224655b7aa6f8"
    )


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0.0, "negative"),
        (0.000001, "challenge"),
        (999.999999, "challenge"),
        (1000.0, "positive"),
        (2500.0, "positive"),
    ],
)
def test_frozen_stratum_boundaries(rate: float, expected: str) -> None:
    assert evaluator.stratum(rate) == expected


def test_summary_parser_requires_exact_schema_one_row_and_matching_id() -> None:
    source = {"event_id": "08092024_S2A"}

    row = evaluator.parse_summary(synthetic_summary("08092024_S2A", 1000.0), source)

    assert row["event_id"] == "08092024_S2A"
    assert row["ch4_kgh_mean"] == 1000.0
    with pytest.raises(ValueError, match="release_ID differs"):
        evaluator.parse_summary(synthetic_summary("wrong", 1000.0), source)
    with pytest.raises(ValueError, match="Negative methane rate"):
        evaluator.parse_summary(synthetic_summary("08092024_S2A", -1.0), source)


def test_source_digest_validator_rejects_size_sha1_and_md5_changes() -> None:
    payload = b"frozen source"
    source = {
        "event_id": "event",
        "size": len(payload),
        "sha1": evaluator.digest_bytes(payload, "sha1"),
        "md5": evaluator.digest_bytes(payload, "md5"),
    }

    evaluator.verify_source_bytes(payload, source)
    with pytest.raises(ValueError, match="size mismatch"):
        evaluator.verify_source_bytes(payload + b"x", source)
    changed = dict(source, sha1="0" * 40)
    with pytest.raises(ValueError, match="SHA-1 mismatch"):
        evaluator.verify_source_bytes(payload, changed)
    changed = dict(source, md5="0" * 32)
    with pytest.raises(ValueError, match="MD5 mismatch"):
        evaluator.verify_source_bytes(payload, changed)


def test_exact_binomial_intervals_and_one_class_metrics_are_explicit() -> None:
    assert evaluator.exact_binomial_interval(0, 0) == [None, None]
    lower_zero, upper_zero = evaluator.exact_binomial_interval(0, 3)
    lower_all, upper_all = evaluator.exact_binomial_interval(3, 3)
    assert lower_zero == 0.0 and 0.0 < upper_zero < 1.0
    assert 0.0 < lower_all < 1.0 and upper_all == 1.0

    labels = np.zeros(3, dtype=np.int8)
    scores = np.asarray([0.1, 0.2, 0.3])
    result = evaluator.model_metrics(labels, scores, evaluator.MODEL_CONTRACTS["released_mars_v3"])
    assert result["ranking"]["average_precision"] is None
    assert result["ranking"]["undefined_reason"]
    assert result["fixed_threshold"]["recall"] is None
    assert result["fixed_threshold"]["false_positive_rate"] == 0.0


def test_empty_primary_comparison_remains_undefined_without_failure() -> None:
    empty = np.asarray([], dtype=np.float64)
    result = evaluator.paired_comparison(
        np.asarray([], dtype=np.int8),
        np.asarray([], dtype=str),
        empty,
        empty,
        evaluator.MODEL_CONTRACTS["released_mars_v3"],
        evaluator.MODEL_CONTRACTS["gaussian_dofa"],
        replicates=10,
        seed=20260812,
    )

    assert result["paired_date_bootstrap"] is None
    assert result["superiority_gate"]["passed"] is False
    assert result["exact_mcnemar"]["two_sided_p_value"] is None
