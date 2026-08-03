from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/evaluate_mars_gaussian_dofa_strict_spatial.py"
SPEC = importlib.util.spec_from_file_location("strict_gaussian_dofa_evaluator", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

PROTOCOL_PATH = ROOT / "configs/mars_gaussian_dofa_strict_spatial_evaluation_protocol.json"


def test_protocol_is_authorized_after_pre_outcome_validation():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["status"] == "authorized_after_comparator_provenance_correction_before_metric_computation"
    assert protocol["evaluation_gate"]["one_shot_outcome_access_authorized"] is True
    assert protocol["cohort"]["rows"] == 4401
    assert protocol["cohort"]["strict_spatial_components"] == 150
    assert protocol["bootstrap"] == {
        "replicates": 10000,
        "seed": 20260803,
        "confidence": 0.95,
        "unit": "strict 25 km connected spatial component",
        "paired": True,
    }
    assert module.sha256(MODULE_PATH) == protocol["score_dependencies"]["evaluator"]["sha256"]
    assert all(value is True for key, value in protocol["superiority_gate"].items() if key != "interpretation")


def test_cli_has_no_paths_thresholds_subsets_or_replicate_overrides():
    for arguments in (
        ["--scores", "replacement.npz"],
        ["--threshold", "0.2"],
        ["--limit", "100"],
        ["--replicates", "10"],
        ["--output", "replacement.json"],
    ):
        with pytest.raises(SystemExit):
            module.parse_args(arguments)


def test_score_side_validates_without_outcome_dependencies():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    scores, bindings = module.validate_score_side(protocol)
    assert len(bindings) == 6
    assert scores["sample_ids"].shape == (4401,)
    assert len(set(scores["sample_ids"].tolist())) == 4401
    assert np.unique(scores["groups"]).size == 373
    assert np.isfinite(scores["gaussian_dofa_scores"]).all()
    assert int(scores["released_mars_v3_decisions"].sum()) == 455
    assert int(scores["gaussian_dofa_decisions"].sum()) == 288


def test_dry_run_never_calls_outcome_loader(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("outcome loader must not run during dry-run")

    monkeypatch.setattr(module, "load_outcomes", fail)
    output_paths = [ROOT / path for path in json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))["outputs"].values()]
    if any(path.exists() for path in output_paths):
        with pytest.raises(FileExistsError, match="already exists"):
            module.main(["--dry-run"])
        assert capsys.readouterr().out == ""
    else:
        assert module.main(["--dry-run"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload == {
            "outcomes_opened": False,
            "rows": 4401,
            "status": "pre_outcome_dry_run_passed",
            "verified_score_bindings": 6,
        }


def test_scene_metrics_and_matched_fpr_are_tie_safe():
    labels = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.int8)
    scores = np.asarray([0.9, 0.6, 0.8, 0.6, 0.2, 0.1])
    predictions = scores >= 0.5
    metrics = module.model_metrics(labels, scores, predictions, 0.5, ">=")
    assert metrics["fixed_operating_point"]["tp"] == 2
    assert metrics["fixed_operating_point"]["fp"] == 2
    matched = module.matched_fpr_point(labels, scores, target_fpr=0.25)
    assert matched["fp"] <= 1
    assert matched["recall"] == 0.5
    assert matched["threshold"] == 0.9


def test_component_bootstrap_is_deterministic_and_paired():
    labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    components = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    baseline_scores = np.asarray([0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3])
    candidate_scores = np.asarray([0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.4])
    baseline_predictions = baseline_scores >= 0.5
    candidate_predictions = candidate_scores >= 0.5
    first = module.paired_component_bootstrap(
        labels=labels,
        components=components,
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
        replicates=200,
        seed=17,
        confidence=0.95,
        batch_size=23,
    )
    second = module.paired_component_bootstrap(
        labels=labels,
        components=components,
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
        replicates=200,
        seed=17,
        confidence=0.95,
        batch_size=23,
    )
    assert first == second
    assert first["components"] == 4
    assert first["delta_intervals"]["average_precision"]["lower"] >= 0.0
    assert first["delta_intervals"]["fixed_false_positive_rate"]["upper"] <= 0.0


def test_mcnemar_discordance_orientation():
    labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
    baseline = np.asarray([1, 0, 1, 0], dtype=bool)
    candidate = np.asarray([1, 1, 0, 1], dtype=bool)
    result = module.exact_mcnemar(labels, baseline, candidate)
    assert result["baseline_wrong_candidate_correct"] == 2
    assert result["baseline_correct_candidate_wrong"] == 1
    assert result["discordant"] == 3
    assert 0.0 <= result["two_sided_exact_pvalue"] <= 1.0


def test_writers_refuse_overwrite(tmp_path):
    sandbox = ROOT / ".research" / f"pytest_strict_eval_{tmp_path.name}"
    shutil.rmtree(sandbox, ignore_errors=True)
    sandbox.mkdir(parents=True)
    json_path = sandbox / "report.json"
    markdown_path = sandbox / "report.md"
    report = {
        "cohort": {"rows": 2, "positives": 1, "negatives": 1, "strict_spatial_components": 1},
        "comparisons": {
            "gaussian_dofa_vs_released_mars_v3": {
                "baseline_name": "base",
                "candidate_name": "candidate",
                "baseline": {"average_precision": 0.5, "roc_auc": 0.5, "fixed_operating_point": {"recall": 0.0, "false_positive_rate": 0.0, "precision": 0.0}},
                "candidate": {"average_precision": 1.0, "roc_auc": 1.0, "fixed_operating_point": {"recall": 1.0, "false_positive_rate": 0.0, "precision": 1.0}},
                "superiority_gate": {"passed": True},
            }
        },
    }
    try:
        module.write_json(json_path, {"ok": True})
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            module.write_json(json_path, {"ok": False})
        module.write_markdown(markdown_path, report)
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            module.write_markdown(markdown_path, report)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
