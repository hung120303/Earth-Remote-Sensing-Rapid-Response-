#!/usr/bin/env python3
"""Confirm protected DOFA-v2 fusion with train-fitted normalization only."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.random_projection import SparseRandomProjection

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from confirm_mars_dofa_v2_projection_ensemble import (  # noqa: E402
    mean_logit_probabilities,
)
from train_mars_crossfold_bagged_scene_head import (  # noqa: E402
    DEFAULT_FOLD0_CACHE,
    DEFAULT_FOLD0_SHA256,
    DEFAULT_FOLD1_CACHE,
    DEFAULT_FOLD1_SHA256,
    DEFAULT_INNER_CACHE,
    DEFAULT_INNER_SHA256,
    DEFAULT_SCORE_CACHE,
    DEFAULT_SCORE_SHA256,
    load_development,
)
from train_mars_dofa_v2_protected_fusion import (  # noqa: E402
    operating_counts_preserved,
    protected_logit_blend,
)
from train_mars_dofa_v2_scene_probe import (  # noqa: E402
    DEFAULT_DOFA,
    PROJECTION_DIM,
    SELECTION_FOLDS,
    align_features,
    candidate_summary,
    crossfit_scores,
    evaluate_candidate,
    select_features,
    write_json,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import metric_summary  # noqa: E402

DEFAULT_PROTOCOL = Path(
    "configs/mars_dofa_v2_train_fitted_normalization_protocol.json"
)
DEFAULT_PROTECTED_REPORT = Path(
    "reports/experiments/mars_dofa_v2_protected_fusion_folds34.json"
)
DEFAULT_JSON = Path(
    "reports/experiments/mars_dofa_v2_train_fitted_normalization_folds34.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/experiments/MARS_DOFA_V2_TRAIN_FITTED_NORMALIZATION_FOLDS34.md"
)
NORMALIZATION_MODES = ("global_train_fitted", "sensor_train_fitted")


def source_fitted_normalize(
    source: np.ndarray,
    target: np.ndarray,
    source_sensors: np.ndarray,
    target_sensors: np.ndarray,
    *,
    mode: str,
    epsilon: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit normalization statistics on source rows and apply them to both arrays."""
    source_values = np.asarray(source, dtype=np.float32)
    target_values = np.asarray(target, dtype=np.float32)
    source_domains = np.asarray(source_sensors)
    target_domains = np.asarray(target_sensors)
    if source_values.ndim != 2 or target_values.ndim != 2:
        raise ValueError("Source and target features must be matrices")
    if source_values.shape[1] != target_values.shape[1]:
        raise ValueError("Source and target feature widths differ")
    if source_domains.shape != (source_values.shape[0],) or target_domains.shape != (
        target_values.shape[0],
    ):
        raise ValueError("Sensor vectors do not align with feature rows")
    if mode == "global_train_fitted":
        source_domains = np.zeros(source_domains.shape, dtype=np.uint8)
        target_domains = np.zeros(target_domains.shape, dtype=np.uint8)
    elif mode != "sensor_train_fitted":
        raise ValueError(f"Unknown normalization mode: {mode}")

    normalized_source = np.empty(source_values.shape, dtype=np.float32)
    normalized_target = np.empty(target_values.shape, dtype=np.float32)
    target_seen = np.zeros(target_values.shape[0], dtype=bool)
    for domain in np.unique(source_domains):
        fit_rows = source_domains == domain
        held_rows = target_domains == domain
        center = source_values[fit_rows].mean(axis=0, dtype=np.float64)
        scale = np.maximum(
            source_values[fit_rows].std(axis=0, dtype=np.float64), epsilon
        )
        normalized_source[fit_rows] = (source_values[fit_rows] - center) / scale
        normalized_target[held_rows] = (target_values[held_rows] - center) / scale
        target_seen[held_rows] = True
    if not target_seen.all():
        raise ValueError("Held data contains a sensor absent from fit data")
    return normalized_source, normalized_target


def build_source_fitted_views(
    features: np.ndarray,
    folds: np.ndarray,
    sensors: np.ndarray,
    *,
    seed: int,
    mode: str,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    views = {}
    for holdout in SELECTION_FOLDS:
        fit, held = folds != holdout, folds == holdout
        source, target = source_fitted_normalize(
            features[fit],
            features[held],
            sensors[fit],
            sensors[held],
            mode=mode,
        )
        projection = SparseRandomProjection(
            n_components=PROJECTION_DIM,
            density="auto",
            dense_output=True,
            random_state=seed,
        )
        source_projected = projection.fit_transform(source).astype(np.float32)
        target_projected = projection.transform(target).astype(np.float32)
        source_projected, target_projected = source_fitted_normalize(
            source_projected,
            target_projected,
            sensors[fit],
            sensors[held],
            mode=mode,
        )
        views[holdout] = (fit, held, source_projected, target_projected)
    return views


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["evaluation"]["versus_current"]["delta"]
    interval = selected["paired_group_bootstrap_ap_delta_vs_current"]
    lines = [
        "# Train-fitted DOFA-v2 normalization confirmation - folds 3/4",
        "",
        "All feature normalization statistics are fit on the training fold and "
        "applied unchanged to the held fold. The protected gate, DOFA weight, "
        "projection seeds, and downstream promotion gates are fixed.",
        "",
        f"- Selected mode: `{selected['normalization_mode']}`",
        f"- AP delta vs current: {delta['average_precision']:+.6f}",
        f"- Recall delta at FPR 0.0713: {delta['recall_at_fpr_0_0713']:+.6f}",
        f"- Paired-site AP 95% interval: [{interval['lower']:+.6f}, "
        f"{interval['upper']:+.6f}]",
        f"- Operating confusion counts preserved: "
        f"{'yes' if selected['operating_counts_preserved'] else 'no'}",
        "",
        "| Normalization | AP delta | Recall delta | AP CI lower | Promoted |",
        "|---|---:|---:|---:|:---:|",
    ]
    for candidate in report["candidate_summaries"]:
        local = candidate["evaluation"]["delta_vs_current"]
        lower = candidate["paired_group_bootstrap_ap_delta_vs_current"]["lower"]
        lines.append(
            f"| `{candidate['normalization_mode']}` | "
            f"{local['average_precision']:+.6f} | "
            f"{local['recall_at_fpr_0_0713']:+.6f} | {lower:+.6f} | "
            f"{'yes' if candidate['promotion_gates_pass'] else 'no'} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if tuple(protocol["normalization_modes"]) != NORMALIZATION_MODES:
        raise ValueError("Train-fitted normalization candidates differ")
    if protocol["fixed_candidate"] != {
        "feature_set": "change_extreme",
        "C": 0.01,
        "projection_dimension": 2048,
        "projection_seeds": [20260780, 20260781, 20260782, 20260783, 20260784],
        "aggregation": "mean_logit",
        "gate": 0.5,
        "weight": 0.05,
    }:
        raise ValueError("Protected candidate differs from frozen result")
    if int(protocol["fixed_candidate"]["projection_dimension"]) != PROJECTION_DIM:
        raise ValueError("Projection dimension differs")
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--dofa", default=DEFAULT_DOFA.as_posix())
    parser.add_argument("--dofa-sha256", required=True)
    parser.add_argument("--protected-report", default=DEFAULT_PROTECTED_REPORT.as_posix())
    parser.add_argument("--inner-cache", default=DEFAULT_INNER_CACHE.as_posix())
    parser.add_argument("--inner-sha256", default=DEFAULT_INNER_SHA256)
    parser.add_argument("--fold0-cache", default=DEFAULT_FOLD0_CACHE.as_posix())
    parser.add_argument("--fold0-sha256", default=DEFAULT_FOLD0_SHA256)
    parser.add_argument("--fold1-cache", default=DEFAULT_FOLD1_CACHE.as_posix())
    parser.add_argument("--fold1-sha256", default=DEFAULT_FOLD1_SHA256)
    parser.add_argument("--score-cache", default=DEFAULT_SCORE_CACHE.as_posix())
    parser.add_argument("--score-sha256", default=DEFAULT_SCORE_SHA256)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()

    root = repo_root()
    protocol_path = (root / args.protocol).resolve()
    protected_report_path = (root / args.protected_report).resolve()
    protocol = load_protocol(protocol_path)
    dependencies = protocol["dependencies"]
    if sha256(protected_report_path) != dependencies["protected_report_sha256"]:
        raise ValueError("Protected-fusion report hash mismatch")
    prior = json.loads(protected_report_path.read_text(encoding="utf-8"))
    if not prior["all_promotion_gates_pass"]:
        raise ValueError("Protected transductive prerequisite did not pass")
    if (
        float(prior["selected"]["gate"]),
        float(prior["selected"]["weight"]),
    ) != (0.5, 0.05):
        raise ValueError("Protected candidate differs from dependency")

    paths = {
        "dofa": (root / args.dofa).resolve(),
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
        "score": (root / args.score_cache).resolve(),
    }
    expected = {
        "dofa": args.dofa_sha256,
        "inner": args.inner_sha256,
        "fold0": args.fold0_sha256,
        "fold1": args.fold1_sha256,
        "score": args.score_sha256,
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} hash mismatch")
    if expected["dofa"] != dependencies["dofa_cache_sha256"]:
        raise ValueError("DOFA cache differs from the frozen protocol")

    all_values = load_development(
        {name: paths[name] for name in ("inner", "fold0", "fold1")}, paths["score"]
    )
    encoded, names = align_features(paths["dofa"], all_values)
    selection = np.isin(all_values["folds"], SELECTION_FOLDS)
    values = {
        key: np.asarray(all_values[key])[selection]
        for key in (
            "labels",
            "sensors",
            "sample_ids",
            "groups",
            "folds",
            "primary",
            "current",
        )
    }
    fixed = protocol["fixed_candidate"]
    features, _ = select_features(encoded, names, fixed["feature_set"])
    del encoded
    gc.collect()
    current_metrics = metric_summary(
        values["labels"], values["current"], values["sensors"]
    )

    candidates: list[dict[str, Any]] = []
    for mode in NORMALIZATION_MODES:
        raw_scores: list[np.ndarray] = []
        for seed in map(int, fixed["projection_seeds"]):
            views = build_source_fitted_views(
                features,
                values["folds"],
                values["sensors"],
                seed=seed,
                mode=mode,
            )
            raw_scores.append(crossfit_scores(views, values["labels"], float(fixed["C"])))
            del views
            gc.collect()
        aggregate_raw = mean_logit_probabilities(raw_scores)
        scores = protected_logit_blend(
            values["current"],
            aggregate_raw,
            gate=float(fixed["gate"]),
            weight=float(fixed["weight"]),
        )
        evaluation = evaluate_candidate(
            values,
            scores,
            {
                "family": "protected_logit_blend",
                "normalization_mode": mode,
                "gate": fixed["gate"],
                "weight": fixed["weight"],
            },
            1.0,
        )
        preserved = operating_counts_preserved(evaluation, current_metrics)
        bootstrap = ap_group_bootstrap(
            values["labels"],
            values["current"],
            scores,
            values["groups"],
            replicates=int(protocol["bootstrap"]["replicates"]),
            seed=int(protocol["bootstrap"]["seed"]),
        )
        promoted = bool(
            evaluation["stable"] and preserved and bootstrap["lower"] > 0.0
        )
        fold_ap = [
            evaluation["per_fold"][str(fold)]["versus_current"]["delta"][
                "average_precision"
            ]
            for fold in SELECTION_FOLDS
        ]
        sensor_ap = evaluation["versus_current"]["delta"]["sensor_average_precision"]
        candidates.append(
            {
                "normalization_mode": mode,
                "evaluation": evaluation,
                "operating_counts_preserved": preserved,
                "paired_group_bootstrap_ap_delta_vs_current": bootstrap,
                "promotion_gates_pass": promoted,
                "rank": [
                    int(promoted),
                    int(evaluation["stable"]),
                    bootstrap["lower"],
                    min(fold_ap),
                    min(sensor_ap.values()),
                    evaluation["versus_current"]["delta"]["average_precision"],
                    -NORMALIZATION_MODES.index(mode),
                ],
            }
        )

    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    passed = bool(selected["promotion_gates_pass"])
    summaries = [
        {
            "normalization_mode": value["normalization_mode"],
            "evaluation": candidate_summary(value["evaluation"]),
            "operating_counts_preserved": value["operating_counts_preserved"],
            "paired_group_bootstrap_ap_delta_vs_current": value[
                "paired_group_bootstrap_ap_delta_vs_current"
            ],
            "promotion_gates_pass": value["promotion_gates_pass"],
            "rank": value["rank"],
        }
        for value in candidates
    ]
    selected_report = {key: value for key, value in selected.items() if key != "rank"}
    report = {
        "schema_version": 1,
        "scope": "development-only train-fitted DOFA-v2 normalization confirmation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_folds": list(SELECTION_FOLDS),
        "fixed_candidate": fixed,
        "normalization_contract": {
            "modes": list(NORMALIZATION_MODES),
            "fit_statistics": "fit-fold rows only",
            "held_statistics_accessed": False,
            "label_free_at_inference": True,
            "sample_independent_at_inference": True,
        },
        "candidate_summaries": summaries,
        "selected": selected_report,
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the selected train-fitted DOFA-v2 fusion for one-shot fold-2 extraction."
            if passed
            else "Reject deployable train-fitted DOFA-v2 fusion before fold-2 extraction."
        ),
        "provenance": {
            **{f"{name}_sha256": digest for name, digest in expected.items()},
            "protocol_sha256": sha256(protocol_path),
            "protected_report_sha256": sha256(protected_report_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    delta = selected["evaluation"]["versus_current"]["delta"]
    print(
        json.dumps(
            {
                "ok": passed,
                "normalization_mode": selected["normalization_mode"],
                "ap_delta_vs_current": delta["average_precision"],
                "recall_delta_vs_current": delta["recall_at_fpr_0_0713"],
                "ap_lower_vs_current": selected[
                    "paired_group_bootstrap_ap_delta_vs_current"
                ]["lower"],
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
