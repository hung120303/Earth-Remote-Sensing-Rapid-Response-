#!/usr/bin/env python3
"""One-shot fold-2 confirmation of the fixed protected DOFA+anchored ensemble."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from confirm_mars_dofa_v2_projection_ensemble import mean_logit_probabilities  # noqa: E402
from evaluate_mars_dofa_anchored_protected_ensemble import (  # noqa: E402
    load_anchored_scores,
    protected_residual_ensemble,
)
from evaluate_mars_dofa_v2_fold2 import align_fold_cache, fit_predict_fold2  # noqa: E402
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_dofa_v2_protected_fusion import protected_logit_blend  # noqa: E402
from train_mars_dofa_v2_scene_probe import align_features, select_features  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_dofa_anchored_protected_fold2_protocol.json")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    delta = report["versus_current"]["delta"]
    interval = report["paired_site_ap_delta"]
    dense = report["dense_confirmation"]
    lines = [
        "# One-shot protected DOFA + anchored fold-2 confirmation",
        "",
        f"- All confirmation gates pass: **{report['all_confirmation_gates_pass']}**",
        f"- AP delta: **{delta['average_precision']:+.6f}**",
        f"- Matched-FPR recall delta: **{delta['recall_at_fpr_0_0713']:+.6f}**",
        f"- Paired-site AP interval: **[{interval['lower']:+.6f}, {interval['upper']:+.6f}]**",
        f"- Dense IoU delta / lower bound: **{dense['pixel_iou_delta']:+.6f} / {dense['paired_site_lower']:+.6f}**",
        "",
        report["decision"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not str(protocol["status"]).startswith("frozen"):
        raise ValueError("Fold-2 ensemble confirmation requires a frozen protocol")
    if sha256(Path(__file__).resolve()) != protocol["evaluator"]["sha256"]:
        raise ValueError("Frozen fold-2 ensemble evaluator hash mismatch")
    for dependency in protocol["code_dependencies"]:
        if sha256((ROOT / dependency["path"]).resolve()) != dependency["sha256"]:
            raise ValueError(f"Frozen fold-2 dependency mismatch: {dependency['path']}")
    fixed = protocol["candidate"]
    if fixed != {
        "anchored_strength": 0.5,
        "anchored_multiplier": 1.0,
        "final_gate": 0.25,
        "dense_strength": 0.1,
        "dofa_feature_set": "change_extreme",
        "dofa_C": 0.01,
        "dofa_projection_seeds": [20260780, 20260781, 20260782, 20260783, 20260784],
        "dofa_gate": 0.5,
        "dofa_weight": 0.05,
        "fit_folds": [3, 4],
        "held_fold": 2,
    }:
        raise ValueError("Fold-2 ensemble candidate differs from selection")
    paths = {name: (ROOT / contract["path"]).resolve() for name, contract in protocol["inputs"].items()}
    for name, contract in protocol["inputs"].items():
        if sha256(paths[name]) != contract["sha256"]:
            raise ValueError(f"Frozen fold-2 input hash mismatch: {name}")
    selection_report = json.loads(paths["selection_result"].read_text(encoding="utf-8"))
    selection = selection_report["selected"]
    if not selection_report["all_promotion_gates_pass"] or (
        float(selection["anchored_strength"]),
        float(selection["anchored_multiplier"]),
    ) != (float(fixed["anchored_strength"]), float(fixed["anchored_multiplier"])):
        raise ValueError("Fold-2 candidate differs from the passed selection result")
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_markdown = (ROOT / protocol["outputs"]["markdown"]).resolve()
    if output_json.exists() or output_markdown.exists():
        raise FileExistsError("Refusing to repeat one-shot ensemble fold-2 confirmation")

    all_values = load_development(
        {name: paths[name] for name in ("inner", "fold0", "fold1")}, paths["current_scores"]
    )
    source_encoded, source_names = align_features(paths["dofa_development"], all_values)
    target_encoded, target_names = align_fold_cache(paths["dofa_fold2"], all_values, 2)
    if not np.array_equal(source_names, target_names):
        raise ValueError("DOFA development and fold-2 schemas differ")
    source_rows = np.isin(all_values["folds"], fixed["fit_folds"])
    target_rows = np.asarray(all_values["folds"]) == int(fixed["held_fold"])
    source_features, _ = select_features(source_encoded, source_names, fixed["dofa_feature_set"])
    target_features, _ = select_features(target_encoded, target_names, fixed["dofa_feature_set"])
    raw_scores = []
    for seed in map(int, fixed["dofa_projection_seeds"]):
        raw_scores.append(
            fit_predict_fold2(
                source_features,
                target_features,
                np.asarray(all_values["labels"])[source_rows],
                np.asarray(all_values["sensors"])[source_rows],
                np.asarray(all_values["sensors"])[target_rows],
                seed=seed,
                c_value=float(fixed["dofa_C"]),
            )
        )
        gc.collect()
    aggregate = mean_logit_probabilities(raw_scores)
    values = {
        key: np.asarray(all_values[key])[target_rows]
        for key in ("labels", "sensors", "sample_ids", "groups", "folds", "primary", "current")
    }
    dofa = protected_logit_blend(
        values["current"], aggregate, gate=float(fixed["dofa_gate"]), weight=float(fixed["dofa_weight"])
    )
    _, anchored = load_anchored_scores(
        paths["anchored_scores"], values, protocol["anchored_cache_protocol_sha256"]
    )
    scores = protected_residual_ensemble(
        values["current"], anchored[float(fixed["anchored_strength"])], dofa,
        gate=float(fixed["final_gate"]), anchored_multiplier=float(fixed["anchored_multiplier"]),
    )
    candidate_metrics = metric_summary(values["labels"], scores, values["sensors"])
    current_metrics = metric_summary(values["labels"], values["current"], values["sensors"])
    primary_metrics = metric_summary(values["labels"], values["primary"], values["sensors"])
    versus_current = comparison(candidate_metrics, current_metrics)
    versus_primary = comparison(candidate_metrics, primary_metrics)
    interval = ap_group_bootstrap(
        values["labels"], values["current"], scores, values["groups"].astype(str),
        replicates=int(protocol["bootstrap"]["replicates"]), seed=int(protocol["bootstrap"]["seed"]),
    )
    anchored_report = json.loads(paths["anchored_result"].read_text(encoding="utf-8"))
    dense_row = next(
        row for row in anchored_report["candidates"]
        if float(row["strength"]) == float(fixed["dense_strength"])
    )
    dense = {
        "strength": float(fixed["dense_strength"]),
        "pixel_iou_delta": float(dense_row["pixel_iou_delta"]),
        "paired_site_lower": float(dense_row["paired_site_pixel_iou_delta"]["lower"]),
    }
    preserved = all(candidate_metrics[key] == current_metrics[key] for key in ("tp", "fp", "tn", "fn"))
    gates = protocol["gates"]
    checks = {
        "minimum_ap_delta": versus_current["delta"]["average_precision"] >= float(gates["minimum_ap_delta_vs_current"]),
        "recall_no_worse": versus_current["delta"]["recall_at_fpr_0_0713"] >= 0.0,
        "fpr_no_worse": candidate_metrics["false_positive_rate"] <= current_metrics["false_positive_rate"],
        "operating_counts_preserved": preserved,
        "each_sensor_ap_positive": min(versus_current["delta"]["sensor_average_precision"].values()) > 0.0,
        "paired_site_ap_lower_positive": interval["lower"] > 0.0,
        "dense_iou_positive": dense["pixel_iou_delta"] > 0.0,
        "dense_paired_lower_positive": dense["paired_site_lower"] > 0.0,
        "ap_vs_primary_positive": versus_primary["delta"]["average_precision"] > 0.0,
        "recall_vs_primary_positive": versus_primary["delta"]["recall_at_fpr_0_0713"] > 0.0,
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "scope": protocol["scope"],
        "status": "passed_fold2_confirmation" if passed else "rejected_on_fold2_confirmation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(protocol_path),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidate": fixed,
        "rows": {"total": int(values["labels"].size), "positive": int(values["labels"].sum()), "groups": int(len(np.unique(values["groups"])))},
        "metrics": candidate_metrics,
        "versus_current": versus_current,
        "versus_released_primary": versus_primary,
        "paired_site_ap_delta": interval,
        "dense_confirmation": dense,
        "operating_counts_preserved": preserved,
        "confirmation_checks": checks,
        "all_confirmation_gates_pass": passed,
        "folds_0_1_accessed": False,
        "external_or_official_test_accessed": False,
        "decision": (
            "Authorize a separately frozen new-seed/multi-fold confirmation; official test remains closed."
            if passed else
            "Reject the fixed ensemble before folds 0/1, external, or official-test evaluation."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_json)
    write_markdown(output_markdown, report)
    print(json.dumps({"ok": passed, "ap_delta": versus_current["delta"]["average_precision"], "ap_lower": interval["lower"], "dense_iou_delta": dense["pixel_iou_delta"]}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
