#!/usr/bin/env python3
"""Cross-fit a target-adapted ExtraTrees and XGBoost consensus scene head."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import xgboost

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_crossfold_bagged_scene_head import (  # noqa: E402
    DEFAULT_FOLD0_CACHE, DEFAULT_FOLD0_SHA256, DEFAULT_FOLD1_CACHE,
    DEFAULT_FOLD1_SHA256, DEFAULT_INNER_CACHE, DEFAULT_INNER_SHA256,
    DEFAULT_SCORE_CACHE, DEFAULT_SCORE_SHA256, load_development,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_target_weighted_scene_head import (  # noqa: E402
    WEIGHT_SPECS, evaluate_candidate, oof_scores as target_oof,
)
from train_mars_xgboost_scene_head import MODEL_SPECS, oof_scores as xgboost_oof  # noqa: E402

DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_target_xgboost_consensus.joblib")
DEFAULT_JSON = Path("reports/experiments/mars_target_xgboost_consensus.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_TARGET_XGBOOST_CONSENSUS.md")
TARGET_SPEC = WEIGHT_SPECS[0]
XGBOOST_SPEC = MODEL_SPECS[0]
TARGET_WEIGHTS = (0.2, 0.3, 0.4, 0.5)
XGBOOST_WEIGHTS = (0.05, 0.1, 0.2)


def consensus_scores(current: np.ndarray, target: np.ndarray, boosted: np.ndarray,
                     target_weight: float, boosted_weight: float) -> np.ndarray:
    """Combine three probabilities with direct convex weights in logit space."""
    current_weight = 1.0 - target_weight - boosted_weight
    if current_weight <= 0.0:
        raise ValueError("Consensus weights must leave positive current-head weight")
    arrays = [np.asarray(value, dtype=np.float64) for value in (current, target, boosted)]
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError("Consensus score arrays do not align")
    logits = [np.log(np.clip(value, 1e-6, 1 - 1e-6) / np.clip(1 - value, 1e-6, 1)) for value in arrays]
    combined = current_weight * logits[0] + target_weight * logits[1] + boosted_weight * logits[2]
    scores = 1 / (1 + np.exp(-combined))
    if not np.isfinite(scores).all():
        raise RuntimeError("Consensus scores are non-finite")
    return scores


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["versus_current"]["delta"]
    interval = selected["paired_group_bootstrap_ap_delta_vs_current"]
    lines = [
        "# Target-adapted XGBoost consensus", "",
        "Each held fold is unlabeled to its density-ratio estimator; both component learners exclude that fold's labels.", "",
        f"- Logit weights (current / target / XGBoost): {selected['current_weight']:.2f} / {selected['target_weight']:.2f} / {selected['xgboost_weight']:.2f}",
        f"- AP delta vs current: {delta['average_precision']:+.5f}",
        f"- Recall delta vs current: {delta['recall_at_fpr_0_0713']:+.5f}",
        f"- Paired-site AP interval vs current: [{interval['lower']:+.5f}, {interval['upper']:+.5f}]", "",
        "| Fold | AP delta | Recall delta |", "|---|---:|---:|",
    ]
    for fold, value in selected["per_fold"].items():
        local = value["versus_current"]["delta"]
        lines.append(f"| {fold} | {local['average_precision']:+.5f} | {local['recall_at_fpr_0_0713']:+.5f} |")
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-cache", default=DEFAULT_INNER_CACHE.as_posix())
    parser.add_argument("--inner-sha256", default=DEFAULT_INNER_SHA256)
    parser.add_argument("--fold0-cache", default=DEFAULT_FOLD0_CACHE.as_posix())
    parser.add_argument("--fold0-sha256", default=DEFAULT_FOLD0_SHA256)
    parser.add_argument("--fold1-cache", default=DEFAULT_FOLD1_CACHE.as_posix())
    parser.add_argument("--fold1-sha256", default=DEFAULT_FOLD1_SHA256)
    parser.add_argument("--score-cache", default=DEFAULT_SCORE_CACHE.as_posix())
    parser.add_argument("--score-sha256", default=DEFAULT_SCORE_SHA256)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {"inner": (root / args.inner_cache).resolve(), "fold0": (root / args.fold0_cache).resolve(),
             "fold1": (root / args.fold1_cache).resolve(), "score": (root / args.score_cache).resolve()}
    expected = {"inner": args.inner_sha256, "fold0": args.fold0_sha256,
                "fold1": args.fold1_sha256, "score": args.score_sha256}
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} cache hash mismatch")
    values = load_development({name: paths[name] for name in ("inner", "fold0", "fold1")}, paths["score"])
    target, domain_audits = target_oof(values, TARGET_SPEC)
    boosted = xgboost_oof(values, XGBOOST_SPEC)
    candidates, score_store = [], {}
    for target_weight in TARGET_WEIGHTS:
        for boosted_weight in XGBOOST_WEIGHTS:
            scores = consensus_scores(values["current"], target, boosted, target_weight, boosted_weight)
            key = f"target-{target_weight}_xgboost-{boosted_weight}"
            candidate = evaluate_candidate(values, scores, TARGET_SPEC, 1.0)
            candidate.update({"key": key, "target_weight": target_weight,
                              "xgboost_weight": boosted_weight,
                              "current_weight": 1 - target_weight - boosted_weight})
            candidate["rank"] = [int(candidate["stable"]),
                min(f["versus_current"]["delta"]["recall_at_fpr_0_0713"] for f in candidate["per_fold"].values()),
                min(f["versus_current"]["delta"]["average_precision"] for f in candidate["per_fold"].values()),
                candidate["versus_current"]["delta"]["average_precision"], -target_weight - boosted_weight]
            candidates.append(candidate)
            score_store[key] = scores
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_scores = score_store[selected["key"]]
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(values["labels"], values["primary"], selected_scores, values["groups"], replicates=10_000, seed=20261480)
    selected["paired_group_bootstrap_ap_delta_vs_current"] = ap_group_bootstrap(values["labels"], values["current"], selected_scores, values["groups"], replicates=10_000, seed=20261481)
    passed = bool(selected["stable"] and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"] > 0 and selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"] > 0)
    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    joblib.dump({"schema_version": 1, "kind": "mars_target_xgboost_consensus_control",
                 "target_spec": TARGET_SPEC, "xgboost_spec": XGBOOST_SPEC,
                 "target_weight": selected["target_weight"], "xgboost_weight": selected["xgboost_weight"],
                 "current_weight": selected["current_weight"],
                 "label_contract": "target labels are never used for adaptation"}, temporary, compress=3)
    os.replace(temporary, artifact_path)
    report = {"schema_version": 1, "scope": "five-fold development-only target/XGBoost consensus; paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "target_spec": TARGET_SPEC,
        "xgboost_spec": XGBOOST_SPEC, "target_weights": list(TARGET_WEIGHTS), "xgboost_weights": list(XGBOOST_WEIGHTS),
        "domain_audits": domain_audits,
        "candidate_summaries": [{"key": v["key"], "stable": v["stable"],
            "ap_delta_vs_current": v["versus_current"]["delta"]["average_precision"],
            "recall_delta_vs_current": v["versus_current"]["delta"]["recall_at_fpr_0_0713"],
            "worst_fold_ap_delta": min(f["versus_current"]["delta"]["average_precision"] for f in v["per_fold"].values()),
            "worst_fold_recall_delta": min(f["versus_current"]["delta"]["recall_at_fpr_0_0713"] for f in v["per_fold"].values())} for v in candidates],
        "selected": selected, "all_promotion_gates_pass": passed,
        "decision": "Freeze consensus for one label-free paper adaptation and replay." if passed else "Reject consensus before paper adaptation.",
        "provenance": {**{f"{name}_cache_sha256": digest for name, digest in expected.items()},
            "artifact_sha256": sha256(artifact_path), "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "numpy": np.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}}
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(json.dumps({"ok": passed, "weights": [selected["current_weight"], selected["target_weight"], selected["xgboost_weight"]],
        "ap_delta_vs_current": selected["versus_current"]["delta"]["average_precision"],
        "recall_delta_vs_current": selected["versus_current"]["delta"]["recall_at_fpr_0_0713"],
        "ap_lower_vs_current": selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"],
        "artifact_sha256": report["provenance"]["artifact_sha256"]}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
