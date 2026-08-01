#!/usr/bin/env python3
"""Evaluate a preregistered protected DOFA-v2 plus anchored-U-Net ensemble."""

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
from confirm_mars_dofa_v2_train_fitted_normalization import (  # noqa: E402
    build_source_fitted_views,
)
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_dofa_v2_protected_fusion import protected_logit_blend  # noqa: E402
from train_mars_dofa_v2_scene_probe import (  # noqa: E402
    align_features,
    crossfit_scores,
    select_features,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_dofa_anchored_protected_ensemble_protocol.json")
SELECTION_FOLDS = (3, 4)


def local_logit(probability: np.ndarray, gate: float) -> np.ndarray:
    local = (np.asarray(probability, dtype=np.float64) - gate) / (1.0 - gate)
    clipped = np.clip(local, 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def protected_residual_ensemble(
    current: np.ndarray,
    anchored: np.ndarray,
    dofa: np.ndarray,
    *,
    gate: float,
    anchored_multiplier: float,
) -> np.ndarray:
    """Add two protected local-logit residuals without crossing the operating gate."""
    current_values = np.asarray(current, dtype=np.float64)
    anchored_values = np.asarray(anchored, dtype=np.float64)
    dofa_values = np.asarray(dofa, dtype=np.float64)
    if not (
        current_values.shape == anchored_values.shape == dofa_values.shape
        and current_values.ndim == 1
    ):
        raise ValueError("Protected ensemble inputs must be aligned vectors")
    if not 0.0 < gate < 1.0 or anchored_multiplier < 0.0:
        raise ValueError("Invalid protected ensemble gate or multiplier")
    if not all(
        np.isfinite(values).all() and (values >= 0.0).all() and (values <= 1.0).all()
        for values in (current_values, anchored_values, dofa_values)
    ):
        raise ValueError("Protected ensemble inputs must be finite probabilities")
    eligible = current_values >= gate
    if eligible.any() and (
        (anchored_values[eligible] < gate).any() or (dofa_values[eligible] < gate).any()
    ):
        raise ValueError("A component crossed the final protection gate")
    result = current_values.copy()
    if eligible.any():
        current_logit = local_logit(current_values[eligible], gate)
        anchored_residual = local_logit(anchored_values[eligible], gate) - current_logit
        dofa_residual = local_logit(dofa_values[eligible], gate) - current_logit
        combined = current_logit + anchored_multiplier * anchored_residual + dofa_residual
        local = np.where(
            combined >= 0.0,
            1.0 / (1.0 + np.exp(-combined)),
            np.exp(combined) / (1.0 + np.exp(combined)),
        )
        result[eligible] = gate + (1.0 - gate) * local
    if not np.array_equal(result[~eligible], current_values[~eligible]):
        raise RuntimeError("Protected ensemble altered a below-gate score")
    if eligible.any() and float(result[eligible].min()) < gate:
        raise RuntimeError("Protected ensemble crossed its gate")
    return result


def evaluate_scores(values: dict[str, np.ndarray], scores: np.ndarray) -> dict[str, Any]:
    candidate = metric_summary(values["labels"], scores, values["sensors"])
    current = metric_summary(values["labels"], values["current"], values["sensors"])
    primary = metric_summary(values["labels"], values["primary"], values["sensors"])
    per_fold = {}
    for fold in SELECTION_FOLDS:
        rows = values["folds"] == fold
        local = metric_summary(values["labels"][rows], scores[rows], values["sensors"][rows])
        per_fold[str(fold)] = {
            "versus_current": comparison(
                local,
                metric_summary(
                    values["labels"][rows], values["current"][rows], values["sensors"][rows]
                ),
            ),
            "versus_primary": comparison(
                local,
                metric_summary(
                    values["labels"][rows], values["primary"][rows], values["sensors"][rows]
                ),
            ),
        }
    return {
        "versus_current": comparison(candidate, current),
        "versus_primary": comparison(candidate, primary),
        "per_fold": per_fold,
        "operating_counts_preserved": all(
            candidate[key] == current[key] for key in ("tp", "fp", "tn", "fn")
        ),
    }


def fixed_dofa_scores(
    protocol: dict[str, Any], values: dict[str, np.ndarray], encoded: np.ndarray, names: list[str]
) -> np.ndarray:
    fixed = protocol["fixed_dofa"]
    features, _ = select_features(encoded, names, str(fixed["feature_set"]))
    raw_scores = []
    for seed in map(int, fixed["projection_seeds"]):
        views = build_source_fitted_views(
            features,
            values["folds"],
            values["sensors"],
            seed=seed,
            mode=str(fixed["normalization_mode"]),
        )
        raw_scores.append(crossfit_scores(views, values["labels"], float(fixed["C"])))
        del views
        gc.collect()
    aggregate = mean_logit_probabilities(raw_scores)
    return protected_logit_blend(
        values["current"],
        aggregate,
        gate=float(fixed["gate"]),
        weight=float(fixed["weight"]),
    )


def load_anchored_scores(
    path: Path, values: dict[str, np.ndarray], expected_protocol_sha256: str
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    with np.load(path, allow_pickle=False) as cache:
        if str(cache["protocol_sha256"].item()) != expected_protocol_sha256:
            raise ValueError("Anchored cache protocol binding differs")
        ids = cache["sample_ids"].astype(str)
        if len(set(ids.tolist())) != ids.size:
            raise ValueError("Anchored cache contains duplicate sample IDs")
        index = {sample_id: position for position, sample_id in enumerate(ids.tolist())}
        try:
            order = np.asarray([index[str(value)] for value in values["sample_ids"]])
        except KeyError as exc:
            raise ValueError("Anchored cache is missing a development identity") from exc
        for key in ("labels", "sensors", "groups", "folds"):
            observed = np.asarray(cache[key])[order]
            expected = np.asarray(values[key])
            if not np.array_equal(observed.astype(str), expected.astype(str)):
                raise ValueError(f"Anchored cache {key} contract differs")
        base = np.asarray(cache["base_scores"], dtype=np.float64)[order]
        if not np.allclose(base, values["current"], rtol=0.0, atol=1e-7):
            raise ValueError("Anchored cache base scores differ from current scores")
        strengths = np.asarray(cache["strengths"], dtype=np.float64)
        candidates = {
            float(strength): np.asarray(cache[f"candidate_{position}"], dtype=np.float64)[order]
            for position, strength in enumerate(strengths)
        }
    return base, candidates


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["evaluation"]["versus_current"]["delta"]
    interval = selected["paired_site_ap_delta"]
    lines = [
        "# Protected DOFA-v2 + anchored-U-Net ensemble",
        "",
        f"- Promotion gates pass: **{report['all_promotion_gates_pass']}**",
        f"- Anchored source strength / multiplier: **{selected['anchored_strength']} / {selected['anchored_multiplier']}**",
        f"- AP delta: **{delta['average_precision']:+.6f}**",
        f"- Matched-FPR recall delta: **{delta['recall_at_fpr_0_0713']:+.6f}**",
        f"- Paired-site AP interval: **[{interval['lower']:+.6f}, {interval['upper']:+.6f}]**",
        f"- Fixed dense IoU delta / lower bound: **{report['fixed_dense_evidence']['pixel_iou_delta']:+.6f} / {report['fixed_dense_evidence']['paired_site_lower']:+.6f}**",
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
        raise ValueError("Ensemble outcome evaluation requires a frozen protocol")
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen protected-ensemble trainer hash mismatch")
    for dependency in protocol.get("code_dependencies", []):
        if sha256((ROOT / dependency["path"]).resolve()) != dependency["sha256"]:
            raise ValueError(f"Frozen protected-ensemble dependency mismatch: {dependency['path']}")
    expected_grid = [(float(s), float(m)) for s in (0.25, 0.5, 1.0) for m in (0.5, 1.0)]
    observed_grid = [
        (float(s), float(m))
        for s in protocol["search"]["anchored_strengths"]
        for m in protocol["search"]["anchored_multipliers"]
    ]
    if observed_grid != expected_grid:
        raise ValueError("Protected ensemble candidate grid differs")
    paths = {name: (ROOT / contract["path"]).resolve() for name, contract in protocol["inputs"].items()}
    for name, contract in protocol["inputs"].items():
        if sha256(paths[name]) != contract["sha256"]:
            raise ValueError(f"Frozen ensemble input hash mismatch: {name}")

    all_values = load_development(
        {name: paths[name] for name in ("inner", "fold0", "fold1")}, paths["current_scores"]
    )
    dofa_report = json.loads(paths["dofa_result"].read_text(encoding="utf-8"))
    dofa_selected = dofa_report["selected"]
    fixed_dofa = protocol["fixed_dofa"]
    if not dofa_report["all_promotion_gates_pass"] or (
        str(dofa_selected["normalization_mode"]),
        float(dofa_selected["evaluation"]["spec"]["gate"]),
        float(dofa_selected["evaluation"]["spec"]["weight"]),
    ) != (
        str(fixed_dofa["normalization_mode"]),
        float(fixed_dofa["gate"]),
        float(fixed_dofa["weight"]),
    ):
        raise ValueError("Fixed DOFA candidate differs from its passed dependency")
    selection = np.isin(all_values["folds"], SELECTION_FOLDS)
    values = {
        key: np.asarray(all_values[key])[selection]
        for key in ("labels", "sensors", "sample_ids", "groups", "folds", "primary", "current")
    }
    encoded, names = align_features(paths["dofa_features"], all_values)
    dofa = fixed_dofa_scores(protocol, values, encoded, names)
    del encoded
    gc.collect()
    _, anchored_by_strength = load_anchored_scores(
        paths["anchored_scores"], values, protocol["anchored_cache_protocol_sha256"]
    )

    dense_report = json.loads(paths["anchored_result"].read_text(encoding="utf-8"))
    dense_strength = float(protocol["fixed_dense"]["strength"])
    dense = next(
        row for row in dense_report["candidates"] if float(row["strength"]) == dense_strength
    )
    fixed_dense = {
        "strength": dense_strength,
        "pixel_iou_delta": float(dense["pixel_iou_delta"]),
        "paired_site_lower": float(dense["paired_site_pixel_iou_delta"]["lower"]),
        "fold_iou_delta": {
            fold: float(dense["by_fold"][fold]["pixel_iou_delta"])
            for fold in map(str, SELECTION_FOLDS)
        },
    }
    dense_pass = bool(
        fixed_dense["pixel_iou_delta"] > 0.0
        and fixed_dense["paired_site_lower"] > 0.0
        and min(fixed_dense["fold_iou_delta"].values()) > 0.0
    )

    candidates = []
    gate = float(protocol["architecture"]["final_protection_gate"])
    gates = protocol["gates"]
    for anchored_strength, anchored_multiplier in observed_grid:
        scores = protected_residual_ensemble(
            values["current"],
            anchored_by_strength[anchored_strength],
            dofa,
            gate=gate,
            anchored_multiplier=anchored_multiplier,
        )
        evaluation = evaluate_scores(values, scores)
        interval = ap_group_bootstrap(
            values["labels"], values["current"], scores, values["groups"],
            replicates=int(protocol["bootstrap"]["replicates"]),
            seed=int(protocol["bootstrap"]["seed"]),
        )
        delta = evaluation["versus_current"]["delta"]
        fold_ap = [
            evaluation["per_fold"][fold]["versus_current"]["delta"]["average_precision"]
            for fold in map(str, SELECTION_FOLDS)
        ]
        sensor_ap = list(delta["sensor_average_precision"].values())
        passed = bool(
            delta["average_precision"] >= float(gates["average_precision_delta_minimum"])
            and delta["recall_at_fpr_0_0713"] >= 0.0
            and min(fold_ap) > 0.0
            and min(sensor_ap) > 0.0
            and interval["lower"] > 0.0
            and evaluation["operating_counts_preserved"]
            and dense_pass
        )
        candidates.append({
            "anchored_strength": anchored_strength,
            "anchored_multiplier": anchored_multiplier,
            "evaluation": evaluation,
            "paired_site_ap_delta": interval,
            "passed": passed,
            "rank": [
                int(passed), min(fold_ap), interval["lower"], min(sensor_ap),
                delta["average_precision"], -(anchored_strength * anchored_multiplier),
            ],
        })
    selected = max(candidates, key=lambda row: tuple(row["rank"]))
    passed = bool(selected["passed"])
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "rejected",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": protocol["scope"],
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(protocol_path),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "fixed_dofa": protocol["fixed_dofa"],
        "fixed_dense_evidence": fixed_dense,
        "candidate_summaries": [
            {
                "anchored_strength": row["anchored_strength"],
                "anchored_multiplier": row["anchored_multiplier"],
                "ap_delta": row["evaluation"]["versus_current"]["delta"]["average_precision"],
                "recall_delta": row["evaluation"]["versus_current"]["delta"]["recall_at_fpr_0_0713"],
                "fold_ap_delta": {
                    fold: row["evaluation"]["per_fold"][fold]["versus_current"]["delta"]["average_precision"]
                    for fold in map(str, SELECTION_FOLDS)
                },
                "sensor_ap_delta": row["evaluation"]["versus_current"]["delta"]["sensor_average_precision"],
                "ap_interval": row["paired_site_ap_delta"],
                "passed": row["passed"],
            }
            for row in candidates
        ],
        "selected": {key: value for key, value in selected.items() if key != "rank"},
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the selected ensemble for a new fold-2 confirmation; folds 0/1 and official test remain closed."
            if passed else
            "Reject the protected independent-signal ensemble before fold 2 or official-test scoring."
        ),
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_json)
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({
        "ok": passed,
        "anchored_strength": selected["anchored_strength"],
        "anchored_multiplier": selected["anchored_multiplier"],
        "ap_delta": selected["evaluation"]["versus_current"]["delta"]["average_precision"],
        "ap_lower": selected["paired_site_ap_delta"]["lower"],
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
