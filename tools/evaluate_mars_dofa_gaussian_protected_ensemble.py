#!/usr/bin/env python3
"""Evaluate a preregistered DOFA plus conservative Gaussian-ViT scene ensemble."""

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
from evaluate_mars_dofa_anchored_protected_ensemble import (  # noqa: E402
    evaluate_scores,
    fixed_dofa_scores,
    local_logit,
    protected_residual_ensemble,
)
from train_mars_dofa_v2_scene_probe import align_features  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_dofa_gaussian_protected_ensemble_protocol.json")
SELECTION_FOLDS = (3, 4)


def restricted_selection_values(inner_path: Path, score_path: Path) -> dict[str, np.ndarray]:
    """Load folds 3/4 without opening the separate fold-0/fold-1 caches."""
    with np.load(inner_path, allow_pickle=False) as inner:
        values = {
            key: np.asarray(inner[key])
            for key in ("labels", "sensors", "sample_ids", "groups", "folds")
        }
    with np.load(score_path, allow_pickle=False) as scores:
        for key in ("labels", "sensors", "groups", "folds"):
            if not np.array_equal(values[key].astype(str), scores[f"inner_{key}"].astype(str)):
                raise ValueError(f"Inner score-cache {key} contract differs")
        primary = np.asarray(scores["inner_primary"], dtype=np.float64)
        current = np.asarray(scores["inner_new"], dtype=np.float64)
    selected = np.isin(values["folds"], SELECTION_FOLDS)
    result = {key: np.asarray(value)[selected] for key, value in values.items()}
    result["primary"] = primary[selected]
    result["current"] = current[selected]
    if result["labels"].size != 17745 or set(map(int, result["folds"])) != set(SELECTION_FOLDS):
        raise ValueError("Restricted folds-3/4 row contract differs")
    return result


def load_gaussian_scene_cache(
    path: Path,
    values: dict[str, np.ndarray],
    expected_protocol_sha256: str,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as cache:
        if str(cache["protocol_sha256"].item()) != expected_protocol_sha256:
            raise ValueError("Gaussian scene cache protocol binding differs")
        ids = cache["sample_ids"].astype(str)
        if len(set(ids.tolist())) != ids.size:
            raise ValueError("Gaussian scene cache identities are not unique")
        index = {sample_id: position for position, sample_id in enumerate(ids.tolist())}
        try:
            order = np.asarray([index[str(value)] for value in values["sample_ids"]])
        except KeyError as exc:
            raise ValueError("Gaussian scene cache is missing a selection identity") from exc
        for key in ("labels", "sensors", "groups", "folds"):
            if not np.array_equal(
                np.asarray(cache[key])[order].astype(str), np.asarray(values[key]).astype(str)
            ):
                raise ValueError(f"Gaussian scene cache {key} contract differs")
        base = np.asarray(cache["base_scores"], dtype=np.float64)[order]
        if not np.allclose(base, values["current"], rtol=0.0, atol=1e-7):
            raise ValueError("Gaussian scene cache base scores differ")
        raw = np.asarray(cache["raw_scene_logits"], dtype=np.float64)[order]
    if not np.isfinite(raw).all():
        raise ValueError("Gaussian scene logits are non-finite")
    return raw


def gaussian_local_candidate(
    current: np.ndarray,
    raw_scene_logits: np.ndarray,
    *,
    strength: float,
    gate: float,
) -> np.ndarray:
    """Apply bounded Gaussian evidence entirely inside the above-gate interval."""
    base = np.asarray(current, dtype=np.float64)
    raw = np.asarray(raw_scene_logits, dtype=np.float64)
    if base.shape != raw.shape or base.ndim != 1:
        raise ValueError("Gaussian candidate inputs must be aligned vectors")
    eligible = base >= float(gate)
    result = base.copy()
    if eligible.any():
        combined = local_logit(base[eligible], gate) + float(strength) * 2.0 * np.tanh(
            raw[eligible] / 2.0
        )
        local = np.where(
            combined >= 0.0,
            1.0 / (1.0 + np.exp(-combined)),
            np.exp(combined) / (1.0 + np.exp(combined)),
        )
        result[eligible] = gate + (1.0 - gate) * local
    if not np.array_equal(result[~eligible], base[~eligible]):
        raise RuntimeError("Gaussian candidate altered a below-gate score")
    if eligible.any() and float(result[eligible].min()) < gate:
        raise RuntimeError("Gaussian candidate crossed the protection gate")
    return result


def dense_evidence(paths: dict[str, Path], strength: float) -> dict[str, Any]:
    development = json.loads(paths["anchored_development_result"].read_text(encoding="utf-8"))
    confirmation = json.loads(paths["anchored_fold2_result"].read_text(encoding="utf-8"))
    development_row = next(
        row for row in development["candidates"] if float(row["strength"]) == strength
    )
    confirmation_row = next(
        row for row in confirmation["candidates"] if float(row["strength"]) == strength
    )
    result = {
        "strength": strength,
        "development_iou_delta": float(development_row["pixel_iou_delta"]),
        "development_paired_lower": float(
            development_row["paired_site_pixel_iou_delta"]["lower"]
        ),
        "fold2_iou_delta": float(confirmation_row["pixel_iou_delta"]),
        "fold2_paired_lower": float(
            confirmation_row["paired_site_pixel_iou_delta"]["lower"]
        ),
    }
    result["passed"] = all(float(result[key]) > 0.0 for key in result if key != "strength")
    return result


def validate_fixed_dofa_result(path: Path, fixed: dict[str, Any]) -> None:
    """Bind the reused DOFA residual to its previously selected specification."""
    report = json.loads(path.read_text(encoding="utf-8"))
    candidate = report["fixed_candidate"]
    selected = report["selected"]
    selected_spec = selected["evaluation"]["spec"]
    expected = {
        "feature_set": str(candidate["feature_set"]),
        "C": float(candidate["C"]),
        "projection_seeds": list(map(int, candidate["projection_seeds"])),
        "normalization_mode": str(selected["normalization_mode"]),
        "gate": float(selected_spec["gate"]),
        "weight": float(selected_spec["weight"]),
    }
    observed = {
        "feature_set": str(fixed["feature_set"]),
        "C": float(fixed["C"]),
        "projection_seeds": list(map(int, fixed["projection_seeds"])),
        "normalization_mode": str(fixed["normalization_mode"]),
        "gate": float(fixed["gate"]),
        "weight": float(fixed["weight"]),
    }
    if observed != expected or not bool(report["all_promotion_gates_pass"]):
        raise ValueError("Fixed DOFA candidate differs from its passed selection report")


def validate_gaussian_replicate_result(
    result_path: Path,
    cache_path: Path,
    expected_protocol_sha256: str,
) -> tuple[dict[str, Any], set[float]]:
    report = json.loads(result_path.read_text(encoding="utf-8"))
    if str(report["protocol_sha256"]) != expected_protocol_sha256:
        raise ValueError("Gaussian replicate result protocol binding differs")
    if str(report["scene_cache"]["sha256"]) != sha256(cache_path):
        raise ValueError("Gaussian replicate result cache binding differs")
    eligible = set(map(float, report.get("eligible_strengths", [])))
    if bool(report.get("eligible_for_preregistered_ensemble")) != bool(eligible):
        raise ValueError("Gaussian replicate eligibility contract differs")
    return report, eligible


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["evaluation"]["versus_current"]["delta"]
    interval = selected["paired_site_ap_delta"]
    lines = [
        "# Protected DOFA + conservative Gaussian-ViT ensemble",
        "",
        f"- Promotion gates pass: **{report['all_promotion_gates_pass']}**",
        f"- Gaussian strength: **{selected['gaussian_strength']}**",
        f"- AP delta: **{delta['average_precision']:+.6f}**",
        f"- Matched-FPR recall delta: **{delta['recall_at_fpr_0_0713']:+.6f}**",
        f"- Paired-site AP interval: **[{interval['lower']:+.6f}, {interval['upper']:+.6f}]**",
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
        raise ValueError("Gaussian ensemble evaluation requires a frozen protocol")
    if sha256(Path(__file__).resolve()) != protocol["evaluator"]["sha256"]:
        raise ValueError("Frozen Gaussian ensemble evaluator hash mismatch")
    for dependency in protocol["code_dependencies"]:
        if sha256((ROOT / dependency["path"]).resolve()) != dependency["sha256"]:
            raise ValueError(f"Frozen Gaussian ensemble dependency mismatch: {dependency['path']}")
    expected = [0.05, 0.1]
    strengths = list(map(float, protocol["search"]["gaussian_strengths"]))
    if strengths != expected:
        raise ValueError("Gaussian ensemble candidate grid differs")
    paths = {
        name: (ROOT / contract["path"]).resolve()
        for name, contract in protocol["inputs"].items()
    }
    for name, contract in protocol["inputs"].items():
        if sha256(paths[name]) != contract["sha256"]:
            raise ValueError(f"Frozen Gaussian ensemble input mismatch: {name}")
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_markdown = (ROOT / protocol["outputs"]["markdown"]).resolve()
    if output_json.exists() or output_markdown.exists():
        raise FileExistsError("Refusing to repeat Gaussian ensemble selection")

    values = restricted_selection_values(paths["inner"], paths["current_scores"])
    validate_fixed_dofa_result(paths["dofa_result"], protocol["fixed_dofa"])
    encoded, names = align_features(paths["dofa_features"], values)
    dofa = fixed_dofa_scores(protocol, values, encoded, names)
    del encoded
    gc.collect()
    raw_gaussian = load_gaussian_scene_cache(
        paths["gaussian_scene_cache"], values, protocol["gaussian_cache_protocol_sha256"]
    )
    gaussian_replicate, eligible_strengths = validate_gaussian_replicate_result(
        paths["gaussian_cache_result"],
        paths["gaussian_scene_cache"],
        protocol["gaussian_cache_protocol_sha256"],
    )
    dense = dense_evidence(paths, float(protocol["fixed_dense"]["strength"]))
    gate = float(protocol["architecture"]["final_protection_gate"])
    candidates = []
    gates = protocol["gates"]
    for strength in strengths:
        gaussian = gaussian_local_candidate(
            values["current"], raw_gaussian, strength=strength, gate=gate
        )
        scores = protected_residual_ensemble(
            values["current"], gaussian, dofa, gate=gate, anchored_multiplier=1.0
        )
        evaluation = evaluate_scores(values, scores)
        interval = ap_group_bootstrap(
            values["labels"],
            values["current"],
            scores,
            values["groups"],
            replicates=int(protocol["bootstrap"]["replicates"]),
            seed=int(protocol["bootstrap"]["seed"]),
        )
        delta = evaluation["versus_current"]["delta"]
        fold_ap = [
            evaluation["per_fold"][str(fold)]["versus_current"]["delta"][
                "average_precision"
            ]
            for fold in SELECTION_FOLDS
        ]
        sensor_ap = list(delta["sensor_average_precision"].values())
        checks = {
            "gaussian_replicate_eligible": strength in eligible_strengths,
            "minimum_ap_delta": delta["average_precision"]
            >= float(gates["average_precision_delta_minimum"]),
            "recall_no_worse": delta["recall_at_fpr_0_0713"] >= 0.0,
            "operating_counts_preserved": evaluation["operating_counts_preserved"],
            "each_fold_ap_positive": min(fold_ap) > 0.0,
            "each_sensor_ap_positive": min(sensor_ap) > 0.0,
            "paired_site_ap_lower_positive": interval["lower"] > 0.0,
            "dense_development_and_fold2_pass": bool(dense["passed"]),
        }
        passed = all(checks.values())
        candidates.append(
            {
                "gaussian_strength": strength,
                "evaluation": evaluation,
                "paired_site_ap_delta": interval,
                "checks": checks,
                "passed": passed,
                "rank": [
                    int(passed),
                    min(fold_ap),
                    interval["lower"],
                    min(sensor_ap),
                    delta["average_precision"],
                    -strength,
                ],
            }
        )
    selected = max(candidates, key=lambda row: tuple(row["rank"]))
    passed = bool(selected["passed"])
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "rejected",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": protocol["scope"],
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha256(protocol_path),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "fixed_dofa": protocol["fixed_dofa"],
        "gaussian_replicate": {
            "status": gaussian_replicate["status"],
            "validation_mode": gaussian_replicate["validation_mode"],
            "eligible_strengths": sorted(eligible_strengths),
        },
        "fixed_dense_evidence": dense,
        "candidates": [{key: value for key, value in row.items() if key != "rank"} for row in candidates],
        "selected": {key: value for key, value in selected.items() if key != "rank"},
        "all_promotion_gates_pass": passed,
        "folds_0_1_files_opened": False,
        "fold2_or_official_test_accessed": False,
        "decision": (
            "Authorize a separately frozen external/new-cohort confirmation; do not reopen fold 2 or official test."
            if passed
            else "Reject the conservative Gaussian plus DOFA ensemble before external or official-test scoring."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_json)
    write_markdown(output_markdown, report)
    print(
        json.dumps(
            {
                "ok": passed,
                "strength": selected["gaussian_strength"],
                "ap_delta": selected["evaluation"]["versus_current"]["delta"][
                    "average_precision"
                ],
                "ap_lower": selected["paired_site_ap_delta"]["lower"],
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
