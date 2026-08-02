#!/usr/bin/env python3
"""Cross-fit a site-balanced Prithvi-100M CLS probe against the champion."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_prithvi_100m_cls_probe_protocol.json")


def domain_normalize(
    source: np.ndarray, target: np.ndarray, epsilon: float = 1e-4
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    source_mean = source.mean(axis=0, dtype=np.float64)
    target_mean = target.mean(axis=0, dtype=np.float64)
    source_scale = np.maximum(source.std(axis=0, dtype=np.float64), epsilon)
    target_scale = np.maximum(target.std(axis=0, dtype=np.float64), epsilon)
    return (
        ((source - source_mean) / source_scale).astype(np.float32),
        ((target - target_mean) / target_scale).astype(np.float32),
        {
            "source_mean": source_mean.astype(np.float32),
            "source_scale": source_scale.astype(np.float32),
            "target_mean": target_mean.astype(np.float32),
            "target_scale": target_scale.astype(np.float32),
        },
    )


def site_label_weights(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint8)
    groups = np.asarray(groups).astype(str)
    counts = Counter((int(label), group) for label, group in zip(labels, groups))
    groups_per_label = Counter(label for label, _ in counts)
    weights = np.asarray(
        [
            0.5 / groups_per_label[int(label)] / counts[(int(label), group)]
            for label, group in zip(labels, groups)
        ],
        dtype=np.float64,
    )
    return weights / weights.mean()


def load_inputs(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    with np.load(paths["features"], allow_pickle=False) as source:
        features = source["features"].astype(np.float32)
        feature_ids = source["sample_ids"].astype(str)
        names = source["feature_names"].astype(str)
    with np.load(paths["champion"], allow_pickle=False) as source:
        values = {name: np.asarray(source[name]) for name in source.files}
    lookup = {value: index for index, value in enumerate(feature_ids)}
    if len(lookup) != feature_ids.size or set(lookup) != set(values["sample_ids"].astype(str)):
        raise ValueError("Prithvi-100M features do not align with champion identities")
    order = np.asarray([lookup[value] for value in values["sample_ids"].astype(str)])
    values["features"] = features[order]
    values["feature_names"] = names
    if values["features"].shape != (17745, 3072) or not np.isfinite(values["features"]).all():
        raise ValueError("Prithvi-100M feature contract differs")
    return values


def fit_crossfold(
    values: dict[str, np.ndarray], c_value: float, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.empty(values["labels"].size, dtype=np.float64)
    models: dict[str, Any] = {}
    for held_fold in (3, 4):
        fit = values["folds"] != held_fold
        held = values["folds"] == held_fold
        source, target, moments = domain_normalize(
            values["features"][fit], values["features"][held]
        )
        model = LogisticRegression(
            C=float(c_value),
            max_iter=750,
            solver="lbfgs",
            random_state=seed + held_fold,
        )
        weights = site_label_weights(values["labels"][fit], values["groups"][fit])
        model.fit(source, values["labels"][fit], sample_weight=weights)
        raw[held] = model.predict_proba(target)[:, 1]
        models[str(held_fold)] = {"model": model, "moments": moments}
        print(json.dumps({
            "completed_holdout": held_fold,
            "C": c_value,
            "iterations": int(model.n_iter_.max()),
        }), flush=True)
    if not np.isfinite(raw).all():
        raise RuntimeError("Prithvi-100M cross-fit scores are non-finite")
    return raw, models


def evaluate(
    values: dict[str, np.ndarray], raw: np.ndarray, blend: float
) -> dict[str, Any]:
    scores = blend_scores(values["champion_scores"].astype(float), raw, float(blend))
    baseline = metric_summary(values["labels"], values["champion_scores"], values["sensors"])
    candidate = metric_summary(values["labels"], scores, values["sensors"])
    versus = comparison(candidate, baseline)
    per_fold = {}
    for fold in (3, 4):
        local = values["folds"] == fold
        per_fold[str(fold)] = comparison(
            metric_summary(values["labels"][local], scores[local], values["sensors"][local]),
            metric_summary(
                values["labels"][local], values["champion_scores"][local],
                values["sensors"][local],
            ),
        )
    return {
        "blend_weight": float(blend),
        "metrics": candidate,
        "versus_champion": versus,
        "per_fold": per_fold,
        "scores": scores,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    delta = selected["versus_champion"]["delta"]
    interval = selected["paired_site_ap_delta"]
    lines = [
        "# Prithvi-EO-2.0-100M-TL CLS probe",
        "",
        f"- Promotion gates pass: **{report['all_promotion_gates_pass']}**",
        f"- Selected C / blend: **{selected['C']} / {selected['blend_weight']}**",
        f"- AP delta versus Gaussian+DOFA champion: **{delta['average_precision']:+.6f}**",
        f"- Matched-FPR recall delta: **{delta['recall_at_fpr_0_0713']:+.6f}**",
        f"- Paired-site AP interval: **[{interval['lower']:+.6f}, {interval['upper']:+.6f}]**",
        "",
        report["decision"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen = str(protocol["status"]).startswith("frozen")
    if not args.smoke and not frozen:
        raise ValueError("Prithvi-100M outcome scoring requires a frozen protocol")
    if frozen and sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen Prithvi-100M trainer hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen Prithvi-100M input mismatch: {name}")
        paths[name] = path
    values = load_inputs(paths)
    if args.smoke:
        fit = np.r_[np.flatnonzero(values["labels"] == 0)[:128], np.flatnonzero(values["labels"] == 1)[:128]]
        source, target, _ = domain_normalize(values["features"][fit[:192]], values["features"][fit[192:]])
        model = LogisticRegression(C=0.003, max_iter=20, solver="lbfgs", random_state=0)
        model.fit(source, values["labels"][fit[:192]])
        predictions = model.predict_proba(target)[:, 1]
        report = {
            "schema_version": 1,
            "scope": "finite Prithvi-100M probe smoke; no held metric",
            "fit_rows": 192,
            "prediction_rows": int(predictions.size),
            "finite": bool(np.isfinite(predictions).all()),
            "held_outcomes_accessed": False,
        }
        output = (ROOT / protocol["outputs"]["smoke"]).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report))
        return 0

    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    if output_json.exists():
        raise FileExistsError("Refusing to repeat the Prithvi-100M probe")
    candidates = []
    raw_by_c: dict[float, np.ndarray] = {}
    models_by_c: dict[float, Any] = {}
    for c_value in map(float, protocol["search"]["regularization"]):
        raw, models = fit_crossfold(values, c_value, int(protocol["training"]["seed"]))
        raw_by_c[c_value] = raw
        models_by_c[c_value] = models
        for blend in map(float, protocol["search"]["blend_weights"]):
            row = evaluate(values, raw, blend)
            row["C"] = c_value
            delta = row["versus_champion"]["delta"]
            fold_ap = [row["per_fold"][str(fold)]["delta"]["average_precision"] for fold in (3, 4)]
            sensor_ap = list(delta["sensor_average_precision"].values())
            row["rank"] = [
                min(fold_ap), delta["average_precision"], min(sensor_ap),
                delta["recall_at_fpr_0_0713"], -blend, -abs(np.log10(c_value) - np.log10(0.003)),
            ]
            del row["scores"]
            candidates.append(row)
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    for row in candidates:
        del row["rank"]
    selected_scores = evaluate(
        values, raw_by_c[float(selected["C"])], float(selected["blend_weight"])
    )["scores"]
    interval = ap_group_bootstrap(
        values["labels"], values["champion_scores"], selected_scores,
        values["groups"], replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["seed"]),
    )
    selected["paired_site_ap_delta"] = interval
    delta = selected["versus_champion"]["delta"]
    fold_ap = [selected["per_fold"][str(fold)]["delta"]["average_precision"] for fold in (3, 4)]
    sensor_ap = list(delta["sensor_average_precision"].values())
    checks = {
        "minimum_ap_delta": delta["average_precision"] >= float(protocol["gates"]["average_precision_delta_minimum"]),
        "paired_site_lower_positive": interval["lower"] > 0.0,
        "each_fold_ap_positive": min(fold_ap) > 0.0,
        "each_sensor_ap_positive": min(sensor_ap) > 0.0,
        "matched_fpr_recall_no_worse": delta["recall_at_fpr_0_0713"] >= 0.0,
    }
    selected["checks"] = checks
    selected["passed"] = all(checks.values())
    passed = bool(selected["passed"])
    raw_path = (ROOT / protocol["outputs"]["raw_cache"]).resolve()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_path,
        sample_ids=values["sample_ids"],
        **{f"raw_C_{value:g}": raw for value, raw in raw_by_c.items()},
    )
    artifact_record = None
    if passed:
        artifact = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_suffix(artifact.suffix + ".tmp")
        joblib.dump({
            "schema_version": 1,
            "kind": "mars_prithvi_100m_cls_probe_crossfold",
            "C": selected["C"],
            "blend_weight": selected["blend_weight"],
            "models": models_by_c[float(selected["C"])],
            "protocol_sha256": sha256(protocol_path),
        }, temporary, compress=3)
        os.replace(temporary, artifact)
        artifact_record = {
            "path": protocol["outputs"]["artifact"],
            "bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
            "tracked": False,
        }
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "rejected",
        "scope": protocol["scope"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(protocol_path),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidates": candidates,
        "selected": selected,
        "all_promotion_gates_pass": passed,
        "artifact": artifact_record,
        "raw_cache": {
            "path": protocol["outputs"]["raw_cache"],
            "bytes": raw_path.stat().st_size,
            "sha256": sha256(raw_path),
            "tracked": False,
        },
        "held_external_or_official_outcomes_accessed": False,
        "decision": (
            "Authorize folds-0/1 representation confirmation before any external scoring."
            if passed
            else "Reject Prithvi-100M CLS fusion before confirmation or external scoring."
        ),
        "runtime": {"numpy": np.__version__, "sklearn": sklearn.__version__},
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(json.dumps({"passed": passed, "selected": selected}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
