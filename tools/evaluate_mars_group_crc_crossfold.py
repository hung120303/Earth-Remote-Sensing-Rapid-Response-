#!/usr/bin/env python3
"""Audit geographic transfer of group-level conformal risk control on MARS.

Each direction calibrates a threshold on one untouched cross-fit development
fold and evaluates that fixed threshold on the other fold.  The conformal
claim is limited to expected group-balanced false-positive risk under
exchangeability; held-fold results are an empirical geographic-transfer test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from calibrate_mars_v6_group_risk import (
    crc_threshold,
    group_balanced_recall,
    group_losses,
)


DEFAULT_PROTOCOL = Path("configs/mars_group_crc_crossfold_protocol.json")
SENSOR_NAMES = {0: "Sentinel-2", 1: "Landsat"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aligned(**arrays: np.ndarray) -> dict[str, np.ndarray]:
    values = {name: np.asarray(value) for name, value in arrays.items()}
    sizes = {value.size for value in values.values()}
    if len(sizes) != 1:
        raise ValueError("Cross-fold risk arrays are not aligned")
    if "scores" in values and not np.isfinite(values["scores"].astype(float)).all():
        raise ValueError("Scores contain non-finite values")
    return values


def threshold_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    values = _aligned(scores=scores, labels=labels, groups=groups)
    labels_int = values["labels"].astype(int)
    negative = labels_int == 0
    positive = labels_int == 1
    predictions = values["scores"].astype(float) >= float(threshold)
    losses = group_losses(
        values["scores"], values["labels"], values["groups"], float(threshold)
    )
    if not losses.size or not negative.any() or not positive.any():
        raise ValueError("Threshold evaluation requires positive and negative groups")
    return {
        "rows": int(labels_int.size),
        "positive_rows": int(positive.sum()),
        "negative_rows": int(negative.sum()),
        "positive_groups": int(np.unique(values["groups"][positive]).size),
        "negative_groups": int(losses.size),
        "threshold": float(threshold),
        "crop_false_positive_rate": float(predictions[negative].mean()),
        "group_balanced_false_positive_rate": float(losses.mean()),
        "maximum_group_false_positive_rate": float(losses.max()),
        "crop_recall": float(predictions[positive].mean()),
        "group_balanced_recall": group_balanced_recall(
            values["scores"], values["labels"], values["groups"], float(threshold)
        ),
        "tp": int(np.sum(predictions & positive)),
        "fp": int(np.sum(predictions & negative)),
        "tn": int(np.sum(~predictions & negative)),
        "fn": int(np.sum(~predictions & positive)),
    }


def sensor_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sensors: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    values = _aligned(
        scores=scores, labels=labels, groups=groups, sensors=sensors
    )
    report: dict[str, Any] = {}
    for raw in np.unique(values["sensors"]):
        local = values["sensors"] == raw
        labels_local = values["labels"][local].astype(int)
        name = SENSOR_NAMES.get(int(raw), f"sensor-{int(raw)}")
        if not np.any(labels_local == 0) or not np.any(labels_local == 1):
            report[name] = {"eligible": False, "rows": int(local.sum())}
            continue
        report[name] = {
            "eligible": True,
            **threshold_metrics(
                values["scores"][local],
                values["labels"][local],
                values["groups"][local],
                threshold,
            ),
        }
    return report


def crossfold_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    sensors: np.ndarray,
    alphas: Iterable[float],
) -> dict[str, Any]:
    values = _aligned(
        scores=scores,
        labels=labels,
        groups=groups,
        folds=folds,
        sensors=sensors,
    )
    fold_values = sorted(int(value) for value in np.unique(values["folds"]))
    if fold_values != [3, 4]:
        raise ValueError(f"Expected folds [3, 4], found {fold_values}")
    alpha_values = [float(value) for value in alphas]
    if not alpha_values or len(set(alpha_values)) != len(alpha_values):
        raise ValueError("alphas must be nonempty and unique")

    directions: dict[str, Any] = {}
    pooled_predictions: dict[float, np.ndarray] = {
        alpha: np.zeros(values["scores"].size, dtype=bool) for alpha in alpha_values
    }
    pooled_assigned: dict[float, np.ndarray] = {
        alpha: np.zeros(values["scores"].size, dtype=bool) for alpha in alpha_values
    }

    for calibration_fold, confirmation_fold in ((3, 4), (4, 3)):
        calibration = values["folds"].astype(int) == calibration_fold
        confirmation = values["folds"].astype(int) == confirmation_fold
        direction_rows = []
        for alpha in alpha_values:
            calibrated = crc_threshold(
                values["scores"][calibration],
                values["labels"][calibration],
                values["groups"][calibration],
                alpha,
            )
            row: dict[str, Any] = {
                "alpha": alpha,
                "calibration": calibrated,
                "confirmation": None,
                "confirmation_by_sensor": {},
            }
            if calibrated["feasible"]:
                threshold = float(calibrated["threshold"])
                row["confirmation"] = threshold_metrics(
                    values["scores"][confirmation],
                    values["labels"][confirmation],
                    values["groups"][confirmation],
                    threshold,
                )
                row["confirmation_by_sensor"] = sensor_metrics(
                    values["scores"][confirmation],
                    values["labels"][confirmation],
                    values["groups"][confirmation],
                    values["sensors"][confirmation],
                    threshold,
                )
                pooled_predictions[alpha][confirmation] = (
                    values["scores"][confirmation] >= threshold
                )
                pooled_assigned[alpha][confirmation] = True
            direction_rows.append(row)
        directions[f"fold{calibration_fold}_to_fold{confirmation_fold}"] = {
            "calibration_fold": calibration_fold,
            "confirmation_fold": confirmation_fold,
            "curve": direction_rows,
        }

    pooled: dict[str, Any] = {}
    for alpha in alpha_values:
        if not pooled_assigned[alpha].all():
            pooled[str(alpha)] = {"feasible": False}
            continue
        labels_int = values["labels"].astype(int)
        negative = labels_int == 0
        positive = labels_int == 1
        group_fprs = []
        group_recalls = []
        for group in np.unique(values["groups"]):
            local = values["groups"] == group
            if np.any(local & negative):
                group_fprs.append(float(pooled_predictions[alpha][local & negative].mean()))
            if np.any(local & positive):
                group_recalls.append(float(pooled_predictions[alpha][local & positive].mean()))
        pooled[str(alpha)] = {
            "feasible": True,
            "rows": int(labels_int.size),
            "crop_false_positive_rate": float(pooled_predictions[alpha][negative].mean()),
            "group_balanced_false_positive_rate": float(np.mean(group_fprs)),
            "crop_recall": float(pooled_predictions[alpha][positive].mean()),
            "group_balanced_recall": float(np.mean(group_recalls)),
            "tp": int(np.sum(pooled_predictions[alpha] & positive)),
            "fp": int(np.sum(pooled_predictions[alpha] & negative)),
            "tn": int(np.sum(~pooled_predictions[alpha] & negative)),
            "fn": int(np.sum(~pooled_predictions[alpha] & positive)),
        }
    return {"directions": directions, "pooled_crossfit": pooled}


def _curve_row(direction: dict[str, Any], alpha: float) -> dict[str, Any]:
    matches = [row for row in direction["curve"] if math.isclose(row["alpha"], alpha)]
    if len(matches) != 1:
        raise ValueError(f"Missing unique alpha={alpha} direction row")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source_path = Path(protocol["inputs"]["champion_scores"]["path"])
    expected = protocol["inputs"]["champion_scores"]["sha256"]
    if sha256(source_path) != expected:
        raise ValueError("Champion score cache hash mismatch")
    source = np.load(source_path, allow_pickle=False)
    required = {"champion_scores", "labels", "groups", "folds", "sensors"}
    if not required.issubset(source.files):
        raise ValueError(f"Champion cache missing {sorted(required - set(source.files))}")

    curve = crossfold_curve(
        source["champion_scores"],
        source["labels"],
        source["groups"],
        source["folds"],
        source["sensors"],
        protocol["risk_control"]["alphas"],
    )
    primary_alpha = float(protocol["risk_control"]["primary_alpha"])
    primary_rows = [
        _curve_row(direction, primary_alpha)
        for direction in curve["directions"].values()
    ]
    held_fprs = [
        row["confirmation"]["group_balanced_false_positive_rate"]
        for row in primary_rows
        if row["confirmation"] is not None
    ]
    pooled_primary = curve["pooled_crossfit"][str(primary_alpha)]
    risk_transfer_supported = (
        len(held_fprs) == 2
        and all(value <= primary_alpha for value in held_fprs)
        and pooled_primary.get("feasible", False)
        and pooled_primary["crop_false_positive_rate"] <= primary_alpha
    )
    operationally_useful = (
        risk_transfer_supported
        and pooled_primary["crop_recall"]
        >= float(protocol["gates"]["minimum_pooled_crop_recall"])
    )

    report = {
        "schema_version": 1,
        "scope": "honest folds-3/4 geographic-transfer audit of group conformal risk control",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "method": (
            "calibrate bounded per-25km-group negative-crop loss on one fold and "
            "evaluate the fixed threshold on the other fold"
        ),
        "guarantee_scope": (
            "CRC bounds expected group-balanced FPR only for future exchangeable "
            "physical groups; held-fold, sensor, and geographic results are empirical"
        ),
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "input_sha256": expected,
        **curve,
        "primary_alpha": primary_alpha,
        "primary_decision": {
            "risk_transfer_supported": risk_transfer_supported,
            "operationally_useful": operationally_useful,
            "minimum_pooled_crop_recall": float(
                protocol["gates"]["minimum_pooled_crop_recall"]
            ),
            "action": (
                "retain group CRC as the deployment-calibration layer"
                if operationally_useful
                else "do not substitute group CRC for the current operating rule"
            ),
        },
        "access_audit": {
            "folds": [3, 4],
            "fold2_accessed": False,
            "folds0_1_accessed": False,
            "official_test_accessed": False,
            "external_held_outcomes_accessed": False,
        },
    }
    output = Path(protocol["outputs"]["json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = Path(protocol["outputs"]["markdown"])
    primary = report["pooled_crossfit"][str(primary_alpha)]
    markdown.write_text(
        "\n".join(
            [
                "# MARS group-CRC cross-fold transfer audit",
                "",
                f"- Primary target group FPR: **{primary_alpha:.3f}**",
                f"- Pooled held-fold crop FPR: **{primary['crop_false_positive_rate']:.6f}**",
                f"- Pooled held-fold group-balanced FPR: **{primary['group_balanced_false_positive_rate']:.6f}**",
                f"- Pooled held-fold crop recall: **{primary['crop_recall']:.6f}**",
                f"- Geographic risk transfer supported: **{str(risk_transfer_supported).lower()}**",
                f"- Operationally useful at frozen recall floor: **{str(operationally_useful).lower()}**",
                "",
                "The conformal claim is limited to future exchangeable 25 km groups; fold and sensor transfer are empirical tests.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
