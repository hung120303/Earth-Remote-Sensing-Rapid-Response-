#!/usr/bin/env python3
"""Calibrate ERSRR v6 with exchangeable 25 km-group conformal risk control.

The primary loss is the false-positive fraction within a physical group.  It
is bounded in [0, 1] and monotone in the decision threshold, so the finite-
sample conformal risk correction applies when calibration and future groups
are exchangeable.  Geographic/product transport is reported separately and is
never described as guaranteed.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _aligned(**values: np.ndarray) -> dict[str, np.ndarray]:
    result = {name: np.asarray(value) for name, value in values.items()}
    sizes = {value.size for value in result.values()}
    if len(sizes) != 1:
        raise ValueError("Risk-calibration arrays are not aligned")
    if "scores" in result and not np.isfinite(result["scores"].astype(float)).all():
        raise ValueError("Risk-calibration scores contain non-finite values")
    return result


def group_losses(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return one crop-FPR loss per physical group containing negatives."""

    values = _aligned(scores=scores, labels=labels, groups=groups)
    negative = values["labels"].astype(int) == 0
    losses = []
    for group in np.unique(values["groups"][negative]):
        local = negative & (values["groups"] == group)
        losses.append(float(np.mean(values["scores"][local] >= threshold)))
    return np.asarray(losses, dtype=np.float64)


def group_balanced_recall(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    threshold: float,
) -> float:
    values = _aligned(scores=scores, labels=labels, groups=groups)
    positive = values["labels"].astype(int) == 1
    recalls = []
    for group in np.unique(values["groups"][positive]):
        local = positive & (values["groups"] == group)
        recalls.append(float(np.mean(values["scores"][local] >= threshold)))
    return float(np.mean(recalls)) if recalls else math.nan


def candidate_thresholds(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    negative_scores = np.asarray(scores, dtype=np.float64)[np.asarray(labels).astype(int) == 0]
    if not negative_scores.size:
        raise ValueError("Risk calibration requires negative examples")
    unique = np.unique(negative_scores)
    return np.concatenate(
        (
            np.asarray([-math.inf]),
            np.nextafter(unique, math.inf),
            np.asarray([math.inf]),
        )
    )


def crc_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    """Select the most permissive threshold satisfying group-level CRC.

    For n bounded group losses and B=1, CRC uses
    ``(n * empirical_risk + 1) / (n + 1) <= alpha``.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    selected: tuple[float, np.ndarray, float] | None = None
    for threshold in candidate_thresholds(scores, labels):
        losses = group_losses(scores, labels, groups, float(threshold))
        if not losses.size:
            raise ValueError("No negative physical groups are available")
        empirical = float(losses.mean())
        corrected = (losses.size * empirical + 1.0) / (losses.size + 1.0)
        if corrected <= alpha:
            selected = float(threshold), losses, corrected
            break
    if selected is None:
        losses = group_losses(scores, labels, groups, math.inf)
        return {
            "alpha": float(alpha),
            "feasible": False,
            "threshold": None,
            "negative_groups": int(losses.size),
            "minimum_achievable_crc_bound": float(1.0 / (losses.size + 1.0)),
        }
    threshold, losses, corrected = selected
    negative = np.asarray(labels).astype(int) == 0
    return {
        "alpha": float(alpha),
        "feasible": True,
        "threshold": threshold,
        "negative_groups": int(losses.size),
        "negative_crops": int(negative.sum()),
        "empirical_group_balanced_fpr": float(losses.mean()),
        "crc_expected_risk_bound": float(corrected),
        "maximum_group_fpr": float(losses.max()),
        "group_balanced_recall": group_balanced_recall(scores, labels, groups, threshold),
        "crop_recall": float(
            np.mean(np.asarray(scores)[np.asarray(labels).astype(int) == 1] >= threshold)
        )
        if np.any(np.asarray(labels).astype(int) == 1)
        else math.nan,
    }


def calibration_report(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    alphas: Iterable[float],
    *,
    strata: dict[str, np.ndarray] | None = None,
    minimum_stratum_negative_groups: int = 25,
) -> dict[str, Any]:
    values = _aligned(scores=scores, labels=labels, groups=groups)
    alpha_values = [float(value) for value in alphas]
    if not alpha_values or len(set(alpha_values)) != len(alpha_values):
        raise ValueError("alphas must be nonempty and unique")
    report: dict[str, Any] = {
        "curve": [
            crc_threshold(
                values["scores"], values["labels"], values["groups"], alpha
            )
            for alpha in alpha_values
        ],
        "strata": {},
        "guarantee_scope": (
            "expected group-balanced negative-crop FPR for future exchangeable 25 km "
            "groups; product/geographic shift results are empirical transport tests"
        ),
    }
    for family, raw in (strata or {}).items():
        stratum = np.asarray(raw).astype(str)
        if stratum.size != values["scores"].size:
            raise ValueError(f"Stratum {family!r} is not aligned")
        family_report: dict[str, Any] = {}
        for name in np.unique(stratum):
            local = stratum == name
            negative_groups = np.unique(values["groups"][local & (values["labels"].astype(int) == 0)])
            if negative_groups.size < minimum_stratum_negative_groups:
                family_report[str(name)] = {
                    "eligible": False,
                    "negative_groups": int(negative_groups.size),
                    "minimum_required": int(minimum_stratum_negative_groups),
                }
                continue
            family_report[str(name)] = {
                "eligible": True,
                "negative_groups": int(negative_groups.size),
                "curve": [
                    crc_threshold(
                        values["scores"][local],
                        values["labels"][local],
                        values["groups"][local],
                        alpha,
                    )
                    for alpha in alpha_values
                ],
            }
        report["strata"][family] = family_report
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="NPZ with scores, labels, groups")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
    )
    parser.add_argument("--minimum-stratum-negative-groups", type=int, default=25)
    args = parser.parse_args()
    source = np.load(args.input, allow_pickle=False)
    required = {"scores", "labels", "groups"}
    if not required.issubset(source.files):
        raise ValueError(f"Input is missing {sorted(required - set(source.files))}")
    strata = {
        name: source[name]
        for name in ("products", "sensors", "geographies")
        if name in source.files
    }
    report = calibration_report(
        source["scores"],
        source["labels"],
        source["groups"],
        args.alphas,
        strata=strata,
        minimum_stratum_negative_groups=args.minimum_stratum_negative_groups,
    )
    report.update(
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": str(Path(args.input)),
            "method": "conformal risk control over bounded per-25km-group FPR losses",
            "assumptions": [
                "calibration and future physical groups are exchangeable",
                "the model and score transformation are frozen before calibration",
                "the threshold is selected only from calibration negatives",
                "each physical group contributes one bounded loss regardless of crop count",
            ],
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

