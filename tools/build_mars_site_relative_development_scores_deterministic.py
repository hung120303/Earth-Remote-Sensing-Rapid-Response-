#!/usr/bin/env python3
"""Build a deterministic, leakage-controlled site-relative development score cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_mars_site_relative_development_scores as base  # noqa: E402
from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_site_relative_development_scores_deterministic_protocol.json")


def evaluate(
    labels: np.ndarray, current: np.ndarray, candidate: np.ndarray, sensors: np.ndarray
) -> dict[str, Any]:
    old = metric_summary(labels, current, sensors)
    new = metric_summary(labels, candidate, sensors)
    return comparison(new, old)["delta"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["deterministic_wrapper"]["sha256"]:
        raise ValueError("Deterministic wrapper hash mismatch")
    if sha256(Path(base.__file__).resolve()) != protocol["builder"]["sha256"]:
        raise ValueError("Frozen underlying score-builder hash mismatch")

    original_assert = base.assert_delta

    def v2_assert(
        actual: dict[str, Any], expected: dict[str, Any], tolerance: float, name: str
    ) -> None:
        if name != "inner":
            original_assert(actual, expected, tolerance, name)
            return
        gates = protocol["deterministic_rebuild_gates"]
        if float(actual["average_precision"]) <= 0.0:
            raise RuntimeError("Deterministic inner spatial score does not improve AP")
        if float(actual["recall_at_fpr_0_0713"]) < -float(gates["combined_recall_tolerance"]):
            raise RuntimeError("Deterministic inner spatial score violates recall tolerance")
        if abs(float(actual["average_precision"]) - float(expected["average_precision"])) > float(
            gates["source_ap_drift_tolerance"]
        ):
            raise RuntimeError("Deterministic inner AP drift exceeds the preregistered tolerance")

    base.assert_delta = v2_assert
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    old_argv = sys.argv
    try:
        sys.argv = [str(Path(base.__file__).resolve()), "--protocol", str(protocol_path)]
        result = base.main()
    finally:
        sys.argv = old_argv
        base.assert_delta = original_assert
    if result != 0:
        return result

    output_path = (ROOT / protocol["outputs"]["scores"]).resolve()
    with np.load(output_path, allow_pickle=False) as scores:
        candidate = {
            "inner": scores["inner_scores"].astype(np.float64),
            "fold0": scores["fold0_scores"].astype(np.float64),
            "fold1": scores["fold1_scores"].astype(np.float64),
        }
        identities = {
            name: scores[f"{name}_sample_ids"].astype(str)
            for name in ("inner", "fold0", "fold1")
        }
    inputs = protocol["inputs"]
    values = load_development(
        {
            name: (ROOT / inputs[name]["path"]).resolve()
            for name in ("inner", "fold0", "fold1")
        },
        (ROOT / inputs["scores"]["path"]).resolve(),
    )
    per_fold: dict[str, Any] = {}
    for fold in (2, 3, 4):
        rows = values["folds"] == fold
        local_ids = values["sample_ids"][rows]
        inner_lookup = {sample_id: index for index, sample_id in enumerate(identities["inner"])}
        local_scores = candidate["inner"][[inner_lookup[sample_id] for sample_id in local_ids]]
        per_fold[str(fold)] = evaluate(
            values["labels"][rows], values["current"][rows], local_scores, values["sensors"][rows]
        )
    gates = protocol["deterministic_rebuild_gates"]
    checks = {
        "each_inner_fold_ap_higher": min(
            value["average_precision"] for value in per_fold.values()
        ) > 0.0,
        "each_inner_fold_recall_within_tolerance": min(
            value["recall_at_fpr_0_0713"] for value in per_fold.values()
        ) >= -float(gates["per_fold_recall_tolerance"]),
    }
    if not all(checks.values()):
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Deterministic site-relative per-fold gates failed: {checks}")

    report_path = (ROOT / protocol["outputs"]["report"]).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["all_frozen_metrics_reproduced"] = False
    report["all_v2_acceptance_checks_pass"] = True
    report["reproduction_interpretation"] = (
        "New deterministic crossfit accepted under preregistered direction/drift gates; "
        "fold-0/1 frozen-artifact metrics reproduced, but old stochastic inner scores are not claimed bitwise."
    )
    report["inner_per_fold_deltas_versus_current"] = per_fold
    report["deterministic_checks"] = checks
    report["provenance"]["deterministic_wrapper_sha256"] = sha256(Path(__file__).resolve())
    report["provenance"]["cuda_deterministic_algorithms"] = True
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "checks": checks, "per_fold": per_fold}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
