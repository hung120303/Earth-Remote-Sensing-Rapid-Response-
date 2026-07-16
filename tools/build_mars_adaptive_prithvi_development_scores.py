#!/usr/bin/env python3
"""Reproduce the selected adaptive Prithvi development scores as an ignored cache."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_adaptive_prithvi_probe import load_features, oof_scores  # noqa: E402
from train_mars_crossfold_bagged_scene_head import load_development  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_adaptive_prithvi_development_scores_protocol.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["builder"]["sha256"]:
        raise ValueError("Adaptive Prithvi score-builder hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen adaptive Prithvi input hash mismatch: {name}")
        paths[name] = path
    values = load_development(
        {"inner": paths["inner"], "fold0": paths["fold0"], "fold1": paths["fold1"]}, paths["scores"]
    )
    source_report = json.loads(paths["source_report"].read_text(encoding="utf-8"))
    selected = source_report["selected"]
    contract = protocol["selected_architecture"]
    for key in ("feature_set", "C", "blend_weight"):
        if selected[key] != contract[key]:
            raise ValueError(f"Selected adaptive Prithvi {key} differs from protocol")
    features = load_features(paths["prithvi"], values, str(contract["feature_set"]))
    raw = oof_scores(features, values, float(contract["C"]))
    candidate = blend_scores(values["current"], raw, float(contract["blend_weight"]))
    old = metric_summary(values["labels"], values["current"], values["sensors"])
    new = metric_summary(values["labels"], candidate, values["sensors"])
    delta = comparison(new, old)["delta"]
    tolerance = float(protocol["reproduction"]["metric_tolerance"])
    expected = selected["versus_current"]["delta"]
    for metric in ("average_precision", "recall_at_fpr_0_0713"):
        if abs(float(delta[metric]) - float(expected[metric])) > tolerance:
            raise RuntimeError(f"Adaptive Prithvi {metric} reproduction failed")
    output_path = (ROOT / protocol["outputs"]["scores"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        sample_ids=values["sample_ids"], groups=values["groups"], folds=values["folds"],
        scores=candidate, raw_scores=raw, protocol_sha256=sha256(protocol_path),
        prithvi_sha256=sha256(paths["prithvi"]),
    )
    os.replace(temporary, output_path)
    report = {
        "schema_version": 1,
        "scope": "deterministic adaptive Prithvi development score reproduction; paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(candidate.size), "deltas_versus_current": delta,
        "frozen_metrics_reproduced": True,
        "output": {"path": protocol["outputs"]["scores"], "bytes": output_path.stat().st_size, "sha256": sha256(output_path), "tracked": False},
        "provenance": {"protocol_sha256": sha256(protocol_path), "script_sha256": sha256(Path(__file__).resolve()), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "numpy": np.__version__},
    }
    report_path = (ROOT / protocol["outputs"]["report"]).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "delta": delta, "output": report["output"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
