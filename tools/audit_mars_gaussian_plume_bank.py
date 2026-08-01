#!/usr/bin/env python3
"""Validate a frozen Gaussian-plume bank against fit-fold real morphology."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from audit_mars_gaussian_plume_morphology import (  # noqa: E402
    mask_geometry,
    summary,
)
from mars_gaussian_plume import (  # noqa: E402
    analytical_gaussian_plume,
    sample_gaussian_parameters,
)
from mars_s2l_adapter import iter_development_manifest  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_gaussian_plume_bank_audit.json")
COMPARE_QUANTILES = ("q05", "q25", "q50", "q75", "q95")
COMPARE_METRICS = (
    "area_pixels",
    "major_4sigma_m",
    "minor_4sigma_m",
    "moment_aspect_ratio",
)


def comparison(
    real: dict[str, Any], synthetic: dict[str, Any]
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in COMPARE_METRICS:
        relative = {
            quantile: float(
                (synthetic[name][quantile] - real[name][quantile])
                / max(abs(real[name][quantile]), 1e-8)
            )
            for quantile in COMPARE_QUANTILES
        }
        metrics[name] = {
            "relative_deltas": relative,
            "maximum_absolute_relative_delta": max(map(abs, relative.values())),
        }
    alignment = {
        quantile: float(
            synthetic["major_axis_wind_alignment"][quantile]
            - real["major_axis_wind_alignment"][quantile]
        )
        for quantile in COMPARE_QUANTILES
    }
    metrics["major_axis_wind_alignment"] = {
        "absolute_deltas": alignment,
        "maximum_absolute_delta": max(map(abs, alignment.values())),
    }
    return metrics


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Gaussian-plume bank distribution audit",
        "",
        "This audit compares a deterministic analytical bank with real positive-mask geometry from MARS development folds 3 and 4. It uses no model outcome.",
        "",
        f"- Synthetic audit members: **{report['bank']['audited_members']:,}**",
        f"- Bank capacity: **{report['bank']['capacity']:,}** disjoint indexed templates",
        f"- Distribution gate: **{'PASS' if report['all_distribution_gates_pass'] else 'FAIL'}**",
        "",
        "| Metric | max discrepancy | gate |",
        "|---|---:|---:|",
    ]
    for name, result in report["comparison"].items():
        key = (
            "maximum_absolute_delta"
            if name == "major_axis_wind_alignment"
            else "maximum_absolute_relative_delta"
        )
        gate = (
            report["gates"]["alignment_maximum_absolute_delta"]
            if name == "major_axis_wind_alignment"
            else report["gates"]["geometry_maximum_absolute_relative_delta"]
        )
        lines.append(f"| {name} | {result[key]:.4f} | {gate:.4f} |")
    lines.extend(
        [
            "",
            "The bank matches geometry only. Peak delta-CH4 is sampled log-uniformly from 500 to 10,000 in the released LUT input units; no physical flux claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for dependency in protocol.get("code_dependencies", []):
        path = (ROOT / dependency["path"]).resolve()
        expected = str(dependency["sha256"])
        if expected != "TO_BE_FROZEN" and sha256(path) != expected:
            raise ValueError(f"Code dependency hash mismatch: {dependency['path']}")
    paths = {
        name: (ROOT / value["path"]).resolve()
        for name, value in protocol["inputs"].items()
    }
    for name, path in paths.items():
        expected = protocol["inputs"][name]["sha256"]
        if expected != "TO_BE_FROZEN" and sha256(path) != expected:
            raise ValueError(f"Input hash mismatch: {name}")
    real_report = json.loads(paths["real_morphology_audit"].read_text(encoding="utf-8"))
    folds = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(item["group_id"]): int(item["fold"]) for item in folds["assignments"]
    }
    winds = [
        (float(record.get("wind_u") or 0.0), float(record.get("wind_v") or 0.0))
        for record in iter_development_manifest(paths["manifest"])
        if group_to_fold[str(record["group_id"])] in set(protocol["allowed_folds"])
        and np.hypot(float(record.get("wind_u") or 0.0), float(record.get("wind_v") or 0.0)) >= 0.5
    ]
    if not winds:
        raise ValueError("No valid fit-fold wind vectors")
    rng = np.random.default_rng(int(protocol["bank"]["audit_seed"]))
    metrics: dict[str, list[float]] = {}
    members = int(protocol["bank"]["audited_members"])
    for index in range(members):
        wind = winds[int(rng.integers(len(winds)))]
        parameters = sample_gaussian_parameters((200, 200), wind, rng)
        field = analytical_gaussian_plume((200, 200), 10.0, parameters, rng)
        values = mask_geometry(field.mask, 10.0, *wind)
        for name, value in values.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                metrics.setdefault(name, []).append(float(value))
        if (index + 1) % 1000 == 0:
            print(json.dumps({"generated": index + 1, "total": members}), flush=True)
    synthetic = {name: summary(values) for name, values in sorted(metrics.items())}
    compared = comparison(real_report["geometry"], synthetic)
    geometry_gate = float(protocol["gates"]["geometry_maximum_absolute_relative_delta"])
    alignment_gate = float(protocol["gates"]["alignment_maximum_absolute_delta"])
    passed = all(
        result["maximum_absolute_relative_delta"] <= geometry_gate
        for name, result in compared.items()
        if name != "major_axis_wind_alignment"
    )
    passed &= (
        compared["major_axis_wind_alignment"]["maximum_absolute_delta"]
        <= alignment_gate
    )
    passed &= synthetic["component_count"]["minimum"] == 1
    passed &= synthetic["component_count"]["maximum"] == 1
    report = {
        "schema_version": 1,
        "scope": "fit-fold-only analytical Gaussian bank distribution audit before model scoring",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bank": protocol["bank"],
        "real_cohort": real_report["cohort"],
        "synthetic_geometry": synthetic,
        "comparison": compared,
        "gates": protocol["gates"],
        "all_distribution_gates_pass": bool(passed),
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "generator_sha256": sha256(MODEL_ROOT / "mars_gaussian_plume.py"),
            "real_morphology_audit_sha256": sha256(paths["real_morphology_audit"]),
        },
        "invariants": [
            "Only fit-fold wind metadata and the committed real morphology summary were used.",
            "No candidate model, prediction, loss, or outcome was constructed.",
            "The 20,000 indexed template capacity is partitioned before training.",
            "Train, validation, and diagnostic index ranges never overlap.",
            "Folds 0, 1, and 2 and all official-test data remain closed.",
        ],
    }
    outputs = {
        name: (ROOT / value).resolve() for name, value in protocol["outputs"].items()
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs["json"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs["markdown"].write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "all_distribution_gates_pass": passed}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
