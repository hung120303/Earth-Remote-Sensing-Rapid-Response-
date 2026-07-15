#!/usr/bin/env python3
"""Cross-fit a scene complement over frozen Prithvi-EO-2.0 MARS features."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_encoder_scene_probe import (  # noqa: E402
    confirm_partition,
    load_aligned_partitions,
    metric_comparison,
    predict_model,
    train_model,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap, sample_weights  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402

DEFAULT_ENCODER_CACHE = Path("outputs/mars_prithvi_eo_2_tiny_tl_features_all_folds.npz")
DEFAULT_SCORE_CACHE = Path("outputs/mars_scene_domain_routing_development_scores.npz")
DEFAULT_INNER_CACHE = Path("outputs/mars_scene_features_folds234.npz")
DEFAULT_FOLD0_CACHE = Path("outputs/mars_scene_features_fold0.npz")
DEFAULT_FOLD1_CACHE = Path("outputs/mars_scene_features_fold1_crossfit.npz")
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_prithvi_scene_probe.pt")
DEFAULT_JSON = Path("reports/experiments/mars_prithvi_scene_probe.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_PRITHVI_SCENE_PROBE.md")
EXPECTED = {
    "score": "fd955b78b26a3b2a5165b4abab02180ccf4dad433511bf4da7afbff44275c1c7",
    "inner": "01d8587e283c1179d61a7c789eb514b3f699d3e7a75bf8c50e4baff3f1698b89",
    "fold0": "372e152734db1314417ed385b099af54acd182bf758b1d2eabcedfeb64a709e7",
    "fold1": "2b62e03215047d6a49639fdaead7e9d3cf7939b8eda26fb9442210b49c3ba108",
}
INNER_FOLDS = (2, 3, 4)
CLS_WIDTH = 4 * 192
TEMPORAL_CHANGE_OFFSET = CLS_WIDTH + 2 * 3 * 192
BLENDS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def candidate_specs() -> list[dict[str, Any]]:
    specs = []
    for feature_set in (
        "cls_plus_base",
        "temporal_change_plus_base",
        "all_prithvi_plus_base",
    ):
        for weighting in ("uniform", "group", "site_cell"):
            specs.append(
                {
                    "feature_set": feature_set,
                    "hidden": 0,
                    "dropout": 0.0,
                    "weighting": weighting,
                    "epochs": 20,
                    "learning_rate": 0.001,
                    "weight_decay": 0.001,
                }
            )
        specs.append(
            {
                "feature_set": feature_set,
                "hidden": 0,
                "dropout": 0.0,
                "weighting": "uniform",
                "epochs": 20,
                "learning_rate": 0.001,
                "weight_decay": 0.01,
            }
        )
    for feature_set, hidden in (
        ("temporal_change_plus_base", 64),
        ("all_prithvi_plus_base", 128),
    ):
        specs.append(
            {
                "feature_set": feature_set,
                "hidden": hidden,
                "dropout": 0.25,
                "weighting": "uniform",
                "epochs": 20,
                "learning_rate": 0.001,
                "weight_decay": 0.001,
            }
        )
    return specs


def spec_key(spec: dict[str, Any]) -> str:
    return "_".join(f"{key}-{spec[key]}" for key in sorted(spec))


def select_features(
    encoder: np.ndarray,
    base: np.ndarray,
    encoder_names: np.ndarray,
    base_names: np.ndarray,
    feature_set: str,
) -> tuple[np.ndarray, list[str]]:
    if feature_set == "cls_plus_base":
        selected = encoder[:, :CLS_WIDTH]
        names = encoder_names[:CLS_WIDTH].astype(str).tolist()
        if not names[0].startswith("prithvi_block3_cls_") or not names[-1].startswith("prithvi_block12_cls_"):
            raise ValueError("Prithvi CLS feature slice differs from its frozen schema")
    elif feature_set == "temporal_change_plus_base":
        selected = encoder[:, TEMPORAL_CHANGE_OFFSET:]
        names = encoder_names[TEMPORAL_CHANGE_OFFSET:].astype(str).tolist()
        if (
            not names[0].startswith("prithvi_target_minus_reference_mean_")
            or not names[-1].startswith("prithvi_absolute_difference_max_")
        ):
            raise ValueError("Prithvi temporal-change slice differs from its frozen schema")
    elif feature_set == "all_prithvi_plus_base":
        selected = encoder
        names = encoder_names.astype(str).tolist()
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return (
        np.concatenate([selected.astype(np.float32), base.astype(np.float32)], axis=1),
        [*names, *base_names.astype(str).tolist()],
    )


def crossfit_raw(
    spec: dict[str, Any],
    partition: dict[str, np.ndarray],
    encoder_names: np.ndarray,
    base_names: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    features, names = select_features(
        partition["encoder"], partition["base"], encoder_names, base_names, str(spec["feature_set"])
    )
    scores = np.empty(partition["labels"].shape, dtype=np.float64)
    for fold in INNER_FOLDS:
        fit = partition["folds"] != fold
        held = partition["folds"] == fold
        weights = sample_weights(
            str(spec["weighting"]),
            partition["groups"][fit],
            partition["labels"][fit],
            partition["sensors"][fit],
        )
        fitted = train_model(
            spec,
            features[fit],
            partition["labels"][fit],
            weights,
            seed=20261000 + fold,
            device=device,
        )
        scores[held] = predict_model(fitted, features[held], device)
    return scores, names


def screen(
    spec: dict[str, Any], partition: dict[str, np.ndarray], raw: np.ndarray, blend: float
) -> dict[str, Any]:
    scores = blend_scores(partition["new"], raw, blend)
    versus_primary = metric_comparison(partition, scores, "primary")
    versus_new = metric_comparison(partition, scores, "new")
    per_fold = {}
    for fold in INNER_FOLDS:
        rows = partition["folds"] == fold
        local = {
            key: value[rows]
            for key, value in partition.items()
            if key not in {"folds", "encoder", "base"}
        }
        per_fold[str(fold)] = {
            "versus_primary": metric_comparison(local, scores[rows], "primary"),
            "versus_new": metric_comparison(local, scores[rows], "new"),
        }
    primary_fold_ap = [value["versus_primary"]["delta"]["average_precision"] for value in per_fold.values()]
    new_fold_ap = [value["versus_new"]["delta"]["average_precision"] for value in per_fold.values()]
    stable = (
        versus_primary["delta"]["average_precision"] > 0.0
        and versus_primary["delta"]["recall_at_fpr_0_0713"] > 0.0
        and min(primary_fold_ap) > 0.0
        and min(versus_primary["delta"]["sensor_average_precision"].values()) >= -0.005
        and versus_new["delta"]["average_precision"] > 0.0
        and versus_new["delta"]["recall_at_fpr_0_0713"] >= -0.0025
        and min(new_fold_ap) >= -0.005
    )
    return {
        "spec": spec,
        "spec_key": spec_key(spec),
        "blend_weight": blend,
        "stable": stable,
        "versus_primary": versus_primary,
        "versus_new": versus_new,
        "per_fold": per_fold,
        "rank": [
            int(stable),
            min(new_fold_ap),
            versus_new["delta"]["average_precision"],
            versus_primary["delta"]["average_precision"],
        ],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    lines = [
        "# Prithvi-EO-2.0 MARS scene probe",
        "",
        "Frozen foundation features were selected with cross-fitted folds 2/3/4; the chosen "
        "probe and blend were then evaluated once on held folds 0 and 1.",
        "",
        f"- Selected: `{selected['spec_key']}`, blend {selected['blend_weight']:.2f}",
        f"- Inner AP delta vs current stronger head: {selected['versus_new']['delta']['average_precision']:+.5f}",
        "",
        "| Partition | AP delta vs primary | Recall delta | AP 95% CI | AP delta vs current head | Gates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, value in report["confirmation"].items():
        delta = value["versus_primary"]["delta"]
        interval = value["paired_group_bootstrap_ap_delta_vs_primary"]
        lines.append(
            f"| {name} | {delta['average_precision']:+.5f} | "
            f"{delta['recall_at_fpr_0_0713']:+.5f} | [{interval['lower']:+.5f}, {interval['upper']:+.5f}] | "
            f"{value['versus_new']['delta']['average_precision']:+.5f} | "
            f"{'PASS' if value['passed'] else 'FAIL'} |"
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-cache", default=DEFAULT_ENCODER_CACHE.as_posix())
    parser.add_argument("--encoder-sha256", required=True)
    parser.add_argument("--score-cache", default=DEFAULT_SCORE_CACHE.as_posix())
    parser.add_argument("--inner-cache", default=DEFAULT_INNER_CACHE.as_posix())
    parser.add_argument("--fold0-cache", default=DEFAULT_FOLD0_CACHE.as_posix())
    parser.add_argument("--fold1-cache", default=DEFAULT_FOLD1_CACHE.as_posix())
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    paths = {
        "encoder": (root / args.encoder_cache).resolve(),
        "score": (root / args.score_cache).resolve(),
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
    }
    expected = {"encoder": args.encoder_sha256, **EXPECTED}
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"Frozen {name} cache hash mismatch")
    partitions, encoder_names, base_names = load_aligned_partitions(
        paths["encoder"],
        paths["score"],
        {name: paths[name] for name in ("inner", "fold0", "fold1")},
    )
    if len(encoder_names) != 3072:
        raise ValueError("Expected the frozen 3,072-feature Prithvi schema")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = []
    candidate_summaries = []
    raw_by_spec = {}
    names_by_spec = {}
    for index, spec in enumerate(candidate_specs()):
        raw, names = crossfit_raw(spec, partitions["inner"], encoder_names, base_names, device)
        key = spec_key(spec)
        raw_by_spec[key] = raw
        names_by_spec[key] = names
        local = [screen(spec, partitions["inner"], raw, blend) for blend in BLENDS]
        candidates.extend(local)
        best = max(local, key=lambda value: tuple(value["rank"]))
        candidate_summaries.append(
            {
                "spec": spec,
                "spec_key": key,
                "best_blend_weight": best["blend_weight"],
                "stable": best["stable"],
                "rank": best["rank"],
                "best_delta_vs_primary": best["versus_primary"]["delta"],
                "best_delta_vs_new": best["versus_new"]["delta"],
                "per_fold_average_precision_delta": {
                    fold: {
                        "versus_primary": value["versus_primary"]["delta"][
                            "average_precision"
                        ],
                        "versus_new": value["versus_new"]["delta"]["average_precision"],
                    }
                    for fold, value in best["per_fold"].items()
                },
            }
        )
        print(
            json.dumps(
                {
                    "spec": index + 1,
                    "total": len(candidate_specs()),
                    "key": key,
                    "best_blend": best["blend_weight"],
                    "ap_delta_vs_new": best["versus_new"]["delta"]["average_precision"],
                    "stable": best["stable"],
                }
            ),
            flush=True,
        )
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    selected_raw = raw_by_spec[selected["spec_key"]]
    selected_scores = blend_scores(partitions["inner"]["new"], selected_raw, selected["blend_weight"])
    selected["paired_group_bootstrap_ap_delta_vs_primary"] = ap_group_bootstrap(
        partitions["inner"]["labels"],
        partitions["inner"]["primary"],
        selected_scores,
        partitions["inner"]["groups"],
        replicates=10000,
        seed=20261040,
    )
    selected["paired_group_bootstrap_ap_delta_vs_new"] = ap_group_bootstrap(
        partitions["inner"]["labels"],
        partitions["inner"]["new"],
        selected_scores,
        partitions["inner"]["groups"],
        replicates=10000,
        seed=20261041,
    )
    selected["inner_passed"] = bool(
        selected["stable"]
        and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"] > 0.0
        and selected["paired_group_bootstrap_ap_delta_vs_new"]["lower"] > -0.0025
    )

    spec = selected["spec"]
    inner_features, feature_names = select_features(
        partitions["inner"]["encoder"],
        partitions["inner"]["base"],
        encoder_names,
        base_names,
        str(spec["feature_set"]),
    )
    fitted = train_model(
        spec,
        inner_features,
        partitions["inner"]["labels"],
        sample_weights(
            str(spec["weighting"]),
            partitions["inner"]["groups"],
            partitions["inner"]["labels"],
            partitions["inner"]["sensors"],
        ),
        seed=20261050,
        device=device,
    )
    confirmation = {}
    thresholds = []
    for index, name in enumerate(("fold0", "fold1")):
        partition = partitions[name]
        features, local_names = select_features(
            partition["encoder"], partition["base"], encoder_names, base_names, str(spec["feature_set"])
        )
        if local_names != feature_names:
            raise ValueError("Held Prithvi probe feature schema mismatch")
        raw = predict_model(fitted, features, device)
        scores = blend_scores(partition["new"], raw, selected["blend_weight"])
        confirmation[name] = confirm_partition(partition, scores, seed=20261060 + index)
        thresholds.append(
            confirmation[name]["versus_primary"]["metrics"]["operating_point"]["threshold"]
        )

    artifact_path = (root / args.artifact).resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "kind": "mars_prithvi_eo_2_tiny_tl_scene_probe",
            "spec": spec,
            "blend_weight": selected["blend_weight"],
            "feature_names": feature_names,
            "fitted": fitted,
            "operational_scene_threshold": max(thresholds),
            "encoder_cache_sha256": args.encoder_sha256,
            "score_cache_sha256": EXPECTED["score"],
        },
        temporary,
    )
    os.replace(temporary, artifact_path)
    passed = selected["inner_passed"] and all(value["passed"] for value in confirmation.values())
    report = {
        "schema_version": 1,
        "scope": "development-only Prithvi transfer probe; paper test not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_model_count": len(candidate_specs()),
        "candidate_blend_count": len(BLENDS),
        "selection_candidate_count": len(candidates),
        "candidate_summaries": candidate_summaries,
        "selected": selected,
        "confirmation": confirmation,
        "operational_scene_threshold": max(thresholds),
        "all_promotion_gates_pass": passed,
        "decision": (
            "Freeze the Prithvi scene complement for one transparent paper benchmark."
            if passed
            else "Reject the Prithvi scene complement before paper-test feature extraction."
        ),
        "provenance": {
            **{f"{name}_cache_sha256": value for name, value in expected.items()},
            "artifact_sha256": sha256(artifact_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    write_json((root / args.output_json).resolve(), report)
    write_markdown((root / args.output_markdown).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "selected": selected["spec_key"],
                "blend": selected["blend_weight"],
                "inner_ap_delta_vs_new": selected["versus_new"]["delta"]["average_precision"],
                "confirmation": {
                    name: {
                        "passed": value["passed"],
                        "ap_delta_vs_new": value["versus_new"]["delta"]["average_precision"],
                    }
                    for name, value in confirmation.items()
                },
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
