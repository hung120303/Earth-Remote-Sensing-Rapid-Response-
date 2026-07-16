#!/usr/bin/env python3
"""Reproduce leakage-controlled site-relative spatial scores for all development folds."""

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

from acquire_mars_metadata import sha256  # noqa: E402
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402
from train_mars_site_relative_spatial_classifier import (  # noqa: E402
    build_site_templates,
    predict_model,
    train_model,
)
from train_mars_oof_scene_ensemble_v2 import sample_weights  # noqa: E402
from train_mars_spatial_scene_classifier import load_partitions  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_site_relative_development_scores_protocol.json")


def metric_delta(partition: dict[str, np.ndarray], scores: np.ndarray) -> dict[str, Any]:
    current = metric_summary(partition["labels"], partition["new"], partition["sensors"])
    candidate = metric_summary(partition["labels"], scores, partition["sensors"])
    return comparison(candidate, current)["delta"]


def assert_delta(
    actual: dict[str, Any], expected: dict[str, Any], tolerance: float, name: str
) -> None:
    for metric in ("average_precision", "recall_at_fpr_0_0713"):
        difference = abs(float(actual[metric]) - float(expected[metric]))
        if difference > tolerance:
            raise RuntimeError(
                f"{name} {metric} differs from the frozen report by {difference:.3g}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["builder"]["sha256"]:
        raise ValueError("Site-relative score builder hash mismatch")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen site-relative score input hash mismatch: {name}")
        paths[name] = path

    artifact = torch.load(paths["artifact"], map_location="cpu", weights_only=False)
    source_report = json.loads(paths["source_report"].read_text(encoding="utf-8"))
    spec = dict(artifact["spec"])
    blend = float(artifact["blend_weight"])
    if spec != source_report["selected"]["spec"] or blend != float(
        source_report["selected"]["blend_weight"]
    ):
        raise ValueError("Artifact and source report disagree on the selected architecture")

    images = np.load(paths["images"], mmap_mode="r", allow_pickle=False)
    with np.load(paths["metadata"], allow_pickle=False) as metadata:
        all_groups = metadata["groups"].astype(str)
    means, counts, group_indices = build_site_templates(images, all_groups)
    partitions = load_partitions(
        paths["metadata"],
        paths["scores"],
        {name: paths[name] for name in ("inner", "fold0", "fold1")},
    )
    with np.load(paths["inner"], allow_pickle=False) as cache:
        inner_folds = cache["folds"].astype(np.uint8)
        inner_ids = cache["sample_ids"].astype(str)
    with np.load(paths["fold0"], allow_pickle=False) as cache:
        fold0_ids = cache["sample_ids"].astype(str)
    with np.load(paths["fold1"], allow_pickle=False) as cache:
        fold1_ids = cache["sample_ids"].astype(str)
    if set(np.unique(inner_folds).tolist()) != {2, 3, 4}:
        raise ValueError("Inner site-relative rows do not contain folds 2/3/4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    raw_inner = np.full(partitions["inner"]["labels"].size, np.nan, dtype=np.float64)
    for holdout in (2, 3, 4):
        fit = inner_folds != holdout
        held = inner_folds == holdout
        weights = sample_weights(
            str(spec["weighting"]),
            partitions["inner"]["groups"][fit],
            partitions["inner"]["labels"][fit],
            partitions["inner"]["sensors"][fit],
        )
        fitted = train_model(
            spec,
            images,
            partitions["inner"]["image_indices"][fit],
            partitions["inner"]["labels"][fit],
            partitions["inner"]["sensors"][fit],
            weights,
            means,
            counts,
            group_indices,
            seed=int(protocol["seeds"]["inner_base"]) + holdout,
            device=device,
        )
        raw_inner[held] = predict_model(
            fitted,
            images,
            partitions["inner"]["image_indices"][held],
            partitions["inner"]["sensors"][held],
            means,
            counts,
            group_indices,
            device,
        )
        print(json.dumps({"completed_inner_holdout": holdout}), flush=True)
    if not np.isfinite(raw_inner).all():
        raise RuntimeError("Inner OOF site-relative scores are incomplete")
    inner_scores = blend_scores(partitions["inner"]["new"], raw_inner, blend)

    held_scores: dict[str, np.ndarray] = {}
    for name in ("fold0", "fold1"):
        raw = predict_model(
            artifact["fitted"],
            images,
            partitions[name]["image_indices"],
            partitions[name]["sensors"],
            means,
            counts,
            group_indices,
            device,
        )
        held_scores[name] = blend_scores(partitions[name]["new"], raw, blend)

    tolerance = float(protocol["reproduction"]["metric_tolerance"])
    deltas = {
        "inner": metric_delta(partitions["inner"], inner_scores),
        "fold0": metric_delta(partitions["fold0"], held_scores["fold0"]),
        "fold1": metric_delta(partitions["fold1"], held_scores["fold1"]),
    }
    assert_delta(
        deltas["inner"], source_report["selected"]["versus_new"]["delta"], tolerance, "inner"
    )
    for name in ("fold0", "fold1"):
        assert_delta(
            deltas[name], source_report["confirmation"][name]["versus_new"]["delta"], tolerance, name
        )

    output_path = (ROOT / protocol["outputs"]["scores"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        inner_scores=inner_scores,
        inner_sample_ids=inner_ids,
        inner_groups=partitions["inner"]["groups"],
        inner_folds=inner_folds,
        fold0_scores=held_scores["fold0"],
        fold0_sample_ids=fold0_ids,
        fold0_groups=partitions["fold0"]["groups"],
        fold1_scores=held_scores["fold1"],
        fold1_sample_ids=fold1_ids,
        fold1_groups=partitions["fold1"]["groups"],
        artifact_sha256=sha256(paths["artifact"]),
        protocol_sha256=sha256(protocol_path),
    )
    os.replace(temporary, output_path)
    report = {
        "schema_version": 1,
        "scope": "label-free score reproduction on authorized development folds only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": {name: int(partitions[name]["labels"].size) for name in ("inner", "fold0", "fold1")},
        "deltas_versus_current": deltas,
        "all_frozen_metrics_reproduced": True,
        "output": {
            "path": protocol["outputs"]["scores"],
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
            "tracked": False,
        },
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    report_path = (ROOT / protocol["outputs"]["report"]).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "output": report["output"], "deltas": deltas}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
