#!/usr/bin/env python3
"""Group-held development audit of morphology-derived ERSRR v4 scene scores."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for directory in (MODEL_ROOT, ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from mars_s2l_adapter import iter_manifest  # noqa: E402
from mars_v4_model import MarsV4Model  # noqa: E402
from mars_v4_scoring import CANDIDATE_FORMULAS, scene_score_candidates  # noqa: E402

from acquire_mars_metadata import (  # noqa: E402
    DEFAULT_OUTPUT,
    checked_output_dir,
    repo_root,
    sha256,
)
from build_mars_v3_training_cohort import V3_SAMPLES  # noqa: E402
from train_mars_v3 import DEFAULT_METADATA_CSV, safe_output, tracked_dirty, write_json  # noqa: E402
from train_mars_v4 import (  # noqa: E402
    DEFAULT_LUT,
    MarsV4Dataset,
    metadata_and_plume_library,
    move_batch,
    v3_internal_reference,
)
from train_mars_v4_cascade import (  # noqa: E402
    balanced_group_splits,
    choose_threshold_at_fpr,
)

DEFAULT_EXPERIMENT = Path(
    "reports/experiments/mars_v4_1_seed606_epoch20_validation.json"
)
DEFAULT_CACHE = DEFAULT_OUTPUT / "publication_v4_2_internal_validation_scores.npz"
DEFAULT_JSON = Path("reports/experiments/mars_v4_2_nested_scoring.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_V4_2_NESTED_SCORING.md")
TARGET_FPRS = (0.05, 0.08, 0.095)
OUTER_FOLDS = 5
RANDOM_SEED = 20_260_713


def empirical_percentile(training: np.ndarray, held_out: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(training, dtype=np.float64))
    values = np.asarray(held_out, dtype=np.float64)
    if reference.ndim != 1 or values.ndim != 1 or not reference.size:
        raise ValueError("Percentile normalization requires nonempty 1D arrays")
    return np.searchsorted(reference, values, side="right") / reference.size


def candidate_ranking(
    labels: np.ndarray, names: Sequence[str], score_matrix: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "average_precision": float(average_precision_score(labels, score_matrix[:, index])),
            "auroc": float(roc_auc_score(labels, score_matrix[:, index])),
        }
        for index, name in enumerate(names)
    ]


def _binary_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=np.uint8)
    predicted = np.asarray(decisions, dtype=bool)
    positive = truth == 1
    negative = ~positive
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & negative))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & negative))
    return {
        "recall": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def nested_score_selection(
    labels: np.ndarray,
    groups: np.ndarray,
    names: Sequence[str],
    score_matrix: np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.uint8)
    group_values = np.asarray(groups).astype(str)
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.shape != (truth.size, len(names)) or truth.shape != group_values.shape:
        raise ValueError("Nested score-selection inputs are misaligned")
    splits = balanced_group_splits(
        truth, group_values, folds=OUTER_FOLDS, seed=RANDOM_SEED
    )
    oof_scores = np.full(truth.shape, np.nan, dtype=np.float64)
    oof_decisions = {
        target: np.zeros(truth.shape, dtype=bool) for target in TARGET_FPRS
    }
    folds = []
    for fold, (training, held_out) in enumerate(splits, start=1):
        ranking = candidate_ranking(truth[training], names, scores[training])
        selected = max(
            ranking,
            key=lambda item: (
                item["average_precision"],
                item["auroc"],
                -names.index(item["name"]),
            ),
        )
        selected_index = names.index(selected["name"])
        training_scores = scores[training, selected_index]
        held_scores = scores[held_out, selected_index]
        oof_scores[held_out] = empirical_percentile(training_scores, held_scores)
        thresholds = {}
        held_operating = {}
        for target in TARGET_FPRS:
            selection = choose_threshold_at_fpr(truth[training], training_scores, target)
            decisions = held_scores >= float(selection["threshold"])
            oof_decisions[target][held_out] = decisions
            thresholds[str(target)] = selection
            held_operating[str(target)] = _binary_metrics(truth[held_out], decisions)
        folds.append(
            {
                "outer_fold": fold,
                "training_groups": int(np.unique(group_values[training]).size),
                "held_out_groups": int(np.unique(group_values[held_out]).size),
                "training_scenes": int(training.size),
                "held_out_scenes": int(held_out.size),
                "held_out_positives": int(np.count_nonzero(truth[held_out] == 1)),
                "selected_candidate": selected,
                "thresholds_selected_on_outer_training": thresholds,
                "held_out_ranking": {
                    "average_precision": float(
                        average_precision_score(truth[held_out], held_scores)
                    ),
                    "auroc": float(roc_auc_score(truth[held_out], held_scores)),
                },
                "held_out_operating_points": held_operating,
            }
        )
    if not np.all(np.isfinite(oof_scores)):
        raise ValueError("Nested score selection did not score every held-out scene")
    return {
        "method": (
            "five outer 25 km group folds; select the fixed label-blind formula on outer-training "
            "AP/AUROC, select thresholds on outer training, and normalize held scores by the "
            "outer-training empirical score distribution"
        ),
        "ranking": {
            "average_precision": float(average_precision_score(truth, oof_scores)),
            "auroc": float(roc_auc_score(truth, oof_scores)),
        },
        "operating_points": {
            str(target): _binary_metrics(truth, decisions)
            for target, decisions in oof_decisions.items()
        },
        "folds": folds,
    }


def write_cache(
    path: Path,
    *,
    sample_ids: Sequence[str],
    groups: Sequence[str],
    labels: np.ndarray,
    names: Sequence[str],
    scores: np.ndarray,
    manifest_sha256: str,
    checkpoint_sha256: str,
    scorer_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            sample_ids=np.asarray(sample_ids),
            groups=np.asarray(groups),
            labels=np.asarray(labels, dtype=np.uint8),
            candidate_names=np.asarray(names),
            candidate_scores=np.asarray(scores, dtype=np.float64),
            manifest_sha256=np.asarray([manifest_sha256]),
            checkpoint_sha256=np.asarray([checkpoint_sha256]),
            scorer_sha256=np.asarray([scorer_sha256]),
        )
    os.replace(temporary, path)


def collect_scores(
    model: MarsV4Model,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[list[str], list[str], np.ndarray, list[str], np.ndarray]:
    model.eval()
    sample_ids: list[str] = []
    groups: list[str] = []
    labels: list[np.ndarray] = []
    rows: list[list[float]] = []
    names = list(CANDIDATE_FORMULAS)
    completed = 0
    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(moved["inputs"], moved["observable"])
            logits = output["segmentation_logits"].float().cpu().numpy()
            observable = batch["observable"].numpy()
            for local in range(logits.shape[0]):
                values = scene_score_candidates(logits[local, 0], observable[local, 0])
                rows.append([values[name] for name in names])
            sample_ids.extend(str(value) for value in batch["sample_id"])
            groups.extend(str(value) for value in batch["group_id"])
            labels.append(batch["presence"].numpy().astype(np.uint8))
            completed += logits.shape[0]
            if completed // 500 != (completed - logits.shape[0]) // 500:
                print(f"Extracted v4.2 scores for {completed} scenes", flush=True)
    return sample_ids, groups, np.concatenate(labels), names, np.asarray(rows)


def load_or_build_scores(
    cache: Path,
    *,
    manifest_sha256: str,
    checkpoint_sha256: str,
    scorer_sha256: str,
    expected_sample_ids: Sequence[str],
    model: MarsV4Model,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    overwrite: bool,
) -> tuple[list[str], list[str], np.ndarray, list[str], np.ndarray]:
    if cache.is_file() and not overwrite:
        with np.load(cache, allow_pickle=False) as source:
            identities = {
                "manifest": str(source["manifest_sha256"][0]),
                "checkpoint": str(source["checkpoint_sha256"][0]),
                "scorer": str(source["scorer_sha256"][0]),
            }
            if identities != {
                "manifest": manifest_sha256,
                "checkpoint": checkpoint_sha256,
                "scorer": scorer_sha256,
            }:
                raise ValueError("V4.2 score cache identity mismatch")
            if list(source["sample_ids"].astype(str)) != list(expected_sample_ids):
                raise ValueError("V4.2 score cache sample order mismatch")
            return (
                list(source["sample_ids"].astype(str)),
                list(source["groups"].astype(str)),
                source["labels"].copy(),
                list(source["candidate_names"].astype(str)),
                source["candidate_scores"].copy(),
            )
    sample_ids, groups, labels, names, scores = collect_scores(model, loader, device)
    if sample_ids != list(expected_sample_ids):
        raise ValueError("V4.2 inference order differs from the frozen manifest")
    write_cache(
        cache,
        sample_ids=sample_ids,
        groups=groups,
        labels=labels,
        names=names,
        scores=scores,
        manifest_sha256=manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        scorer_sha256=scorer_sha256,
    )
    return sample_ids, groups, labels, names, scores


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    nested = report["nested_group_audit"]
    final = report["final_development_rule"]
    op5 = nested["operating_points"]["0.05"]
    lines = [
        "# ERSRR v4.2 morphology scene-ranker audit",
        "",
        (
            "Development-only result on spatially isolated internal groups; "
            "the strict cohort was not loaded."
        ),
        "",
        f"- Frozen segmentation checkpoint: epoch {report['source_experiment']['best_epoch']}",
        f"- Candidate formulas: {len(report['candidate_formulas'])}",
        (
            f"- Nested AP / AUROC: {nested['ranking']['average_precision']:.3f} / "
            f"{nested['ranking']['auroc']:.3f}"
        ),
        (
            f"- Nested recall at 5% FPR target: {op5['recall']:.3f} "
            f"(observed FPR {op5['false_positive_rate']:.3f})"
        ),
        f"- Final development formula: `{final['candidate']}`",
        (
            f"- Final full-validation AP / AUROC: {final['average_precision']:.3f} / "
            f"{final['auroc']:.3f}"
        ),
        "",
        "## Decision",
        "",
        report["decision"],
        "",
        (
            "The final formula is selected for a possible frozen benchmark run only; "
            "nested held-out metrics are the development estimate."
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV.as_posix())
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT.as_posix())
    parser.add_argument("--cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v4.2 score extraction")
    metadata_dir = checked_output_dir(root, args.metadata_dir)
    metadata_csv = (root / args.metadata_csv).resolve()
    experiment_path = (root / args.experiment).resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("scope") != "v4_internal_validation_selection":
        raise ValueError("Expected a frozen v4 internal-validation experiment")
    if experiment["cohort"].get("strict_spatial_test_loaded") is not False:
        raise ValueError("Source experiment does not certify strict-cohort isolation")
    checkpoint = (root / experiment["artifact"]["path"]).resolve()
    if sha256(checkpoint) != experiment["artifact"]["sha256"]:
        raise ValueError("V4 checkpoint identity mismatch")
    model_source = MODEL_ROOT / "mars_v4_model.py"
    trainer = root / experiment["provenance"]["script"]
    if sha256(model_source) != experiment["provenance"]["model_source_sha256"]:
        raise ValueError("V4 model source changed after checkpoint selection")
    if sha256(trainer) != experiment["provenance"]["script_sha256"]:
        raise ValueError("V4 trainer changed after checkpoint selection")

    manifest = metadata_dir / V3_SAMPLES
    manifest_identity = sha256(manifest)
    if manifest_identity != experiment["source"]["manifest_sha256"]:
        raise ValueError("V4.2 manifest differs from the source experiment")
    all_records = list(iter_manifest(manifest))
    records = [
        record for record in all_records if record["research_role"] == "internal_validation"
    ]
    fit_positive_ids = {
        str(record["sample_id"])
        for record in all_records
        if record["research_role"] == "internal_training" and record["label_state"] == "PLUME"
    }
    required_ids = {str(record["sample_id"]) for record in records}
    scene_metadata, _ = metadata_and_plume_library(
        metadata_dir, metadata_csv, required_ids, fit_positive_ids
    )
    dataset = MarsV4Dataset(
        metadata_dir,
        records,
        scene_metadata,
        lut_path=(root / experiment["simulation"].get("lut_path", DEFAULT_LUT)).resolve(),
        plume_library=[],
        augment=False,
        simulation_fraction=0.0,
        seed=int(experiment["training"]["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    model = MarsV4Model().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload["model_metadata"] != experiment["model"]:
        raise ValueError("Checkpoint metadata differs from the frozen experiment")
    model.load_state_dict(payload["state_dict"], strict=True)
    scorer_path = MODEL_ROOT / "mars_v4_scoring.py"
    cache = safe_output(root, args.cache)
    expected_ids = [str(record["sample_id"]) for record in records]
    sample_ids, groups, labels, names, score_matrix = load_or_build_scores(
        cache,
        manifest_sha256=manifest_identity,
        checkpoint_sha256=experiment["artifact"]["sha256"],
        scorer_sha256=sha256(scorer_path),
        expected_sample_ids=expected_ids,
        model=model,
        loader=loader,
        device=device,
        overwrite=args.overwrite_cache,
    )
    if names != list(CANDIDATE_FORMULAS):
        raise ValueError("Cached candidate schema differs from source")
    full_ranking = candidate_ranking(labels, names, score_matrix)
    for item in full_ranking:
        index = names.index(item["name"])
        item["operating_points"] = {
            str(target): choose_threshold_at_fpr(labels, score_matrix[:, index], target)
            for target in TARGET_FPRS
        }
    selected = max(
        full_ranking,
        key=lambda item: (
            item["average_precision"], item["auroc"], -names.index(item["name"])
        ),
    )
    nested = nested_score_selection(labels, np.asarray(groups), names, score_matrix)
    reference = v3_internal_reference(root)
    op5 = nested["operating_points"]["0.05"]
    checks = {
        "nested_ap_not_below_v3_mean": (
            nested["ranking"]["average_precision"] >= reference["mean"]["average_precision"]
        ),
        "nested_auroc_not_below_v3_mean": (
            nested["ranking"]["auroc"] >= reference["mean"]["auroc"]
        ),
        "nested_recall_at_fpr5_not_below_v3_mean": (
            op5["recall"] >= reference["mean"]["recall_at_fpr5"]
        ),
        "nested_fpr_at_most_0_05": op5["false_positive_rate"] <= 0.05,
        "pixel_dice_not_below_v3_mean": (
            experiment["validation"]["positive_pixel_dice"]
            >= reference["mean"]["positive_pixel_dice"]
        ),
    }
    promoted = all(checks.values())
    decision = (
        "Retain v4.2 as a frozen development candidate. Bind the selected morphology formula and "
        "thresholds to this report before one comparison on the already-opened strict MARS cohort; "
        "treat that comparison as development evidence, not a new untouched paper test."
        if promoted
        else "Reject the v4.2 morphology ranker before strict evaluation because its group-held "
        "estimate does not clear every v3 development gate. Preserve the result and do not select "
        "a formula from strict behavior."
    )
    report = {
        "schema_version": 1,
        "scope": "v4_2_internal_group_held_scene_score_selection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "scenes": int(labels.size),
            "positives": int(np.count_nonzero(labels == 1)),
            "negatives": int(np.count_nonzero(labels == 0)),
            "groups": int(np.unique(groups).size),
            "strict_spatial_test_loaded": False,
        },
        "source_experiment": {
            "path": experiment_path.relative_to(root).as_posix(),
            "sha256": sha256(experiment_path),
            "best_epoch": int(experiment["training"]["best_epoch"]),
            "checkpoint": experiment["artifact"],
        },
        "candidate_formulas": dict(CANDIDATE_FORMULAS),
        "full_validation_candidate_audit": full_ranking,
        "nested_group_audit": nested,
        "final_development_rule": {
            "candidate": selected["name"],
            "formula": CANDIDATE_FORMULAS[selected["name"]],
            "average_precision": selected["average_precision"],
            "auroc": selected["auroc"],
            "thresholds": selected["operating_points"],
            "selected_on": "all internal-validation groups after nested held-out audit",
        },
        "v3_internal_reference": reference,
        "development_checks": checks,
        "promotion_gate_passed": promoted,
        "decision": decision,
        "ignored_score_cache": {
            "path": cache.relative_to(root).as_posix(),
            "bytes": cache.stat().st_size,
            "sha256": sha256(cache),
            "tracked": False,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "rasterio": rasterio.__version__,
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": "tools/analyze_mars_v4_scoring.py",
            "script_sha256": sha256(Path(__file__)),
            "scorer": "EarthRemoteSensingRapidResponse/mars_v4_scoring.py",
            "scorer_sha256": sha256(scorer_path),
            "manifest_sha256": manifest_identity,
        },
    }
    output_json = safe_output(root, args.output_json)
    output_markdown = safe_output(root, args.output_markdown)
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
