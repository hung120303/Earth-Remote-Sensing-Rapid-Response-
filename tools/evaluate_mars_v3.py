#!/usr/bin/env python3
"""Evaluate one frozen MARS v3 checkpoint without threshold or architecture tuning."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rasterio
import sklearn
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import iter_manifest  # noqa: E402
from mars_v3_model import MarsV3Model  # noqa: E402
from mars_v3_proposals import (  # noqa: E402
    extract_proposals,
    proposal_feature_names,
    proposal_features,
)

from acquire_mars_metadata import DEFAULT_OUTPUT, REVISION, checked_output_dir, repo_root, sha256  # noqa: E402
from build_mars_v3_strict_cohort import (  # noqa: E402
    DEFAULT_JSON as STRICT_COHORT_JSON,
    V3_STRICT_SAMPLES,
)
from build_mars_v3_training_cohort import V3_SAMPLES  # noqa: E402
from run_mars_dev_pixel_baselines import evaluate_rule  # noqa: E402
from run_mars_dev_scene_baselines import bootstrap_ci, metrics, role_weights  # noqa: E402
from train_mars_joint_model import selective_quality_metrics  # noqa: E402
from train_mars_v3 import (  # noqa: E402
    DEFAULT_METADATA_CSV,
    MarsV3Dataset,
    wind_lookup,
)
from train_mars_v3_proposals import (  # noqa: E402
    MAXIMUM_PROPOSALS_PER_SCENE,
    cache_identity,
    calibrated_probabilities,
)

DEFAULT_EXPERIMENT = Path("reports/experiments/mars_v3_validation.json")
DEFAULT_PROPOSAL_EXPERIMENT = Path("reports/experiments/mars_v3_proposal_validation.json")
DEFAULT_JSON = Path("reports/experiments/mars_v3_strict_evaluation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_V3_STRICT_EVALUATION.md")


def safe_output(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if root not in result.parents:
        raise ValueError("Output must resolve beneath the repository root")
    return result


def tracked_dirty(root: Path) -> bool:
    status = subprocess.check_output(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        cwd=root,
        text=True,
    )
    return bool(status.strip())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_scene_prediction_cache(
    root: Path,
    path: Path,
    *,
    sample_ids: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
    primary_scores: np.ndarray,
    neural_scores: np.ndarray,
    primary_threshold: float,
    seed: int,
    strict_manifest_sha256: str,
    validation_experiment_sha256: str,
    proposal_experiment_sha256: str | None,
) -> dict[str, Any]:
    """Atomically persist compact scene evidence for paired campaign inference."""
    resolved = path.resolve()
    if root not in resolved.parents:
        raise ValueError("Prediction cache must resolve beneath the repository root")
    original_primary_scores = np.asarray(primary_scores)
    fixed_predictions = (original_primary_scores >= float(primary_threshold)).astype(np.uint8)
    arrays = (
        np.asarray(sample_ids).astype(str),
        np.asarray(groups).astype(str),
        np.asarray(labels, dtype=np.uint8),
        np.asarray(original_primary_scores, dtype=np.float32),
        np.asarray(neural_scores, dtype=np.float32),
        fixed_predictions,
    )
    if any(array.ndim != 1 for array in arrays) or len({array.shape for array in arrays}) != 1:
        raise ValueError("Prediction cache arrays must be matching one-dimensional vectors")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray([1], dtype=np.uint16),
            sample_ids=arrays[0],
            groups=arrays[1],
            labels=arrays[2],
            primary_scores=arrays[3],
            primary_predictions=arrays[5],
            neural_scores=arrays[4],
            primary_threshold=np.asarray([primary_threshold], dtype=np.float64),
            seed=np.asarray([seed], dtype=np.int64),
            strict_manifest_sha256=np.asarray([strict_manifest_sha256]),
            validation_experiment_sha256=np.asarray([validation_experiment_sha256]),
            proposal_experiment_sha256=np.asarray(
                [proposal_experiment_sha256 or ""]
            ),
        )
    temporary.replace(resolved)
    return {
        "path": resolved.relative_to(root).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
        "tracked": False,
        "contents": "compact per-scene labels, groups, scores, and fixed-threshold predictions; no raster pixels",
    }


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def calibration_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    bin_count: int = 10,
) -> dict[str, Any]:
    """Return fixed-bin reliability evidence and expected calibration error."""
    y = np.asarray(labels, dtype=np.float64)
    probability = np.asarray(scores, dtype=np.float64)
    if y.shape != probability.shape or y.ndim != 1:
        raise ValueError("Calibration labels and scores must be matching 1D arrays")
    if bin_count <= 1 or not np.all((probability >= 0.0) & (probability <= 1.0)):
        raise ValueError("Calibration requires probabilities in [0,1] and at least two bins")
    sample_weight = (
        np.ones(y.shape, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if sample_weight.shape != y.shape or np.any(sample_weight < 0):
        raise ValueError("Calibration weights must be non-negative and match labels")
    total_weight = float(np.sum(sample_weight))
    if total_weight <= 0:
        raise ValueError("Calibration weights must have positive mass")
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    assignments = np.digitize(probability, edges[1:-1], right=False)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bin_count):
        selected = assignments == index
        if not np.any(selected):
            continue
        mass = float(np.sum(sample_weight[selected]))
        confidence = float(
            np.average(probability[selected], weights=sample_weight[selected])
        )
        prevalence = float(np.average(y[selected], weights=sample_weight[selected]))
        gap = abs(confidence - prevalence)
        ece += (mass / total_weight) * gap
        bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "samples": int(np.sum(selected)),
                "weight": mass,
                "mean_probability": confidence,
                "observed_prevalence": prevalence,
                "absolute_gap": gap,
            }
        )
    return {
        "method": f"{bin_count} fixed-width probability bins",
        "expected_calibration_error": float(ece),
        "bins": bins,
    }


def load_proposal_stage(
    root: Path,
    metadata_dir: Path,
    experiment_path: Path,
    experiment: dict[str, Any],
    checkpoint: Path,
    proposal_experiment_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(proposal_experiment_path.read_text(encoding="utf-8"))
    if report.get("scope") != "v3_connected_proposal_internal_validation":
        raise ValueError("Proposal experiment is not a frozen internal-validation artifact")
    source = report["source"]
    if source["v3_validation_experiment_sha256"] != sha256(experiment_path):
        raise ValueError("Proposal stage was selected from a different v3 validation report")
    if source["v3_checkpoint_sha256"] != experiment["artifact"]["sha256"]:
        raise ValueError("Proposal stage was fit from a different v3 checkpoint")
    if int(report["training"]["seed"]) != int(experiment["training"]["seed"]):
        raise ValueError("Proposal stage seed differs from the v3 training seed")
    training_manifest = metadata_dir / V3_SAMPLES
    if source["manifest_sha256"] != sha256(training_manifest):
        raise ValueError("Proposal-stage training manifest identity mismatch")
    proposal_source = MODEL_ROOT / "mars_v3_proposals.py"
    if report["provenance"]["proposal_source_sha256"] != sha256(proposal_source):
        raise ValueError("Proposal descriptor source changed after validation selection")
    proposal_trainer = root / "tools" / "train_mars_v3_proposals.py"
    if (
        report["provenance"].get("script") != "tools/train_mars_v3_proposals.py"
        or report["provenance"].get("script_sha256") != sha256(proposal_trainer)
    ):
        raise ValueError("Proposal trainer source changed after validation selection")
    artifact_path = (root / report["artifact"]["path"]).resolve()
    if root not in artifact_path.parents:
        raise ValueError("Proposal artifact must resolve beneath the repository root")
    if sha256(artifact_path) != report["artifact"]["sha256"]:
        raise ValueError("Proposal artifact identity does not match its validation report")
    artifact = joblib.load(artifact_path)
    expected_identity = cache_identity(
        training_manifest, checkpoint, experiment_path, decoder_channels=16
    )
    if artifact.get("cache_identity") != expected_identity:
        raise ValueError("Proposal artifact identity differs from the frozen feature contract")
    if int(artifact.get("seed", -1)) != int(experiment["training"]["seed"]):
        raise ValueError("Proposal artifact seed differs from the v3 training seed")
    if float(artifact.get("neural_presence_weight", -1.0)) != float(
        report["operating_rule"].get("neural_presence_weight", -2.0)
    ):
        raise ValueError("Proposal blend weight differs between artifact and report")
    expected_names = proposal_feature_names(16)
    if artifact.get("feature_names") != expected_names:
        raise ValueError("Proposal artifact feature ordering differs from the frozen contract")
    operating = report["operating_rule"]
    if float(artifact["upper_plume_threshold"]) != float(
        operating["upper_plume_threshold"]
    ):
        raise ValueError("Proposal upper threshold differs between artifact and report")
    artifact_lower = artifact["lower_no_plume_threshold"]
    report_lower = operating["lower_no_plume_threshold"]
    if artifact_lower is None or report_lower is None:
        if artifact_lower is not report_lower:
            raise ValueError("Proposal lower threshold differs between artifact and report")
    elif float(artifact_lower) != float(report_lower):
        raise ValueError("Proposal lower threshold differs between artifact and report")
    return report, artifact


@torch.no_grad()
def collect_predictions(
    model: MarsV3Model,
    loader: DataLoader,
    device: torch.device,
    proposal_artifact: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Collect neural outputs and deployable proposal scores in one frozen pass."""
    model.eval()
    values: dict[str, list[Any]] = {
        "presence": [],
        "quality": [],
        "labels": [],
        "quality_labels": [],
        "segmentation": [],
        "observable": [],
        "truth": [],
        "groups": [],
        "sample_ids": [],
        "proposal_scores": [],
        "proposal_counts": [],
    }
    for batch in loader:
        cpu_inputs = batch["inputs"].numpy()
        cpu_observable = batch["observable"].numpy()[:, 0].astype(bool)
        gpu_inputs = batch["inputs"].to(device, non_blocking=True)
        gpu_observable = batch["observable"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(
                gpu_inputs,
                gpu_observable,
                return_dense_features=proposal_artifact is not None,
            )
        probabilities = (
            torch.sigmoid(output["segmentation_logits"])
            .float()
            .cpu()
            .numpy()[:, 0]
        )
        values["presence"].append(
            torch.sigmoid(output["presence_logit"]).float().cpu().numpy()
        )
        values["quality"].append(
            torch.sigmoid(output["quality_logit"]).float().cpu().numpy()
        )
        values["labels"].append(batch["presence"].numpy())
        values["quality_labels"].append(batch["quality"].numpy())
        values["segmentation"].append(probabilities.astype(np.float16))
        values["observable"].extend(
            np.packbits(item.ravel()) for item in cpu_observable
        )
        values["truth"].extend(
            np.packbits(item[0].numpy().astype(bool).ravel()) for item in batch["mask"]
        )
        values["groups"].extend(batch["group_id"])
        values["sample_ids"].extend(batch["sample_id"])
        if proposal_artifact is not None:
            dense = output["component_features"].float().cpu().numpy()
            for index in range(probabilities.shape[0]):
                proposals = extract_proposals(
                    probabilities[index], cpu_observable[index]
                )[:MAXIMUM_PROPOSALS_PER_SCENE]
                values["proposal_counts"].append(len(proposals))
                if not proposals:
                    values["proposal_scores"].append(0.0)
                    continue
                descriptors = np.stack(
                    [
                        proposal_features(
                            proposal,
                            probabilities[index],
                            cpu_inputs[index],
                            cpu_observable[index],
                            dense[index],
                        )
                        for proposal in proposals
                    ]
                )
                scores = calibrated_probabilities(
                    proposal_artifact["classifier"],
                    proposal_artifact["calibrator"],
                    descriptors,
                )
                values["proposal_scores"].append(float(np.max(scores)))
    result = {
        "presence": np.concatenate(values["presence"]),
        "quality": np.concatenate(values["quality"]),
        "labels": np.concatenate(values["labels"]).astype(np.uint8),
        "quality_labels": np.concatenate(values["quality_labels"]).astype(np.uint8),
        "segmentation": np.concatenate(values["segmentation"]),
        "observable": np.stack(values["observable"]),
        "truth": np.stack(values["truth"]),
        "groups": np.asarray(values["groups"]),
        "sample_ids": np.asarray(values["sample_ids"]),
    }
    if proposal_artifact is not None:
        result["proposal_scores"] = np.asarray(values["proposal_scores"], dtype=np.float64)
        result["proposal_counts"] = np.asarray(values["proposal_counts"], dtype=np.uint16)
    return result


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    scene = report["strict_spatial_test"]["scene_unweighted"]
    pixel = report["strict_spatial_test"]["segmentation"]["pixel"]
    interval = report["strict_spatial_test"]["group_bootstrap"]
    lines = [
        "# Frozen ERSRR MARS v3 strict-spatial evaluation",
        "",
        "Checkpoint and all operating rules were selected on internal validation before this run.",
        "",
        f"- Primary scene score: {report['primary_scene_score']}",
        f"- Cohort: {report['cohort']['samples']} scenes / {report['cohort']['groups']} frozen 25 km groups",
        f"- Scene recall / specificity / FPR: {fmt(scene['recall'])} / {fmt(scene['specificity'])} / {fmt(scene['false_positive_rate'])}",
        f"- Scene AUROC / AP: {scene['auroc']:.3f} / {scene['average_precision']:.3f}",
        f"- Recall 95% CI: {interval['recall_95ci'][0]:.3f}-{interval['recall_95ci'][1]:.3f}",
        f"- Specificity 95% CI: {interval['specificity_95ci'][0]:.3f}-{interval['specificity_95ci'][1]:.3f}",
        f"- Pixel AP / IoU / Dice: {pixel['average_precision']:.4f} / {pixel['intersection_over_union']:.4f} / {pixel['dice']:.4f}",
        f"- Promotion gate: {'PASS' if report['promotion_gate_passed'] else 'FAIL'}",
        "",
        "## Decision",
        "",
        report["decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV.as_posix())
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT.as_posix())
    parser.add_argument(
        "--proposal-experiment", default=DEFAULT_PROPOSAL_EXPERIMENT.as_posix()
    )
    parser.add_argument(
        "--neural-only",
        action="store_true",
        help="Evaluate the neural scene head without the frozen proposal stage",
    )
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument(
        "--prediction-cache",
        help="Ignored compact NPZ used only for paired multi-seed strict aggregation",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = repo_root()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v3 evaluation")
    metadata_dir = checked_output_dir(root, args.metadata_dir)
    metadata_csv = (root / args.metadata_csv).resolve()
    experiment_path = (root / args.experiment).resolve()
    proposal_experiment_path = (root / args.proposal_experiment).resolve()
    output_json = safe_output(root, args.output_json)
    output_markdown = safe_output(root, args.output_markdown)
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("smoke_test"):
        raise ValueError("A pipeline smoke checkpoint cannot consume the strict benchmark")
    if experiment["scope"] != "v3_internal_validation_selection":
        raise ValueError("Experiment is not a frozen v3 validation-selection artifact")
    checkpoint = (root / experiment["artifact"]["path"]).resolve()
    if sha256(checkpoint) != experiment["artifact"]["sha256"]:
        raise ValueError("V3 checkpoint identity does not match the validation report")
    if sha256(MODEL_ROOT / "mars_v3_model.py") != experiment["provenance"]["model_source_sha256"]:
        raise ValueError("V3 model source changed after checkpoint selection")
    if sha256(MODEL_ROOT / "mars_s2l_adapter.py") != experiment["provenance"].get(
        "adapter_source_sha256"
    ):
        raise ValueError("MARS adapter source changed after checkpoint selection")
    trainer = root / "tools" / "train_mars_v3.py"
    if (
        experiment["provenance"].get("script") != "tools/train_mars_v3.py"
        or experiment["provenance"].get("script_sha256") != sha256(trainer)
    ):
        raise ValueError("V3 trainer source changed after checkpoint selection")
    proposal_report: dict[str, Any] | None = None
    proposal_artifact: dict[str, Any] | None = None
    if not args.neural_only:
        proposal_report, proposal_artifact = load_proposal_stage(
            root,
            metadata_dir,
            experiment_path,
            experiment,
            checkpoint,
            proposal_experiment_path,
        )

    manifest = metadata_dir / V3_STRICT_SAMPLES
    strict_cohort = json.loads((root / STRICT_COHORT_JSON).read_text(encoding="utf-8"))
    if sha256(manifest) != strict_cohort["identities"]["sample_manifest_sha256"]:
        raise ValueError("Full strict-spatial manifest identity mismatch")
    all_records = list(iter_manifest(manifest))
    records = [
        record for record in all_records if record["research_role"] == "strict_spatial_test"
    ]
    if len(records) != int(strict_cohort["samples"]["total"]):
        raise ValueError("Full strict-spatial manifest row count mismatch")
    required_ids = {str(record["sample_id"]) for record in records}
    winds = wind_lookup(metadata_csv, required_ids)
    dataset = MarsV3Dataset(metadata_dir, records, winds, augment=False, seed=0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    model = MarsV3Model().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload["model_metadata"] != experiment["model"]:
        raise ValueError("Checkpoint model metadata differs from the validation report")
    model.load_state_dict(payload["state_dict"], strict=True)
    predictions = collect_predictions(model, loader, device, proposal_artifact)

    rule = experiment["operating_rule"]
    upper = float(rule["upper_plume_threshold"])
    lower = rule["lower_no_plume_threshold"]
    lower = None if lower is None else float(lower)
    quality_threshold = float(rule["quality_threshold"])
    segmentation_rule = dict(rule["segmentation"])
    labels = predictions["labels"]
    neural_scores = predictions["presence"]
    scores = neural_scores
    if proposal_report is not None:
        neural_weight = float(
            proposal_report["operating_rule"]["neural_presence_weight"]
        )
        scores = (
            neural_weight * neural_scores
            + (1.0 - neural_weight) * predictions["proposal_scores"]
        )
    primary_rule = proposal_report["operating_rule"] if proposal_report else rule
    upper = float(primary_rule["upper_plume_threshold"])
    primary_lower = primary_rule["lower_no_plume_threshold"]
    primary_lower = None if primary_lower is None else float(primary_lower)
    groups = predictions["groups"].astype(str)
    weights = role_weights(labels, "strict_spatial_test")
    scene_unweighted = metrics(labels, scores, upper)
    scene_weighted = metrics(labels, scores, upper, weights=weights)
    neural_scene_unweighted = metrics(
        labels, neural_scores, float(rule["upper_plume_threshold"])
    )
    for scene_metrics in (scene_unweighted, scene_weighted, neural_scene_unweighted):
        fpr = scene_metrics["false_positive_rate"]
        scene_metrics["false_positives_per_100_scenes"] = (
            None if fpr is None else 100.0 * float(fpr)
        )
    segmentation = evaluate_rule(
        predictions["segmentation"],
        predictions["observable"],
        predictions["truth"],
        labels,
        groups,
        segmentation_rule,
    )
    interval = bootstrap_ci(labels, scores, groups, upper, int(experiment["training"]["seed"]))
    selective = selective_quality_metrics(
        labels,
        scores,
        predictions["quality"],
        primary_lower,
        upper,
        quality_threshold,
        weights,
    )
    strict_manifest_identity = sha256(manifest)
    validation_experiment_identity = sha256(experiment_path)
    proposal_experiment_identity = (
        sha256(proposal_experiment_path) if proposal_report is not None else None
    )
    prediction_cache = (
        safe_output(root, args.prediction_cache)
        if args.prediction_cache
        else metadata_dir
        / f"publication_v3_strict_scene_predictions_seed{int(experiment['training']['seed'])}.npz"
    )
    prediction_cache_artifact = write_scene_prediction_cache(
        root,
        prediction_cache,
        sample_ids=predictions["sample_ids"],
        groups=groups,
        labels=labels,
        primary_scores=scores,
        neural_scores=neural_scores,
        primary_threshold=upper,
        seed=int(experiment["training"]["seed"]),
        strict_manifest_sha256=strict_manifest_identity,
        validation_experiment_sha256=validation_experiment_identity,
        proposal_experiment_sha256=proposal_experiment_identity,
    )
    gate = (
        float(interval["recall_95ci"][0]) >= 0.75
        and float(scene_unweighted["false_positive_rate"] or 1.0) <= 0.05
        and float(scene_unweighted["specificity"] or 0.0) >= 0.95
    )
    report = {
        "schema_version": 2,
        "scope": "frozen_v3_full_strict_spatial_evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "UNEP-IMEO/MARS-S2L",
            "revision": REVISION,
            "strict_manifest_sha256": strict_manifest_identity,
            "validation_experiment_sha256": validation_experiment_identity,
            "proposal_experiment_sha256": proposal_experiment_identity,
        },
        "cohort": {
            "samples": len(records),
            "positives": int(np.sum(labels)),
            "negatives": int(np.sum(labels == 0)),
            "groups": int(np.unique(groups).size),
        },
        "model": experiment["model"],
        "artifact": experiment["artifact"],
        "primary_scene_score": (
            "validation-selected neural/proposal blended probability"
            if proposal_report is not None
            else "neural presence head probability"
        ),
        "operating_rule": {
            "primary_scene": primary_rule,
            "neural_scene": rule,
        },
        "proposal_artifact": (
            proposal_report["artifact"] if proposal_report is not None else None
        ),
        "scene_prediction_cache": prediction_cache_artifact,
        "strict_spatial_test": {
            "scene_unweighted": scene_unweighted,
            "scene_representative_weighted": scene_weighted,
            "neural_scene_unweighted": neural_scene_unweighted,
            "proposal_stage": (
                {
                    "proposal_count": int(np.sum(predictions["proposal_counts"])),
                    "scenes_without_proposals": int(
                        np.sum(predictions["proposal_counts"] == 0)
                    ),
                }
                if proposal_report is not None
                else None
            ),
            "calibration": {
                "unweighted": calibration_summary(labels, scores),
                "representative_weighted": calibration_summary(
                    labels, scores, weights=weights
                ),
            },
            "segmentation": segmentation,
            "group_bootstrap": interval,
            "selective_with_quality": selective,
            "quality_auroc": float(
                roc_auc_score(predictions["quality_labels"], predictions["quality"])
            ),
        },
        "promotion_gate_passed": gate,
        "decision": (
            "V3 clears the frozen full-MARS gate. All five fixed seeds and untouched EMIT confirmation remain required before promotion."
            if gate
            else "V3 does not clear the frozen full-MARS gate. Preserve this result; do not retune from strict-test behavior."
        ),
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
            "script": "tools/evaluate_mars_v3.py",
            "script_sha256": sha256(Path(__file__)),
        },
    }
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
