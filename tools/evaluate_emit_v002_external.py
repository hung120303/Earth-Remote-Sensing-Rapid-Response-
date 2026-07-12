#!/usr/bin/env python3
"""Once-only positive confirmation of five frozen ERSRR seeds and MARS-S2L."""

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
import scipy
import sklearn
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from external_emit_adapter import ExternalEmitScene, load_external_scene  # noqa: E402
from mars_v3_model import MarsV3Model  # noqa: E402
from mars_v3_proposals import (  # noqa: E402
    extract_proposals,
    proposal_feature_names,
    proposal_features,
)

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from evaluate_released_marss2l import (  # noqa: E402
    MINIMUM_CONNECTED_PIXELS,
    MODEL_BASE,
    PIXEL_THRESHOLD,
    RELEASE_SPECS,
    component_mask,
    connected_scene_score,
    load_released_model,
)
from train_mars_v3_proposals import (  # noqa: E402
    MAXIMUM_PROPOSALS_PER_SCENE,
    calibrated_probabilities,
)

FIXED_SEEDS = (101, 202, 303, 404, 505)
DEFAULT_SEAL = Path("reports/acquisition/emit_v002_external_cohort_seal.json")
DEFAULT_WIND = Path("reports/acquisition/emit_v002_era5_wind_acquisition.json")
DEFAULT_RAW_ROOT = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
    "emit-v002-external-l1c-2026-07"
)
DEFAULT_JSON = Path("reports/experiments/emit_v002_external_confirmation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/EMIT_V002_EXTERNAL_CONFIRMATION.md")
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20_260_713


def safe_path(root: Path, value: str | Path) -> Path:
    result = (root / value).resolve()
    if result != root and root not in result.parents:
        raise ValueError("Path must resolve beneath the repository root")
    return result


def tracked_dirty(root: Path) -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
    )
    return bool(output.strip())


def connected_mask(
    probability: np.ndarray,
    observable: np.ndarray,
    threshold: float,
    minimum_pixels: int,
) -> np.ndarray:
    candidate = (np.asarray(probability) >= threshold) & np.asarray(observable, dtype=bool)
    labels, count = ndimage.label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros(candidate.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_pixels
    keep[0] = False
    return keep[labels]


def positive_metrics(predictions: np.ndarray, count: int) -> dict[str, Any]:
    values = np.asarray(predictions, dtype=bool)
    if values.ndim != 1 or values.size != count or count <= 0:
        raise ValueError("Positive metrics require one decision per positive scene")
    tp = int(np.count_nonzero(values))
    return {"scenes": count, "true_positive": tp, "false_negative": count - tp, "recall": tp / count}


def pixel_metrics(
    predictions: list[np.ndarray], truths: list[np.ndarray], observables: list[np.ndarray]
) -> dict[str, Any]:
    intersection = predicted = truth = 0
    for prediction, target, observable in zip(predictions, truths, observables):
        valid_prediction = np.asarray(prediction, dtype=bool) & observable
        valid_truth = np.asarray(target, dtype=bool) & observable
        intersection += int(np.count_nonzero(valid_prediction & valid_truth))
        predicted += int(np.count_nonzero(valid_prediction))
        truth += int(np.count_nonzero(valid_truth))
    union = predicted + truth - intersection
    return {
        "intersection_over_union": 0.0 if union == 0 else intersection / union,
        "dice": 0.0 if predicted + truth == 0 else 2.0 * intersection / (predicted + truth),
        "intersection_pixels": intersection,
        "predicted_positive_pixels": predicted,
        "truth_positive_pixels": truth,
    }


def verify_file(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file() or path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{label} is missing or size-mismatched: {path}")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")


def frozen_seed_artifacts(root: Path) -> list[dict[str, Any]]:
    values = []
    for seed in FIXED_SEEDS:
        validation_path = safe_path(root, f"reports/experiments/mars_v3_seed{seed}_validation.json")
        proposal_path = safe_path(root, f"reports/experiments/mars_v3_seed{seed}_proposal_validation.json")
        if not validation_path.is_file() or not proposal_path.is_file():
            raise FileNotFoundError(f"Five-seed campaign is incomplete at seed {seed}")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        if validation.get("scope") != "v3_internal_validation_selection":
            raise ValueError(f"Invalid validation scope for seed {seed}")
        if proposal.get("scope") != "v3_connected_proposal_internal_validation":
            raise ValueError(f"Invalid proposal scope for seed {seed}")
        if int(validation["training"]["seed"]) != seed or int(proposal["training"]["seed"]) != seed:
            raise ValueError(f"Seed identity mismatch for {seed}")
        checkpoint = safe_path(root, validation["artifact"]["path"])
        classifier = safe_path(root, proposal["artifact"]["path"])
        verify_file(checkpoint, validation["artifact"], f"seed {seed} checkpoint")
        verify_file(classifier, proposal["artifact"], f"seed {seed} proposal artifact")
        if proposal["source"]["v3_checkpoint_sha256"] != validation["artifact"]["sha256"]:
            raise ValueError(f"Proposal/checkpoint identity mismatch for seed {seed}")
        if proposal["source"]["v3_validation_experiment_sha256"] != sha256(validation_path):
            raise ValueError(f"Proposal/validation-report identity mismatch for seed {seed}")
        values.append(
            {
                "seed": seed,
                "validation_path": validation_path,
                "proposal_path": proposal_path,
                "validation": validation,
                "proposal": proposal,
                "checkpoint": checkpoint,
                "classifier": classifier,
            }
        )
    return values


def load_scenes(
    root: Path,
    seal: dict[str, Any],
    wind: dict[str, Any],
    raw_root: Path,
) -> list[ExternalEmitScene]:
    if not wind["summary"]["complete"]:
        raise ValueError("ERA5-Land acquisition is incomplete")
    wind_by_group = {item["group_id"]: item for item in wind["records"]}
    scenes = []
    for item in seal["records"]:
        if not item["final_gate_pass"]:
            continue
        group = item["group_id"]
        if group not in wind_by_group:
            raise ValueError(f"Missing frozen ERA5-Land wind for {group}")
        scene_dir = raw_root / item["granule_id"]
        scenes.append(
            load_external_scene(
                root,
                scene_dir / "manifest.json",
                scene_dir / "cloudsen12.manifest.json",
                wind_by_group[group],
            )
        )
    if len(scenes) != int(seal["summary"]["final_gate_pass"]):
        raise ValueError("Loaded external scene count differs from the seal")
    if len({item.group_id for item in scenes}) != len(scenes):
        raise ValueError("External confirmation contains duplicate groups")
    return scenes


def batched_model_outputs(
    model: MarsV3Model,
    scenes: list[ExternalEmitScene],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, np.ndarray | float]]:
    values: list[dict[str, np.ndarray | float]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(scenes), batch_size):
            batch = scenes[start : start + batch_size]
            inputs = torch.from_numpy(np.stack([item.inputs for item in batch])).to(device)
            observable = torch.from_numpy(
                np.stack([item.observable for item in batch])[:, None].astype(np.float32)
            ).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(inputs, observable, return_dense_features=True)
            segmentation = torch.sigmoid(output["segmentation_logits"]).float().cpu().numpy()[:, 0]
            presence = torch.sigmoid(output["presence_logit"]).float().cpu().numpy()
            components = output["component_features"].float().cpu().numpy()
            for index in range(len(batch)):
                values.append(
                    {
                        "segmentation": segmentation[index],
                        "neural_presence": float(presence[index]),
                        "component_features": components[index],
                    }
                )
    return values


def evaluate_seed(
    artifact: dict[str, Any],
    scenes: list[ExternalEmitScene],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model = MarsV3Model().to(device)
    payload = torch.load(artifact["checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    outputs = batched_model_outputs(model, scenes, device, batch_size)
    proposal_artifact = joblib.load(artifact["classifier"])
    expected_names = proposal_feature_names(16)
    if proposal_artifact["feature_names"] != expected_names:
        raise ValueError(f"Proposal feature contract changed for seed {artifact['seed']}")
    neural_weight = float(proposal_artifact["neural_presence_weight"])
    upper = float(proposal_artifact["upper_plume_threshold"])
    segmentation_rule = artifact["validation"]["operating_rule"]["segmentation"]
    scene_scores = []
    decisions = []
    masks = []
    records = []
    for scene, output in zip(scenes, outputs):
        probability = np.asarray(output["segmentation"], dtype=np.float32)
        proposals = extract_proposals(probability, scene.observable)[:MAXIMUM_PROPOSALS_PER_SCENE]
        if proposals and neural_weight < 1.0:
            features = np.stack(
                [
                    proposal_features(
                        proposal,
                        probability,
                        scene.inputs,
                        scene.observable,
                        np.asarray(output["component_features"]),
                    )
                    for proposal in proposals
                ]
            )
            calibrated = calibrated_probabilities(
                proposal_artifact["classifier"], proposal_artifact["calibrator"], features
            )
            proposal_score = float(np.max(calibrated))
        else:
            proposal_score = 0.0
        score = neural_weight * float(output["neural_presence"]) + (1.0 - neural_weight) * proposal_score
        decision = score >= upper
        mask = connected_mask(
            probability,
            scene.observable,
            float(segmentation_rule["pixel_threshold"]),
            int(segmentation_rule["minimum_connected_pixels"]),
        )
        scene_scores.append(score)
        decisions.append(decision)
        masks.append(mask)
        records.append(
            {
                "group_id": scene.group_id,
                "score": score,
                "prediction": bool(decision),
                "neural_score": float(output["neural_presence"]),
                "proposal_score": proposal_score,
            }
        )
    return {
        "seed": artifact["seed"],
        "upper_plume_threshold": upper,
        "neural_presence_weight": neural_weight,
        "scene": positive_metrics(np.asarray(decisions), len(scenes)),
        "segmentation": pixel_metrics(
            masks,
            [item.plume_mask for item in scenes],
            [item.observable for item in scenes],
        ),
        "records": records,
        "scores": np.asarray(scene_scores, dtype=np.float64),
        "predictions": np.asarray(decisions, dtype=bool),
    }


def evaluate_released(
    root: Path,
    scenes: list[ExternalEmitScene],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    spec = RELEASE_SPECS["mars-s2l"]
    checkpoint = safe_path(root, spec["directory"] / "best_epoch")
    if checkpoint.stat().st_size != spec["checkpoint_bytes"] or sha256(checkpoint) != spec["checkpoint_sha256"]:
        raise ValueError("Released MARS-S2L checkpoint identity mismatch")
    model = load_released_model(checkpoint, device, spec["input_channels"])
    scores = []
    decisions = []
    masks = []
    records = []
    with torch.inference_mode():
        for start in range(0, len(scenes), batch_size):
            batch = scenes[start : start + batch_size]
            inputs = torch.from_numpy(np.stack([item.inputs for item in batch])).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                probability = torch.sigmoid(model(inputs)).float().cpu().numpy()
            for scene, value in zip(batch, probability):
                value = value.astype(np.float32)
                value[~scene.observable] = 0.0
                score = connected_scene_score(value)
                mask = component_mask(value) & scene.observable
                decision = bool(np.any(mask))
                scores.append(score)
                decisions.append(decision)
                masks.append(mask)
                records.append({"group_id": scene.group_id, "score": score, "prediction": decision})
    return {
        "model": spec["display_name"],
        "author_rule": {
            "pixel_threshold": PIXEL_THRESHOLD,
            "minimum_connected_pixels": MINIMUM_CONNECTED_PIXELS,
        },
        "scene": positive_metrics(np.asarray(decisions), len(scenes)),
        "segmentation": pixel_metrics(
            masks,
            [item.plume_mask for item in scenes],
            [item.observable for item in scenes],
        ),
        "records": records,
        "scores": np.asarray(scores, dtype=np.float64),
        "predictions": np.asarray(decisions, dtype=bool),
        "checkpoint_sha256": spec["checkpoint_sha256"],
    }


def paired_bootstrap(
    candidates: list[np.ndarray], baseline: np.ndarray, replicates: int
) -> dict[str, Any]:
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    count = baseline.size
    if any(item.size != count for item in candidates):
        raise ValueError("External prediction vectors are not aligned")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = []
    candidate_recalls = []
    baseline_recalls = []
    for _ in range(replicates):
        rows = rng.integers(0, count, size=count)
        seeds = rng.integers(0, len(candidates), size=len(candidates))
        candidate = float(np.mean([np.mean(candidates[index][rows]) for index in seeds]))
        released = float(np.mean(baseline[rows]))
        candidate_recalls.append(candidate)
        baseline_recalls.append(released)
        deltas.append(candidate - released)

    def describe(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values)
        return {"mean": float(np.mean(array)), "95ci": np.quantile(array, [0.025, 0.975]).tolist()}

    return {
        "method": "paired bootstrap over 55 independent groups with five-seed resampling",
        "replicates": replicates,
        "random_seed": BOOTSTRAP_SEED,
        "ersrr_recall": describe(candidate_recalls),
        "released_mars_s2l_recall": describe(baseline_recalls),
        "recall_delta": describe(deltas),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    candidate = report["five_seed_summary"]
    baseline = report["released_mars_s2l"]
    interval = report["paired_bootstrap"]["recall_delta"]["95ci"]
    lines = [
        "# EMIT V002 external positive confirmation",
        "",
        "This prediction-blind sealed cohort contains positives only; it cannot estimate false-positive rate or average precision.",
        "",
        f"- Cohort: {report['cohort']['scenes']} scenes / {report['cohort']['groups']} independent groups",
        f"- ERSRR five-seed mean recall: {candidate['mean_recall']:.3f}",
        f"- Released MARS-S2L recall: {baseline['scene']['recall']:.3f}",
        f"- Recall delta: {candidate['mean_recall'] - baseline['scene']['recall']:+.3f}",
        f"- Paired recall-delta 95% CI: {interval[0]:+.3f} to {interval[1]:+.3f}",
        f"- ERSRR five-seed mean EMIT-mask IoU: {candidate['mean_intersection_over_union']:.3f}",
        f"- Released MARS-S2L EMIT-mask IoU: {baseline['segmentation']['intersection_over_union']:.3f}",
        "",
        "EMIT/Sentinel-2 offsets of up to six hours make this a confirmation and stress test, not exact simultaneous Sentinel-2 ground truth. The sealed MARS strict campaign remains the primary no-plume and same-distribution benchmark.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", default=DEFAULT_SEAL.as_posix())
    parser.add_argument("--wind", default=DEFAULT_WIND.as_posix())
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT.as_posix())
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    # Fail before opening any external raster unless every fixed model artifact exists.
    seed_artifacts = frozen_seed_artifacts(root)
    seal_path = safe_path(root, args.seal)
    wind_path = safe_path(root, args.wind)
    if not wind_path.is_file():
        raise FileNotFoundError("Official ERA5-Land acquisition is required before external inference")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    wind = json.loads(wind_path.read_text(encoding="utf-8"))
    scenes = load_scenes(root, seal, wind, safe_path(root, args.raw_root))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_results = [
        evaluate_seed(item, scenes, device, args.batch_size) for item in seed_artifacts
    ]
    released = evaluate_released(root, scenes, device, args.batch_size)
    bootstrap = paired_bootstrap(
        [item["predictions"] for item in seed_results],
        released["predictions"],
        args.replicates,
    )
    compact_seed_results = []
    for item in seed_results:
        compact_seed_results.append(
            {key: value for key, value in item.items() if key not in {"scores", "predictions"}}
        )
    compact_released = {
        key: value for key, value in released.items() if key not in {"scores", "predictions"}
    }
    report = {
        "schema_version": 1,
        "scope": "once_only_emit_v002_external_positive_confirmation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "scenes": len(scenes),
            "groups": len({item.group_id for item in scenes}),
            "label_scope": "positive confirmation only",
            "seal_sha256": sha256(seal_path),
            "wind_acquisition_sha256": sha256(wind_path),
        },
        "five_seed_summary": {
            "seeds": list(FIXED_SEEDS),
            "mean_recall": float(np.mean([item["scene"]["recall"] for item in seed_results])),
            "standard_deviation_recall": float(
                np.std([item["scene"]["recall"] for item in seed_results], ddof=1)
            ),
            "mean_intersection_over_union": float(
                np.mean([item["segmentation"]["intersection_over_union"] for item in seed_results])
            ),
            "mean_dice": float(np.mean([item["segmentation"]["dice"] for item in seed_results])),
        },
        "seed_results": compact_seed_results,
        "released_mars_s2l": compact_released,
        "paired_bootstrap": bootstrap,
        "limitations": [
            "positive-only cohort cannot estimate false-positive rate, specificity, precision, AP, or AUROC",
            "EMIT and Sentinel-2 acquisition times differ by up to six hours",
            "EMIT polygon overlap is a descriptive segmentation proxy rather than simultaneous Sentinel-2 truth",
        ],
        "runtime": {
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "provenance": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
            "script": "tools/evaluate_emit_v002_external.py",
            "script_sha256": sha256(Path(__file__)),
        },
    }
    json_path = safe_path(root, args.output_json)
    markdown_path = safe_path(root, args.output_markdown)
    write_json(json_path, report)
    write_markdown(markdown_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
