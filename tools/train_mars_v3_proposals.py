#!/usr/bin/env python3
"""Fit the v3 connected-component false-alarm classifier on frozen groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rasterio
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import iter_manifest  # noqa: E402
from mars_v3_model import MarsV3Model  # noqa: E402
from mars_v3_proposals import (  # noqa: E402
    PROPOSAL_THRESHOLDS,
    extract_proposals,
    label_proposal,
    proposal_feature_names,
    proposal_features,
)

from acquire_mars_metadata import DEFAULT_OUTPUT, REVISION, checked_output_dir, repo_root, sha256  # noqa: E402
from build_mars_v3_training_cohort import V3_SAMPLES  # noqa: E402
from run_mars_dev_scene_baselines import (  # noqa: E402
    choose_lower_threshold,
    choose_upper_threshold,
    role_weights,
)
from train_mars_v3 import DEFAULT_METADATA_CSV, MarsV3Dataset, wind_lookup  # noqa: E402

DEFAULT_EXPERIMENT = Path("reports/experiments/mars_v3_validation.json")
DEFAULT_CACHE = "publication_v3_proposal_features_seed303.npz"
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_v3_proposals_seed303.joblib")
DEFAULT_JSON = Path("reports/experiments/mars_v3_proposal_validation.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_V3_PROPOSAL_VALIDATION.md")
MAXIMUM_PROPOSALS_PER_SCENE = 20
CALIBRATION_GROUP_FRACTION = 0.20
DEFAULT_SEED = 303
NEURAL_PRESENCE_BLEND_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)


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


def cache_identity(
    manifest: Path, checkpoint: Path, experiment: Path, decoder_channels: int
) -> dict[str, Any]:
    return {
        "schema": "mars_v3_connected_proposals_v4",
        "manifest_sha256": sha256(manifest),
        "checkpoint_sha256": sha256(checkpoint),
        "validation_experiment_sha256": sha256(experiment),
        "model_source_sha256": sha256(MODEL_ROOT / "mars_v3_model.py"),
        "adapter_source_sha256": sha256(MODEL_ROOT / "mars_s2l_adapter.py"),
        "proposal_source_sha256": sha256(MODEL_ROOT / "mars_v3_proposals.py"),
        "proposal_thresholds": list(PROPOSAL_THRESHOLDS),
        "maximum_proposals_per_scene": MAXIMUM_PROPOSALS_PER_SCENE,
        "neural_presence_blend_weights": list(NEURAL_PRESENCE_BLEND_WEIGHTS),
        "decoder_channels": decoder_channels,
    }


def write_cache(path: Path, data: dict[str, np.ndarray], identity: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **data,
        identity_json=np.asarray([json.dumps(identity, sort_keys=True)]),
    )
    os.replace(temporary, path)


def load_cache(path: Path, expected: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        observed = json.loads(str(payload["identity_json"][0]))
        if observed != expected:
            raise ValueError("Proposal cache identity differs from the frozen model/manifest")
        return {
            key: payload[key].copy() for key in payload.files if key != "identity_json"
        }


def extract_cache(
    metadata_dir: Path,
    records: list[dict[str, Any]],
    winds: dict[str, tuple[float, float]],
    model: MarsV3Model,
    device: torch.device,
    *,
    batch_size: int,
    workers: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    dataset = MarsV3Dataset(metadata_dir, records, winds, augment=False, seed=0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    records_by_id = {str(record["sample_id"]): record for record in records}
    features: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    proposal_groups: list[str] = []
    proposal_roles: list[str] = []
    scene_ids: list[str] = []
    scene_groups: list[str] = []
    scene_roles: list[str] = []
    scene_labels: list[int] = []
    scene_proposal_counts: list[int] = []
    scene_neural_presence: list[float] = []
    ignored_ambiguous = 0
    model.eval()
    completed = 0
    with torch.inference_mode():
        for batch in loader:
            cpu_inputs = batch["inputs"].numpy()
            cpu_observable = batch["observable"].numpy()[:, 0].astype(bool)
            cpu_truth = batch["mask"].numpy()[:, 0].astype(bool)
            gpu_inputs = batch["inputs"].to(device, non_blocking=True)
            gpu_observable = batch["observable"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(
                    gpu_inputs, gpu_observable, return_dense_features=True
                )
            probabilities = (
                torch.sigmoid(output["segmentation_logits"])
                .float()
                .cpu()
                .numpy()[:, 0]
            )
            component_features = (
                output["component_features"].float().cpu().numpy()
            )
            neural_presence = (
                torch.sigmoid(output["presence_logit"]).float().cpu().numpy()
            )
            for index, sample_id in enumerate(batch["sample_id"]):
                sample_id = str(sample_id)
                record = records_by_id[sample_id]
                role = str(record["research_role"])
                group = str(record["group_id"])
                truth = cpu_truth[index] & cpu_observable[index]
                proposals = extract_proposals(
                    probabilities[index], cpu_observable[index]
                )[:MAXIMUM_PROPOSALS_PER_SCENE]
                scene_ids.append(sample_id)
                scene_groups.append(group)
                scene_roles.append(role)
                scene_labels.append(1 if record["label_state"] == "PLUME" else 0)
                scene_neural_presence.append(float(neural_presence[index]))
                retained_for_scene = 0
                for proposal in proposals:
                    target = label_proposal(proposal, truth)
                    features.append(
                        proposal_features(
                            proposal,
                            probabilities[index],
                            cpu_inputs[index],
                            cpu_observable[index],
                            component_features[index],
                        )
                    )
                    # Ambiguous overlaps are not classifier targets, but they
                    # must remain in scene scoring. Dropping them here would
                    # use validation truth to remove false alarms at inference.
                    label = target["label"]
                    labels.append(-1 if label is None else int(label))
                    if label is None:
                        ignored_ambiguous += 1
                    sample_ids.append(sample_id)
                    proposal_groups.append(group)
                    proposal_roles.append(role)
                    retained_for_scene += 1
                scene_proposal_counts.append(retained_for_scene)
            completed += len(batch["sample_id"])
            if completed % 1000 < batch_size or completed == len(records):
                print(f"Proposal extraction: {completed:,}/{len(records):,}", flush=True)
    if not features:
        raise RuntimeError("No labeled v3 proposals were extracted")
    return {
        "x": np.stack(features).astype(np.float16),
        "y": np.asarray(labels, dtype=np.int8),
        "proposal_sample_ids": np.asarray(sample_ids),
        "proposal_groups": np.asarray(proposal_groups),
        "proposal_roles": np.asarray(proposal_roles),
        "scene_ids": np.asarray(scene_ids),
        "scene_groups": np.asarray(scene_groups),
        "scene_roles": np.asarray(scene_roles),
        "scene_labels": np.asarray(scene_labels, dtype=np.uint8),
        "scene_proposal_counts": np.asarray(scene_proposal_counts, dtype=np.uint16),
        "scene_neural_presence": np.asarray(scene_neural_presence, dtype=np.float32),
        "feature_names": np.asarray(proposal_feature_names(component_features.shape[1])),
        "ignored_ambiguous_proposals": np.asarray([ignored_ambiguous], dtype=np.int64),
    }, {"ignored_ambiguous_proposals": ignored_ambiguous}


def deterministic_calibration_groups(groups: np.ndarray) -> set[str]:
    unique = sorted(set(groups.astype(str)))
    ranked = sorted(
        unique,
        key=lambda group: hashlib.sha256(f"proposal-calibration:{group}".encode()).hexdigest(),
    )
    count = max(1, round(len(ranked) * CALIBRATION_GROUP_FRACTION))
    return set(ranked[:count])


def balanced_group_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    classes = Counter(int(value) for value in y)
    group_counts = Counter(groups.astype(str))
    return np.asarray(
        [
            (0.5 / classes[int(label)]) / math.sqrt(group_counts[str(group)])
            for label, group in zip(y, groups)
        ],
        dtype=np.float64,
    )


def calibrated_probabilities(
    model: HistGradientBoostingClassifier,
    calibrator: LogisticRegression,
    values: np.ndarray,
) -> np.ndarray:
    raw = model.predict_proba(values)[:, 1]
    raw_logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1))
    return calibrator.predict_proba(raw_logit[:, None])[:, 1]


def scene_scores(
    cache: dict[str, np.ndarray], proposal_probabilities: np.ndarray, role: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proposal_role = cache["proposal_roles"].astype(str)
    proposal_samples = cache["proposal_sample_ids"].astype(str)
    score_by_sample: dict[str, float] = {}
    for sample, probability in zip(proposal_samples[proposal_role == role], proposal_probabilities[proposal_role == role]):
        score_by_sample[sample] = max(score_by_sample.get(sample, 0.0), float(probability))
    scene_role = cache["scene_roles"].astype(str)
    selected = scene_role == role
    ids = cache["scene_ids"].astype(str)[selected]
    scores = np.asarray([score_by_sample.get(sample, 0.0) for sample in ids], dtype=np.float64)
    return (
        cache["scene_labels"][selected].astype(np.uint8),
        scores,
        cache["scene_groups"].astype(str)[selected],
    )


def neural_scene_scores(
    cache: dict[str, np.ndarray], role: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene_role = cache["scene_roles"].astype(str)
    selected = scene_role == role
    return (
        cache["scene_labels"][selected].astype(np.uint8),
        cache["scene_neural_presence"][selected].astype(np.float64),
        cache["scene_groups"].astype(str)[selected],
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    scene = report["validation"]["scene"]
    proposal_only = report["validation"]["ablations"]["proposal_only"]
    neural_only = report["validation"]["ablations"]["neural_only"]
    lines = [
        "# MARS v3 connected-proposal validation",
        "",
        "Stage-two classifier fit/calibration uses only internal-training groups; operating thresholds use disjoint internal validation.",
        "",
        f"- Proposals: {report['cache']['proposal_count']:,} total / {report['cache']['labeled_proposals']:,} labeled / {report['cache']['ignored_ambiguous_proposals']:,} ambiguous excluded from fitting",
        f"- Validation recall / specificity / FPR: {scene['recall']:.3f} / {scene['specificity']:.3f} / {scene['false_positive_rate']:.3f}",
        f"- Validation AUROC / AP: {scene['auroc']:.3f} / {scene['average_precision']:.3f}",
        f"- Selected neural-presence blend weight: {report['operating_rule']['neural_presence_weight']:.2f}",
        f"- Proposal-only recall / FPR: {proposal_only['recall']:.3f} / {proposal_only['false_positive_rate']:.3f}",
        f"- Neural-only recall / FPR: {neural_only['recall']:.3f} / {neural_only['false_positive_rate']:.3f}",
        f"- Artifact SHA-256: `{report['artifact']['sha256']}`",
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
    parser.add_argument("--cache-file", default=DEFAULT_CACHE)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for proposal extraction")
    metadata_dir = checked_output_dir(root, args.metadata_dir)
    metadata_csv = (root / args.metadata_csv).resolve()
    experiment_path = (root / args.experiment).resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("smoke_test") or experiment["scope"] != "v3_internal_validation_selection":
        raise ValueError("Connected-proposal training requires a full frozen v3 validation checkpoint")
    checkpoint = (root / experiment["artifact"]["path"]).resolve()
    if sha256(checkpoint) != experiment["artifact"]["sha256"]:
        raise ValueError("V3 checkpoint identity differs from its validation report")
    manifest = metadata_dir / V3_SAMPLES
    records = list(iter_manifest(manifest))
    selected_records = [
        record
        for record in records
        if record["research_role"] in {"internal_training", "internal_validation"}
    ]
    required_ids = {str(record["sample_id"]) for record in selected_records}
    winds = wind_lookup(metadata_csv, required_ids)
    device = torch.device("cuda")
    model = MarsV3Model().to(device)
    checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint_payload["state_dict"], strict=True)
    decoder_channels = 16
    identity = cache_identity(manifest, checkpoint, experiment_path, decoder_channels)
    cache_path = metadata_dir / args.cache_file
    extraction = {"ignored_ambiguous_proposals": 0}
    if cache_path.is_file() and not args.rebuild_cache:
        cache = load_cache(cache_path, identity)
    else:
        cache, extraction = extract_cache(
            metadata_dir,
            selected_records,
            winds,
            model,
            device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        write_cache(cache_path, cache, identity)
    if args.extract_only:
        print(
            json.dumps(
                {
                    "ok": True,
                    "cache": cache_path.relative_to(root).as_posix(),
                    "cache_sha256": sha256(cache_path),
                    "proposals": int(cache["y"].size),
                },
                indent=2,
            )
        )
        return 0

    x = cache["x"].astype(np.float32)
    y = cache["y"].astype(np.int8)
    roles = cache["proposal_roles"].astype(str)
    groups = cache["proposal_groups"].astype(str)
    labeled = y >= 0
    training = (roles == "internal_training") & labeled
    validation = (roles == "internal_validation") & labeled
    calibration_groups = deterministic_calibration_groups(groups[training])
    calibration = training & np.isin(groups, list(calibration_groups))
    fit = training & ~calibration
    for name, mask in (("fit", fit), ("calibration", calibration), ("validation", validation)):
        if set(np.unique(y[mask])) != {0, 1}:
            raise ValueError(f"Proposal {name} subset does not contain both classes")
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        # Avoid sklearn's proposal-level random early-stopping split, which
        # would mix spatial groups. Complexity is frozen a priori instead.
        early_stopping=False,
        random_state=args.seed,
    )
    classifier.fit(x[fit], y[fit], sample_weight=balanced_group_weights(y[fit], groups[fit]))
    calibration_raw = classifier.predict_proba(x[calibration])[:, 1]
    calibration_logit = np.log(
        np.clip(calibration_raw, 1e-6, 1 - 1e-6)
        / np.clip(1 - calibration_raw, 1e-6, 1)
    )
    calibrator = LogisticRegression(random_state=args.seed)
    calibrator.fit(
        calibration_logit[:, None],
        y[calibration],
        sample_weight=balanced_group_weights(y[calibration], groups[calibration]),
    )
    proposal_probability = calibrated_probabilities(classifier, calibrator, x)
    validation_y, proposal_scores, validation_groups = scene_scores(
        cache, proposal_probability, "internal_validation"
    )
    neural_y, neural_scores, neural_groups = neural_scene_scores(
        cache, "internal_validation"
    )
    if not (
        np.array_equal(validation_y, neural_y)
        and np.array_equal(validation_groups, neural_groups)
    ):
        raise ValueError("Neural and proposal validation scene ordering differs")
    blend_candidates: list[dict[str, Any]] = []
    selected_blend: tuple[tuple[float, ...], float, np.ndarray, float, dict[str, Any]] | None = None
    for neural_weight in NEURAL_PRESENCE_BLEND_WEIGHTS:
        scores = neural_weight * neural_scores + (1.0 - neural_weight) * proposal_scores
        candidate_upper, candidate_scene = choose_upper_threshold(validation_y, scores)
        rank = (
            float(candidate_scene["recall"] or 0.0),
            float(candidate_scene["average_precision"]),
            -float(candidate_scene["false_positive_rate"] or 0.0),
            neural_weight,
        )
        blend_candidates.append(
            {
                "neural_presence_weight": neural_weight,
                "upper_plume_threshold": candidate_upper,
                "scene": candidate_scene,
            }
        )
        candidate = (rank, neural_weight, scores, candidate_upper, candidate_scene)
        if selected_blend is None or candidate[0] > selected_blend[0]:
            selected_blend = candidate
    if selected_blend is None:
        raise RuntimeError("No neural/proposal blend candidate was evaluated")
    _, neural_weight, validation_scores, upper, validation_scene = selected_blend
    proposal_upper, proposal_only_scene = choose_upper_threshold(
        validation_y, proposal_scores
    )
    neural_upper, neural_only_scene = choose_upper_threshold(validation_y, neural_scores)
    validation_weights = role_weights(validation_y, "internal_validation")
    lower, lower_selection = choose_lower_threshold(
        validation_y, validation_scores, upper, validation_weights
    )

    artifact_path = safe_output(root, args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_artifact = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    joblib.dump(
        {
            "schema_version": 1,
            "classifier": classifier,
            "calibrator": calibrator,
            "feature_names": cache["feature_names"].astype(str).tolist(),
            "cache_identity": identity,
            "upper_plume_threshold": upper,
            "lower_no_plume_threshold": lower,
            "neural_presence_weight": neural_weight,
            "calibration_groups": sorted(calibration_groups),
            "seed": args.seed,
        },
        temporary_artifact,
        compress=3,
    )
    os.replace(temporary_artifact, artifact_path)
    report = {
        "schema_version": 1,
        "scope": "v3_connected_proposal_internal_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "UNEP-IMEO/MARS-S2L",
            "revision": REVISION,
            "manifest_sha256": sha256(manifest),
            "v3_validation_experiment_sha256": sha256(experiment_path),
            "v3_checkpoint_sha256": sha256(checkpoint),
        },
        "cache": {
            "path": cache_path.relative_to(root).as_posix(),
            "sha256": sha256(cache_path),
            "proposal_count": int(y.size),
            "labeled_proposals": int(np.sum(labeled)),
            "positive_proposals": int(np.sum(y == 1)),
            "negative_proposals": int(np.sum(y == 0)),
            "ignored_ambiguous_proposals": int(np.sum(y < 0)),
            "feature_count": int(x.shape[1]),
            "tracked": False,
        },
        "training": {
            "seed": args.seed,
            "fit_proposals": int(np.sum(fit)),
            "calibration_proposals": int(np.sum(calibration)),
            "validation_proposals": int(np.sum(validation)),
            "fit_groups": int(np.unique(groups[fit]).size),
            "calibration_groups": len(calibration_groups),
            "validation_groups": int(np.unique(validation_groups).size),
            "group_overlap": 0,
            "classifier": "HistGradientBoostingClassifier",
            "calibration": "group-held-out, class/group-balanced Platt logistic",
        },
        "operating_rule": {
            "selected_on": "internal_validation_only",
            "scene_score": "validation-selected convex blend of neural presence and maximum calibrated connected-proposal probability; proposal score is zero with no proposal",
            "neural_presence_weight": neural_weight,
            "upper_plume_threshold": upper,
            "lower_no_plume_threshold": lower,
            "lower_selection": lower_selection,
        },
        "validation": {
            "samples": int(validation_y.size),
            "scene": validation_scene,
            "blend_candidates": blend_candidates,
            "ablations": {
                "proposal_only": {
                    **proposal_only_scene,
                    "upper_plume_threshold": proposal_upper,
                },
                "neural_only": {
                    **neural_only_scene,
                    "upper_plume_threshold": neural_upper,
                },
            },
        },
        "artifact": {
            "path": artifact_path.relative_to(root).as_posix(),
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
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
            "script": "tools/train_mars_v3_proposals.py",
            "script_sha256": sha256(Path(__file__)),
            "proposal_source_sha256": sha256(MODEL_ROOT / "mars_v3_proposals.py"),
        },
        "decision": (
            "Freeze the calibrated component classifier with the v3 checkpoint before strict evaluation."
        ),
    }
    output_json = safe_output(root, args.output_json)
    output_markdown = safe_output(root, args.output_markdown)
    write_json(output_json, report)
    write_markdown(output_markdown, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
