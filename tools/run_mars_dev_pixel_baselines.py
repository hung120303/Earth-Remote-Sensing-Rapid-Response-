#!/usr/bin/env python3
"""Run validity-aware spatial MBMP and pixel-logistic MARS baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import sklearn
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_s2l_adapter import iter_manifest, load_sample  # noqa: E402

from acquire_mars_metadata import DEFAULT_OUTPUT, REVISION, checked_output_dir, repo_root, sha256  # noqa: E402
from build_mars_dev_cohort import DEV_SAMPLES, DEFAULT_JSON as DEV_REPORT_JSON  # noqa: E402

DEFAULT_JSON = Path("reports/experiments/mars_dev_pixel_baselines.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_DEV_PIXEL_BASELINES.md")
SCORE_CACHE = "publication_dev_pixel_scores.npz"
MODEL_SEED = 202
MAX_POSITIVE_PIXELS_PER_TRAIN_SCENE = 512
MAX_NEGATIVE_PIXELS_PER_TRAIN_SCENE = 512
MIN_COMPONENT_PIXELS = (10, 25, 50, 100, 200)
BOOTSTRAP_REPLICATES = 2_000
PIXEL_FEATURE_NAMES = (
    "one_minus_valid_mbmp",
    "normalized_change_B02",
    "normalized_change_B03",
    "normalized_change_B04",
    "normalized_change_B08",
    "normalized_change_B11",
    "normalized_change_B12",
    "log_change_B02",
    "log_change_B03",
    "log_change_B04",
    "log_change_B08",
    "log_change_B11",
    "log_change_B12",
)


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def tracked_dirty(root: Path) -> bool:
    output = subprocess.check_output(
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
    return bool(output.strip())


def safe_output(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if root not in path.parents:
        raise ValueError("Experiment output must resolve beneath the repository root")
    return path


def pixel_features(sample: Any) -> np.ndarray:
    normalized_change = (sample.target - sample.reference) / (
        sample.target + sample.reference + np.float32(1e-4)
    )
    log_change = np.log1p(sample.target) - np.log1p(sample.reference)
    mbmp = (1.0 - sample.mbmp_valid_aware)[None, ...]
    return np.concatenate([mbmp, normalized_change, log_change], axis=0).astype(np.float32)


def fit_pixel_model(metadata_dir: Path, records: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
    rng = np.random.default_rng(MODEL_SEED)
    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    processed = 0
    positive_pixels = 0
    negative_pixels = 0
    for record in records:
        if record["research_role"] != "internal_training":
            continue
        sample = load_sample(metadata_dir, record)
        features = np.moveaxis(pixel_features(sample), 0, -1).reshape(-1, len(PIXEL_FEATURE_NAMES))
        positive = np.flatnonzero((sample.plume_mask & sample.observable_mask).ravel())
        negative = np.flatnonzero((~sample.plume_mask & sample.observable_mask).ravel())
        if positive.size > MAX_POSITIVE_PIXELS_PER_TRAIN_SCENE:
            positive = rng.choice(
                positive, MAX_POSITIVE_PIXELS_PER_TRAIN_SCENE, replace=False
            )
        if negative.size > MAX_NEGATIVE_PIXELS_PER_TRAIN_SCENE:
            negative = rng.choice(
                negative, MAX_NEGATIVE_PIXELS_PER_TRAIN_SCENE, replace=False
            )
        selected = np.concatenate([positive, negative])
        feature_rows.append(features[selected])
        labels.append(
            np.concatenate(
                [
                    np.ones(positive.size, dtype=np.uint8),
                    np.zeros(negative.size, dtype=np.uint8),
                ]
            )
        )
        positive_pixels += int(positive.size)
        negative_pixels += int(negative.size)
        processed += 1
        if processed % 200 == 0:
            print(f"Prepared pixel training scenes: {processed:,}/768", file=sys.stderr, flush=True)
    x_train = np.concatenate(feature_rows)
    y_train = np.concatenate(labels)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=2_000,
            random_state=MODEL_SEED,
            solver="lbfgs",
        ),
    )
    model.fit(x_train, y_train)
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    digest = hashlib.sha256()
    for array in (scaler.mean_, scaler.scale_, classifier.coef_, classifier.intercept_):
        digest.update(np.asarray(array, dtype=np.float64).tobytes())
    return model, {
        "training_scenes": processed,
        "sampled_positive_pixels": positive_pixels,
        "sampled_negative_pixels": negative_pixels,
        "feature_names": list(PIXEL_FEATURE_NAMES),
        "model_identity_sha256": digest.hexdigest(),
    }


def score_logistic(model: Any, sample: Any) -> np.ndarray:
    features = np.moveaxis(pixel_features(sample), 0, -1).reshape(-1, len(PIXEL_FEATURE_NAMES))
    result = model.predict_proba(features)[:, 1].reshape(sample.plume_mask.shape)
    result[~sample.observable_mask] = 0.0
    return result.astype(np.float32)


def score_cache_identity(manifest_path: Path, model_info: dict[str, Any]) -> dict[str, str]:
    return {
        "manifest_sha256": sha256(manifest_path),
        "adapter_sha256": sha256(MODEL_ROOT / "mars_s2l_adapter.py"),
        "model_identity_sha256": model_info["model_identity_sha256"],
        "score_schema": "mars_dev_pixel_scores_v1",
    }


def build_score_cache(
    metadata_dir: Path,
    records: list[dict[str, Any]],
    model: Any,
    model_info: dict[str, Any],
    cache_path: Path,
    manifest_path: Path,
) -> dict[str, np.ndarray]:
    evaluation_records = [
        record
        for record in records
        if record["research_role"] in {"internal_validation", "strict_spatial_test"}
    ]
    mbmp_scores: list[np.ndarray] = []
    logistic_scores: list[np.ndarray] = []
    observable_packed: list[np.ndarray] = []
    truth_packed: list[np.ndarray] = []
    presence: list[int] = []
    roles: list[str] = []
    groups: list[str] = []
    sample_ids: list[str] = []
    for index, record in enumerate(evaluation_records, start=1):
        sample = load_sample(metadata_dir, record)
        mbmp = (1.0 - sample.mbmp_valid_aware).astype(np.float32)
        mbmp[~sample.observable_mask] = 0.0
        logistic = score_logistic(model, sample)
        mbmp_scores.append(mbmp.astype(np.float16))
        logistic_scores.append(logistic.astype(np.float16))
        observable_packed.append(np.packbits(sample.observable_mask.ravel()))
        truth_packed.append(np.packbits(sample.plume_mask.ravel()))
        presence.append(sample.presence)
        roles.append(record["research_role"])
        groups.append(record["group_id"])
        sample_ids.append(sample.sample_id)
        if index % 200 == 0 or index == len(evaluation_records):
            print(
                f"Scored evaluation scenes: {index:,}/{len(evaluation_records):,}",
                file=sys.stderr,
                flush=True,
            )
    identity = score_cache_identity(manifest_path, model_info)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        mbmp=np.stack(mbmp_scores),
        logistic=np.stack(logistic_scores),
        observable=np.stack(observable_packed),
        truth=np.stack(truth_packed),
        presence=np.asarray(presence, dtype=np.uint8),
        roles=np.asarray(roles),
        groups=np.asarray(groups),
        sample_ids=np.asarray(sample_ids),
        identity_json=np.asarray([json.dumps(identity, sort_keys=True)]),
    )
    os.replace(temporary, cache_path)
    return load_score_cache(cache_path, identity)


def load_score_cache(path: Path, expected: dict[str, str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        observed = json.loads(str(source["identity_json"][0]))
        if observed != expected:
            raise ValueError("Pixel score cache identity does not match the current model/manifest")
        return {key: source[key].copy() for key in source.files}


def unpack(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed, count=40_000).reshape(200, 200).astype(bool)


def candidate_thresholds(scores: np.ndarray, observables: np.ndarray) -> list[float]:
    observed_values = np.concatenate(
        [score.ravel()[unpack(mask).ravel()] for score, mask in zip(scores, observables)]
    ).astype(np.float32)
    quantiles = (
        0.80,
        0.85,
        0.90,
        0.93,
        0.95,
        0.97,
        0.98,
        0.99,
        0.995,
        0.997,
        0.999,
        0.9995,
        0.9999,
    )
    return sorted(set(float(value) for value in np.quantile(observed_values, quantiles)))


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel()) if count else np.asarray([mask.size])
    return labels, sizes


def scene_confusion(y: np.ndarray, predicted: np.ndarray) -> dict[str, int | float | None]:
    tp = int(np.sum((y == 1) & predicted))
    tn = int(np.sum((y == 0) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "specificity": None if tn + fp == 0 else tn / (tn + fp),
        "false_positive_rate": None if fp + tn == 0 else fp / (fp + tn),
        "precision": None if tp + fp == 0 else tp / (tp + fp),
    }


def select_rule(
    scores: np.ndarray,
    observable_packed: np.ndarray,
    truth_packed: np.ndarray,
    presence: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    truth_area = sum(
        int(np.count_nonzero(unpack(truth) & unpack(observable)))
        for truth, observable in zip(truth_packed, observable_packed)
    )
    best: tuple[tuple[float, ...], dict[str, Any], dict[str, Any]] | None = None
    for threshold in candidate_thresholds(scores, observable_packed):
        prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for score, obs_bits, truth_bits in zip(scores, observable_packed, truth_packed):
            observable = unpack(obs_bits)
            truth = unpack(truth_bits) & observable
            labels, sizes = component_labels((score.astype(np.float32) >= threshold) & observable)
            prepared.append((labels, sizes, truth))
        for minimum_pixels in MIN_COMPONENT_PIXELS:
            scene_predictions: list[bool] = []
            intersection = 0
            predicted_area = 0
            for labels, sizes, truth in prepared:
                keep = sizes >= minimum_pixels
                keep[0] = False
                prediction = keep[labels]
                scene_predictions.append(bool(np.any(prediction)))
                intersection += int(np.count_nonzero(prediction & truth))
                predicted_area += int(np.count_nonzero(prediction))
            scene = scene_confusion(presence, np.asarray(scene_predictions, dtype=bool))
            dice = (
                0.0
                if predicted_area + truth_area == 0
                else 2.0 * intersection / (predicted_area + truth_area)
            )
            fpr = float(scene["false_positive_rate"] or 0.0)
            recall = float(scene["recall"] or 0.0)
            feasible = 1.0 if fpr <= 0.05 else 0.0
            rank = (feasible, recall if feasible else -fpr, dice, -fpr, -minimum_pixels)
            rule = {"pixel_threshold": threshold, "minimum_connected_pixels": minimum_pixels}
            details = {**scene, "pixel_dice": dice}
            candidate = (rank, rule, details)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        raise RuntimeError("No pixel operating rule candidates were evaluated")
    return best[1], best[2]


def evaluate_rule(
    scores: np.ndarray,
    observable_packed: np.ndarray,
    truth_packed: np.ndarray,
    presence: np.ndarray,
    groups: np.ndarray,
    rule: dict[str, Any],
) -> dict[str, Any]:
    predicted_scenes: list[bool] = []
    pixel_truth: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    intersection = predicted_area = truth_area = 0
    for score, obs_bits, truth_bits in zip(scores, observable_packed, truth_packed):
        observable = unpack(obs_bits)
        truth = unpack(truth_bits) & observable
        candidate = (score.astype(np.float32) >= float(rule["pixel_threshold"])) & observable
        labels, sizes = component_labels(candidate)
        keep = sizes >= int(rule["minimum_connected_pixels"])
        keep[0] = False
        prediction = keep[labels]
        predicted_scenes.append(bool(np.any(prediction)))
        intersection += int(np.count_nonzero(prediction & truth))
        predicted_area += int(np.count_nonzero(prediction))
        truth_area += int(np.count_nonzero(truth))
        pixel_truth.append(truth[observable].astype(np.uint8))
        pixel_scores.append(score.ravel()[observable.ravel()])
    scene = scene_confusion(presence, np.asarray(predicted_scenes, dtype=bool))
    union = predicted_area + truth_area - intersection
    scene["pixel"] = {
        "average_precision": float(
            average_precision_score(np.concatenate(pixel_truth), np.concatenate(pixel_scores))
        ),
        "intersection_over_union": 0.0 if union == 0 else intersection / union,
        "dice": 0.0
        if predicted_area + truth_area == 0
        else 2.0 * intersection / (predicted_area + truth_area),
        "truth_positive_pixels": truth_area,
        "predicted_positive_pixels": predicted_area,
    }
    scene["group_bootstrap"] = bootstrap_scene(
        presence,
        np.asarray(predicted_scenes, dtype=bool),
        groups,
    )
    return scene


def bootstrap_scene(y: np.ndarray, predicted: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(MODEL_SEED)
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    recalls: list[float] = []
    specificities: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in selected])
        result = scene_confusion(y[indices], predicted[indices])
        if result["recall"] is not None:
            recalls.append(float(result["recall"]))
        if result["specificity"] is not None:
            specificities.append(float(result["specificity"]))
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "unit": "frozen 25 km group",
        "recall_95ci": [float(value) for value in np.quantile(recalls, (0.025, 0.975))],
        "specificity_95ci": [
            float(value) for value in np.quantile(specificities, (0.025, 0.975))
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MARS-S2L group-disjoint pixel baselines",
        "",
        "Predeclared spatial baseline ladder on the verified development tranche; not a final paper estimate.",
        "",
        f"- Pixel logistic training: {report['training']['training_scenes']} scenes / {report['training']['sampled_positive_pixels']:,} plume pixels / {report['training']['sampled_negative_pixels']:,} background pixels",
        "- Operating threshold and minimum component area selected on 384 internal-validation scenes only",
        "- Benchmark: 579 strict-spatial scenes / 150 groups; 67 plume / 512 no plume",
        "",
        "| Model | Val recall | Val FPR | Test recall | Test specificity | Pixel AP | Pixel IoU | Recall 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in report["models"]:
        val = model["validation"]
        test = model["test"]
        ci = test["group_bootstrap"]["recall_95ci"]
        lines.append(
            f"| {model['name']} | {fmt(val['recall'])} | {fmt(val['false_positive_rate'])} | "
            f"{fmt(test['recall'])} | {fmt(test['specificity'])} | "
            f"{test['pixel']['average_precision']:.4f} | {test['pixel']['intersection_over_union']:.4f} | "
            f"{ci[0]:.3f}-{ci[1]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"],
            "",
            "These baselines test whether local target/reference spectra and MBMP alone provide an adequate operating rule. Candidate neural architecture and calibration remain validation-only until frozen; this benchmark is not used for hyperparameter search.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--refresh-scores", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    try:
        metadata_dir = checked_output_dir(root, args.metadata_dir)
        manifest_path = metadata_dir / DEV_SAMPLES
        dev_report = json.loads((root / DEV_REPORT_JSON).read_text(encoding="utf-8"))
        if sha256(manifest_path) != dev_report["identities"]["sample_manifest_sha256"]:
            raise ValueError("Development sample manifest identity mismatch")
        records = list(iter_manifest(manifest_path))
        model, model_info = fit_pixel_model(metadata_dir, records)
        identity = score_cache_identity(manifest_path, model_info)
        cache_path = metadata_dir / SCORE_CACHE
        if cache_path.is_file() and not args.refresh_scores:
            data = load_score_cache(cache_path, identity)
        else:
            data = build_score_cache(
                metadata_dir, records, model, model_info, cache_path, manifest_path
            )
        roles = data["roles"].astype(str)
        validation = roles == "internal_validation"
        test = roles == "strict_spatial_test"
        models: list[dict[str, Any]] = []
        for name, key in (
            ("valid_aware_mbmp", "mbmp"),
            ("pixel_logistic_13_features", "logistic"),
        ):
            rule, val_metrics = select_rule(
                data[key][validation],
                data["observable"][validation],
                data["truth"][validation],
                data["presence"][validation],
            )
            test_metrics = evaluate_rule(
                data[key][test],
                data["observable"][test],
                data["truth"][test],
                data["presence"][test],
                data["groups"][test].astype(str),
                rule,
            )
            models.append(
                {
                    "name": name,
                    "operating_rule": {
                        **rule,
                        "selected_on": "internal_validation",
                        "objective": "maximum scene recall at observed FPR <= 0.05, then pixel Dice",
                    },
                    "validation": val_metrics,
                    "test": test_metrics,
                }
            )
        validation_selected = max(
            models,
            key=lambda item: (
                float(item["validation"]["recall"] or 0.0),
                float(item["validation"]["pixel_dice"]),
                -float(item["validation"]["false_positive_rate"] or 0.0),
            ),
        )
        selected_test = validation_selected["test"]
        gate = (
            float(selected_test["group_bootstrap"]["recall_95ci"][0]) >= 0.75
            and float(selected_test["false_positive_rate"] or 1.0) <= 0.05
            and float(selected_test["specificity"] or 0.0) >= 0.95
        )
        decision = (
            f"Validation-selected spatial baseline: `{validation_selected['name']}`. "
            + (
                "It clears the provisional development gate; reproduce on the full cohort before promotion."
                if gate
                else "It does not clear the promotion gate. Local per-pixel spectra are insufficient; implement the predeclared multi-scale target/reference encoder with joint scene presence, segmentation, and observability heads."
            )
        )
        output_json = safe_output(root, args.output_json)
        output_markdown = safe_output(root, args.output_markdown)
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "predeclared_spatial_development_baselines_not_final_paper_claim",
            "source": {
                "repository": "UNEP-IMEO/MARS-S2L",
                "revision": REVISION,
                "development_manifest_sha256": identity["manifest_sha256"],
                "score_cache_identity": identity,
            },
            "training": model_info,
            "models": models,
            "validation_selected_model": validation_selected["name"],
            "promotion_gate_passed_on_development_tranche": gate,
            "decision": decision,
            "limitations": [
                "Class-enriched development tranche, not deployment prevalence.",
                "Single classical fit seed; learned candidate requires five seeds.",
                "Per-pixel logistic has no spatial receptive field beyond connected-component filtering.",
                "Released MARS-S2L and CH4Net reproduction remains outstanding.",
            ],
            "provenance": {
                "git_commit": git_commit(root),
                "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
                "script": "tools/run_mars_dev_pixel_baselines.py",
                "script_sha256": sha256(Path(__file__)),
                "adapter_sha256": identity["adapter_sha256"],
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "rasterio": rasterio.__version__,
                "sklearn": sklearn.__version__,
            },
        }
        write_json(output_json, report)
        write_markdown(output_markdown, report)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, rasterio.errors.RasterioError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=None if args.compact else 2))
        return 2
    payload = {
        "ok": True,
        "selected": validation_selected["name"],
        "gate_passed": gate,
        "models": [
            {
                "name": model["name"],
                "test_recall": model["test"]["recall"],
                "test_specificity": model["test"]["specificity"],
                "test_pixel_ap": model["test"]["pixel"]["average_precision"],
            }
            for model in models
        ],
        "output_json": output_json.relative_to(root).as_posix(),
        "output_markdown": output_markdown.relative_to(root).as_posix(),
    }
    print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
