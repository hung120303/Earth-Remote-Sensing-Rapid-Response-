#!/usr/bin/env python3
"""Cross-fit a same-background simulation-augmented MARS scene ranker."""

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
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from extract_mars_counterfactual_scene_inputs import CHANNEL_NAMES  # noqa: E402
from extract_mars_scene_features import atomic_savez  # noqa: E402
from train_mars_counterfactual_scene_ranker import (  # noqa: E402
    CounterfactualSceneRanker,
    align_images,
    augment_batch,
    batch_loss,
    current_spatial_prithvi_scores,
    evaluate_view,
    feature_indices,
    gates,
    predict_model,
)
from train_mars_oof_scene_ensemble_v2 import (  # noqa: E402
    ap_group_bootstrap,
    sample_weights,
)
from train_mars_scene_ranker import blend_scores  # noqa: E402
from train_mars_unseen_low_prevalence_router import low_prevalence_mask  # noqa: E402

DEFAULT_PROTOCOL = Path(
    "configs/mars_simulation_augmented_counterfactual_ranker_protocol.json"
)


def load_simulation_metadata(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        metadata = {name: values[name] for name in values.files}
    required = {
        "channel_names",
        "simulation_ids",
        "background_sample_ids",
        "source_sample_ids",
        "groups",
        "sensors",
        "fit_folds",
        "labels",
        "wind_speed_deltas",
        "scales",
        "rotations",
        "visible_fractions",
        "plume_pixels",
        "images_sha256",
        "protocol_sha256",
        "manifest_sha256",
    }
    if set(metadata) != required:
        raise ValueError("Simulation metadata differs from the frozen schema")
    if not np.array_equal(metadata["channel_names"].astype(str), np.asarray(CHANNEL_NAMES)):
        raise ValueError("Real and simulated counterfactual channels differ")
    if not np.all(metadata["labels"] == 1):
        raise ValueError("Every same-background intervention must be positive")
    if float(np.max(metadata["wind_speed_deltas"])) > 1.5 + 1e-6:
        raise ValueError("Simulation cache violates the strict wind gate")
    if float(np.min(metadata["visible_fractions"])) < 0.5:
        raise ValueError("Simulation cache violates the visibility gate")
    return metadata


def align_simulation_backgrounds(
    values: dict[str, np.ndarray],
    image_rows: np.ndarray,
    simulation: dict[str, np.ndarray],
) -> np.ndarray:
    lookup = {
        str(identifier): int(row)
        for identifier, row in zip(values["sample_ids"], image_rows, strict=True)
    }
    backgrounds = simulation["background_sample_ids"].astype(str)
    if any(identifier not in lookup for identifier in backgrounds):
        raise ValueError("A simulation background is outside development folds 3/4")
    aligned = np.asarray([lookup[identifier] for identifier in backgrounds], dtype=np.int64)
    local_fold = {
        str(identifier): int(fold)
        for identifier, fold in zip(values["sample_ids"], values["folds"], strict=True)
    }
    expected = np.asarray([local_fold[identifier] for identifier in backgrounds], dtype=np.uint8)
    if not np.array_equal(expected, simulation["fit_folds"]):
        raise ValueError("Simulation background fold differs from its fitting-fold contract")
    return aligned


def train_augmented_model(
    spec: dict[str, Any],
    real_images: np.ndarray,
    real_image_rows: np.ndarray,
    real_labels: np.ndarray,
    real_sensors: np.ndarray,
    real_weights: np.ndarray,
    simulation_images: np.ndarray,
    simulation_rows: np.ndarray,
    simulation_sensors: np.ndarray,
    background_image_rows: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    channels = feature_indices(str(spec["feature_set"]))
    model = CounterfactualSceneRanker(len(channels), float(spec["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positive_weight = min(
        4.0,
        float(
            np.sqrt(
                real_weights[real_labels == 0].sum()
                / max(real_weights[real_labels == 1].sum(), 1e-8)
            )
        ),
    )
    actual_batch = int(spec["actual_batch_size"])
    simulation_batch = int(spec["simulation_batch_size"])
    steps = int(np.ceil(real_labels.size / actual_batch))
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(int(spec["epochs"])):
        actual_order = torch.randperm(real_labels.size, generator=generator).numpy()
        simulation_order = torch.randperm(simulation_rows.size, generator=generator).numpy()
        simulation_cursor = 0
        totals = {"loss": 0.0, "actual": 0.0, "simulation": 0.0, "pair": 0.0}
        for step in range(steps):
            actual_indices = actual_order[step * actual_batch : (step + 1) * actual_batch]
            if simulation_cursor + simulation_batch > simulation_rows.size:
                simulation_order = torch.randperm(
                    simulation_rows.size, generator=generator
                ).numpy()
                simulation_cursor = 0
            synthetic_indices = simulation_order[
                simulation_cursor : simulation_cursor + simulation_batch
            ]
            simulation_cursor += simulation_batch
            selected_simulation_rows = simulation_rows[synthetic_indices]
            actual_array = np.asarray(
                real_images[real_image_rows[actual_indices]][:, channels], dtype=np.float32
            )
            simulation_array = np.asarray(
                simulation_images[selected_simulation_rows][:, channels],
                dtype=np.float32,
            )
            background_array = np.asarray(
                real_images[background_image_rows[selected_simulation_rows]][:, channels],
                dtype=np.float32,
            )
            combined = np.concatenate(
                [actual_array, simulation_array, background_array], axis=0
            )
            image_tensor = augment_batch(torch.from_numpy(combined), generator).to(device)
            actual_sensor = real_sensors[actual_indices].astype(np.int64)
            synthetic_sensor = simulation_sensors[selected_simulation_rows].astype(np.int64)
            sensor_tensor = torch.from_numpy(
                np.concatenate([actual_sensor, synthetic_sensor, synthetic_sensor])
            ).to(device)
            logits = model(image_tensor, sensor_tensor)
            actual_count = actual_indices.size
            synthetic_count = synthetic_indices.size
            actual_logits = logits[:actual_count]
            simulation_logits = logits[actual_count : actual_count + synthetic_count]
            background_logits = logits[actual_count + synthetic_count :]
            labels = torch.from_numpy(real_labels[actual_indices].astype(np.float32)).to(device)
            weights = torch.from_numpy(real_weights[actual_indices].astype(np.float32)).to(
                device
            )
            actual_loss = batch_loss(
                actual_logits,
                labels,
                weights,
                positive_weight=positive_weight,
                pair_weight=float(spec["actual_hard_pair_weight"]),
            )
            simulation_loss = F.softplus(-simulation_logits).mean()
            pair_loss = F.softplus(
                background_logits - simulation_logits + float(spec["causal_margin"])
            ).mean()
            loss = (
                actual_loss
                + float(spec["simulation_loss_weight"]) * simulation_loss
                + float(spec["causal_pair_weight"]) * pair_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["actual"] += float(actual_loss.detach().cpu())
            totals["simulation"] += float(simulation_loss.detach().cpu())
            totals["pair"] += float(pair_loss.detach().cpu())
        history.append(
            {"epoch": epoch + 1, **{key: value / steps for key, value in totals.items()}}
        )
    return {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "input_channels": len(channels),
        "channel_indices": channels,
        "dropout": float(spec["dropout"]),
        "positive_weight": positive_weight,
        "history": history,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report.get("selected")
    lines = [
        "# Simulation-augmented counterfactual scene ranker",
        "",
        "Selection used honest fold-3/fold-4 cross-fitting with fit-fold-only plume sources and no-plume backgrounds.",
        "",
    ]
    if selected is None:
        lines.append("No same-background simulation candidate passed every preregistered gate.")
    else:
        lines.extend(
            [
                f"- Model: `{selected['model_key']}`",
                f"- Blend weight: {selected['blend_weight']:.3f}",
                f"- Whole AP delta: {selected['whole']['versus_current']['delta']['average_precision']:+.6f}",
                f"- Rare-site AP delta: {selected['rare']['versus_current']['delta']['average_precision']:+.6f}",
                f"- Whole AP interval: [{selected['whole']['bootstrap']['lower']:+.6f}, {selected['whole']['bootstrap']['upper']:+.6f}]",
                f"- Rare AP interval: [{selected['rare']['bootstrap']['lower']:+.6f}, {selected['rare']['bootstrap']['upper']:+.6f}]",
            ]
        )
    lines.extend(["", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen simulation-augmented trainer hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen trainer dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen augmented-ranker input hash mismatch: {name}")
        paths[name] = path

    real_images = np.load(paths["real_images"], mmap_mode="r", allow_pickle=False)
    simulation_images = np.load(
        paths["simulation_images"], mmap_mode="r", allow_pickle=False
    )
    if real_images.shape[1:] != simulation_images.shape[1:] or real_images.dtype != np.float16:
        raise ValueError("Real/simulated counterfactual image schemas differ")
    values = current_spatial_prithvi_scores(
        paths, float(protocol["current_comparator"]["prithvi_weight"])
    )
    real_image_rows, values = align_images(values, paths["real_metadata"])
    simulation = load_simulation_metadata(paths["simulation_metadata"])
    if simulation_images.shape[0] != simulation["labels"].size:
        raise ValueError("Simulation image and metadata row counts differ")
    background_rows = align_simulation_backgrounds(values, real_image_rows, simulation)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Simulation-augmented ranker requires CUDA")
    torch.set_float32_matmul_precision("high")
    if args.smoke:
        spec = dict(protocol["models"][0])
        spec["epochs"] = 1
        spec["actual_batch_size"] = 8
        spec["simulation_batch_size"] = 4
        actual = np.arange(16, dtype=np.int64)
        synthetic = np.flatnonzero(simulation["fit_folds"] == 4)[:8]
        fitted = train_augmented_model(
            spec,
            real_images,
            real_image_rows[actual],
            values["labels"][actual],
            values["sensors"][actual],
            np.ones(actual.size, dtype=np.float64),
            simulation_images,
            synthetic,
            simulation["sensors"],
            background_rows,
            seed=int(protocol["seeds"][0]),
            device=device,
        )
        scores = predict_model(
            fitted,
            real_images,
            real_image_rows[actual],
            values["sensors"][actual],
            device,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "smoke": True,
                    "finite_scores": bool(np.isfinite(scores).all()),
                    "history": fitted["history"],
                    "strict_simulation_wind_gate": float(
                        np.max(simulation["wind_speed_deltas"])
                    ),
                },
                indent=2,
            )
        )
        return 0

    rare_rows = low_prevalence_mask(
        values["labels"], values["groups"], float(protocol["target_view"]["maximum_site_positive_rate"])
    )
    all_rows = np.ones(values["labels"].shape, dtype=bool)
    candidates: list[dict[str, Any]] = []
    raw_by_model: dict[str, np.ndarray] = {}
    for model_index, spec in enumerate(protocol["models"]):
        model_key = str(spec["model_key"])
        raw = np.empty(values["labels"].shape, dtype=np.float64)
        for holdout in (3, 4):
            fit_fold = 4 if holdout == 3 else 3
            fit_rows = values["folds"] == fit_fold
            held_rows = values["folds"] == holdout
            simulation_rows = np.flatnonzero(simulation["fit_folds"] == fit_fold)
            real_weights = sample_weights(
                str(spec["weighting"]),
                values["groups"][fit_rows],
                values["labels"][fit_rows],
                values["sensors"][fit_rows],
            )
            seed_scores: list[np.ndarray] = []
            for seed in protocol["seeds"]:
                fitted = train_augmented_model(
                    spec,
                    real_images,
                    real_image_rows[fit_rows],
                    values["labels"][fit_rows],
                    values["sensors"][fit_rows],
                    real_weights,
                    simulation_images,
                    simulation_rows,
                    simulation["sensors"],
                    background_rows,
                    seed=int(seed) + holdout,
                    device=device,
                )
                seed_scores.append(
                    predict_model(
                        fitted,
                        real_images,
                        real_image_rows[held_rows],
                        values["sensors"][held_rows],
                        device,
                    )
                )
            stacked = np.clip(np.stack(seed_scores), 1e-5, 1.0 - 1e-5)
            raw[held_rows] = 1.0 / (
                1.0 + np.exp(-np.mean(np.log(stacked / (1.0 - stacked)), axis=0))
            )
        raw_by_model[model_key] = raw
        for blend_weight in protocol["blend_weights"]:
            scores = blend_scores(values["current"], raw, float(blend_weight))
            whole = evaluate_view(values, scores, all_rows)
            rare = evaluate_view(values, scores, rare_rows)
            whole["bootstrap"] = ap_group_bootstrap(
                values["labels"], values["current"], scores, values["groups"],
                replicates=int(protocol["bootstrap"]["replicates"]),
                seed=int(protocol["bootstrap"]["whole_seed"]) + model_index * 100 + int(float(blend_weight) * 100),
            )
            rare["bootstrap"] = ap_group_bootstrap(
                values["labels"][rare_rows], values["current"][rare_rows], scores[rare_rows], values["groups"][rare_rows],
                replicates=int(protocol["bootstrap"]["replicates"]),
                seed=int(protocol["bootstrap"]["rare_seed"]) + model_index * 100 + int(float(blend_weight) * 100),
            )
            checks = gates(whole, rare, protocol["gates"])
            candidate = {
                "model_key": model_key,
                "blend_weight": float(blend_weight),
                "whole": whole,
                "rare": rare,
                "checks": checks,
                "passed": all(checks.values()),
            }
            candidate["rank"] = [
                int(candidate["passed"]),
                rare["bootstrap"]["lower"],
                whole["bootstrap"]["lower"],
                rare["versus_current"]["delta"]["average_precision"],
                whole["versus_current"]["delta"]["average_precision"],
                -float(blend_weight),
            ]
            candidates.append(candidate)
        best = max(
            [value for value in candidates if value["model_key"] == model_key],
            key=lambda value: tuple(value["rank"]),
        )
        print(
            json.dumps(
                {
                    "model": model_index + 1,
                    "models": len(protocol["models"]),
                    "model_key": model_key,
                    "best_weight": best["blend_weight"],
                    "whole_ap_delta": best["whole"]["versus_current"]["delta"]["average_precision"],
                    "rare_ap_delta": best["rare"]["versus_current"]["delta"]["average_precision"],
                    "passed": best["passed"],
                }
            ),
            flush=True,
        )
    winner = max(candidates, key=lambda value: tuple(value["rank"]))
    selected = winner if winner["passed"] else None
    artifact_record = None
    if selected is not None:
        spec = next(
            value for value in protocol["models"] if value["model_key"] == selected["model_key"]
        )
        real_weights = sample_weights(
            str(spec["weighting"]), values["groups"], values["labels"], values["sensors"]
        )
        all_simulation_rows = np.arange(simulation["labels"].size, dtype=np.int64)
        endpoints = [
            train_augmented_model(
                spec,
                real_images,
                real_image_rows,
                values["labels"],
                values["sensors"],
                real_weights,
                simulation_images,
                all_simulation_rows,
                simulation["sensors"],
                background_rows,
                seed=int(seed) + 34,
                device=device,
            )
            for seed in protocol["seeds"]
        ]
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
        torch.save(
            {
                "schema_version": 1,
                "kind": "mars_simulation_augmented_counterfactual_ranker_folds34",
                "fit_folds": [3, 4],
                "forbidden_folds": [0, 1, 2],
                "spec": spec,
                "blend_weight": selected["blend_weight"],
                "channel_names": list(CHANNEL_NAMES),
                "endpoints": endpoints,
                "protocol_sha256": sha256(protocol_path),
            },
            temporary,
        )
        os.replace(temporary, artifact_path)
        artifact_record = {
            "path": protocol["outputs"]["artifact"],
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
            "tracked": False,
        }
    score_path = (ROOT / protocol["outputs"]["crossfit_scores"]).resolve()
    atomic_savez(
        score_path,
        sample_ids=values["sample_ids"], labels=values["labels"], sensors=values["sensors"],
        groups=values["groups"], folds=values["folds"], current_scores=values["current"],
        **{f"raw_{key}": score for key, score in raw_by_model.items()},
        protocol_sha256=np.asarray(sha256(protocol_path)),
    )
    report = {
        "schema_version": 1,
        "scope": "fold-3/fold-4 honest cross-fit with fit-fold-only same-background simulation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_comparator": protocol["current_comparator"],
        "simulation_contract": protocol["simulation_contract"],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "artifact": artifact_record,
        "all_promotion_gates_pass": selected is not None,
        "decision": (
            "Freeze the causal ranker for separately preregistered fold-2 confirmation."
            if selected is not None
            else "Reject same-background simulation augmentation before fold-0/1/2 or external extraction."
        ),
        "provenance": {
            "protocol_sha256": sha256(protocol_path),
            "script_sha256": sha256(Path(__file__).resolve()),
            "crossfit_scores_sha256": sha256(score_path),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_json, output_json)
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(
        json.dumps(
            {
                "ok": selected is not None,
                "selected": None if selected is None else {
                    "model_key": selected["model_key"],
                    "blend_weight": selected["blend_weight"],
                    "whole_ap_delta": selected["whole"]["versus_current"]["delta"]["average_precision"],
                    "rare_ap_delta": selected["rare"]["versus_current"]["delta"]["average_precision"],
                },
                "artifact": artifact_record,
            },
            indent=2,
        )
    )
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
