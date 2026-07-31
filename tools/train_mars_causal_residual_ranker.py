#!/usr/bin/env python3
"""Cross-fit a baseline-preserving causal residual scene ranker on folds 3/4."""

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
from scipy.special import expit
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
)
from train_mars_oof_scene_ensemble_v2 import (  # noqa: E402
    ap_group_bootstrap,
    sample_weights,
)
from train_mars_scene_ranker import safe_logit  # noqa: E402
from train_mars_simulation_augmented_counterfactual_ranker import (  # noqa: E402
    align_simulation_backgrounds,
    load_simulation_metadata,
)
from train_mars_unseen_low_prevalence_router import low_prevalence_mask  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_causal_residual_ranker_protocol.json")


def new_residual_model(input_channels: int, dropout: float) -> CounterfactualSceneRanker:
    model = CounterfactualSceneRanker(input_channels, dropout)
    output = model.classifier[-1]
    if not isinstance(output, torch.nn.Linear) or output.out_features != 1:
        raise ValueError("Counterfactual ranker output layer differs from residual schema")
    torch.nn.init.zeros_(output.weight)
    torch.nn.init.zeros_(output.bias)
    return model


def train_residual_model(
    spec: dict[str, Any],
    real_images: np.ndarray,
    real_image_rows: np.ndarray,
    real_labels: np.ndarray,
    real_sensors: np.ndarray,
    real_scores: np.ndarray,
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
    model = new_residual_model(len(channels), float(spec["dropout"])).to(device)
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
        totals = {
            "loss": 0.0,
            "actual": 0.0,
            "simulation": 0.0,
            "causal_pair": 0.0,
            "residual_l2": 0.0,
        }
        for step in range(steps):
            actual_indices = actual_order[step * actual_batch : (step + 1) * actual_batch]
            if simulation_cursor + simulation_batch > simulation_rows.size:
                simulation_order = torch.randperm(
                    simulation_rows.size, generator=generator
                ).numpy()
                simulation_cursor = 0
            local_simulation = simulation_order[
                simulation_cursor : simulation_cursor + simulation_batch
            ]
            simulation_cursor += simulation_batch
            selected_simulation = simulation_rows[local_simulation]

            actual_array = np.asarray(
                real_images[real_image_rows[actual_indices]][:, channels], dtype=np.float32
            )
            simulation_array = np.asarray(
                simulation_images[selected_simulation][:, channels], dtype=np.float32
            )
            background_array = np.asarray(
                real_images[background_image_rows[selected_simulation]][:, channels],
                dtype=np.float32,
            )
            combined = np.concatenate(
                [actual_array, simulation_array, background_array], axis=0
            )
            images = augment_batch(torch.from_numpy(combined), generator).to(device)
            actual_sensor = real_sensors[actual_indices].astype(np.int64)
            synthetic_sensor = simulation_sensors[selected_simulation].astype(np.int64)
            sensors = torch.from_numpy(
                np.concatenate([actual_sensor, synthetic_sensor, synthetic_sensor])
            ).to(device)
            residual = model(images, sensors)
            actual_count = actual_indices.size
            synthetic_count = selected_simulation.size
            actual_residual = residual[:actual_count]
            simulation_residual = residual[actual_count : actual_count + synthetic_count]
            background_residual = residual[actual_count + synthetic_count :]
            baseline = torch.from_numpy(
                safe_logit(real_scores[actual_indices]).astype(np.float32)
            ).to(device)
            candidate_logits = baseline + actual_residual
            labels = torch.from_numpy(real_labels[actual_indices].astype(np.float32)).to(
                device
            )
            weights = torch.from_numpy(real_weights[actual_indices].astype(np.float32)).to(
                device
            )
            actual_loss = batch_loss(
                candidate_logits,
                labels,
                weights,
                positive_weight=positive_weight,
                pair_weight=float(spec["actual_hard_pair_weight"]),
            )
            simulation_loss = F.softplus(-simulation_residual).mean()
            causal_pair = F.softplus(
                background_residual
                - simulation_residual
                + float(spec["causal_margin"])
            ).mean()
            residual_l2 = torch.cat(
                [actual_residual, simulation_residual, background_residual]
            ).square().mean()
            loss = (
                actual_loss
                + float(spec["simulation_loss_weight"]) * simulation_loss
                + float(spec["causal_pair_weight"]) * causal_pair
                + float(spec["residual_l2_weight"]) * residual_l2
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in (
                ("loss", loss),
                ("actual", actual_loss),
                ("simulation", simulation_loss),
                ("causal_pair", causal_pair),
                ("residual_l2", residual_l2),
            ):
                totals[name] += float(value.detach().cpu())
        history.append(
            {"epoch": epoch + 1, **{name: value / steps for name, value in totals.items()}}
        )
    return {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "input_channels": len(channels),
        "channel_indices": channels,
        "dropout": float(spec["dropout"]),
        "positive_weight": positive_weight,
        "history": history,
    }


@torch.inference_mode()
def predict_residual(
    fitted: dict[str, Any],
    images: np.ndarray,
    image_rows: np.ndarray,
    sensors: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = new_residual_model(
        int(fitted["input_channels"]), float(fitted["dropout"])
    ).to(device)
    model.load_state_dict(fitted["state_dict"])
    model.eval()
    channels = tuple(int(value) for value in fitted["channel_indices"])
    parts: list[np.ndarray] = []
    for start in range(0, image_rows.size, 256):
        rows = slice(start, start + 256)
        values = torch.from_numpy(
            np.asarray(images[image_rows[rows]][:, channels], dtype=np.float32)
        ).to(device)
        sensor = torch.from_numpy(sensors[rows].astype(np.int64)).to(device)
        parts.append(model(values, sensor).cpu().numpy())
    result = np.concatenate(parts).astype(np.float64)
    if result.shape != (image_rows.size,) or not np.isfinite(result).all():
        raise RuntimeError("Causal residual ranker produced invalid corrections")
    return result


def candidate_scores(current: np.ndarray, residual: np.ndarray, strength: float) -> np.ndarray:
    if not 0.0 < strength <= 1.0:
        raise ValueError("Residual strength must be in (0,1]")
    return expit(safe_logit(current) + strength * residual)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report.get("selected")
    lines = [
        "# Baseline-preserving causal residual scene ranker",
        "",
        "The zero-initialized model learns a correction around the frozen spatial-Prithvi logit and uses fit-fold-only same-background interventions.",
        "",
    ]
    if selected is None:
        lines.append("No residual strength passed every preregistered gate.")
    else:
        lines.extend(
            [
                f"- Residual strength: {selected['strength']:.3f}",
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
        raise ValueError("Frozen causal residual trainer hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen residual dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen causal residual input mismatch: {name}")
        paths[name] = path

    real_images = np.load(paths["real_images"], mmap_mode="r", allow_pickle=False)
    simulation_images = np.load(paths["simulation_images"], mmap_mode="r", allow_pickle=False)
    values = current_spatial_prithvi_scores(
        paths, float(protocol["current_comparator"]["prithvi_weight"])
    )
    real_image_rows, values = align_images(values, paths["real_metadata"])
    simulation = load_simulation_metadata(paths["simulation_metadata"])
    background_rows = align_simulation_backgrounds(values, real_image_rows, simulation)
    if simulation_images.shape[1:] != real_images.shape[1:]:
        raise ValueError("Real and simulated residual inputs differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Causal residual training requires CUDA")
    torch.set_float32_matmul_precision("high")
    spec = dict(protocol["model"])
    if args.smoke:
        smoke_spec = dict(spec)
        smoke_spec["epochs"] = 1
        smoke_spec["actual_batch_size"] = 8
        smoke_spec["simulation_batch_size"] = 4
        actual = np.arange(16, dtype=np.int64)
        synthetic = np.flatnonzero(simulation["fit_folds"] == 4)[:8]
        fitted = train_residual_model(
            smoke_spec,
            real_images,
            real_image_rows[actual],
            values["labels"][actual],
            values["sensors"][actual],
            values["current"][actual],
            np.ones(actual.size, dtype=np.float64),
            simulation_images,
            synthetic,
            simulation["sensors"],
            background_rows,
            seed=int(protocol["seeds"][0]),
            device=device,
        )
        residual = predict_residual(
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
                    "finite_residual": bool(np.isfinite(residual).all()),
                    "initialization": "exact zero output before optimization",
                    "history": fitted["history"],
                },
                indent=2,
            )
        )
        return 0

    raw_residual = np.empty(values["labels"].shape, dtype=np.float64)
    histories: dict[str, list[list[dict[str, float]]]] = {}
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
        seed_residuals: list[np.ndarray] = []
        histories[str(holdout)] = []
        for seed in protocol["seeds"]:
            fitted = train_residual_model(
                spec,
                real_images,
                real_image_rows[fit_rows],
                values["labels"][fit_rows],
                values["sensors"][fit_rows],
                values["current"][fit_rows],
                real_weights,
                simulation_images,
                simulation_rows,
                simulation["sensors"],
                background_rows,
                seed=int(seed) + holdout,
                device=device,
            )
            histories[str(holdout)].append(fitted["history"])
            seed_residuals.append(
                predict_residual(
                    fitted,
                    real_images,
                    real_image_rows[held_rows],
                    values["sensors"][held_rows],
                    device,
                )
            )
        raw_residual[held_rows] = np.mean(np.stack(seed_residuals), axis=0)
        print(
            json.dumps(
                {
                    "holdout": holdout,
                    "fit_fold": fit_fold,
                    "residual_mean": float(raw_residual[held_rows].mean()),
                    "residual_std": float(raw_residual[held_rows].std()),
                }
            ),
            flush=True,
        )

    all_rows = np.ones(values["labels"].shape, dtype=bool)
    rare_rows = low_prevalence_mask(
        values["labels"], values["groups"], float(protocol["target_view"]["maximum_site_positive_rate"])
    )
    candidates: list[dict[str, Any]] = []
    for index, strength in enumerate(protocol["strengths"]):
        scores = candidate_scores(values["current"], raw_residual, float(strength))
        whole = evaluate_view(values, scores, all_rows)
        rare = evaluate_view(values, scores, rare_rows)
        whole["bootstrap"] = ap_group_bootstrap(
            values["labels"], values["current"], scores, values["groups"],
            replicates=int(protocol["bootstrap"]["replicates"]),
            seed=int(protocol["bootstrap"]["whole_seed"]) + index,
        )
        rare["bootstrap"] = ap_group_bootstrap(
            values["labels"][rare_rows], values["current"][rare_rows], scores[rare_rows], values["groups"][rare_rows],
            replicates=int(protocol["bootstrap"]["replicates"]),
            seed=int(protocol["bootstrap"]["rare_seed"]) + index,
        )
        checks = gates(whole, rare, protocol["gates"])
        candidate = {
            "strength": float(strength),
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
            -float(strength),
        ]
        candidates.append(candidate)
    winner = max(candidates, key=lambda value: tuple(value["rank"]))
    selected = winner if winner["passed"] else None
    artifact_record = None
    if selected is not None:
        real_weights = sample_weights(
            str(spec["weighting"]), values["groups"], values["labels"], values["sensors"]
        )
        all_simulation = np.arange(simulation["labels"].size, dtype=np.int64)
        endpoints = [
            train_residual_model(
                spec,
                real_images,
                real_image_rows,
                values["labels"],
                values["sensors"],
                values["current"],
                real_weights,
                simulation_images,
                all_simulation,
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
                "kind": "mars_causal_residual_ranker_folds34",
                "fit_folds": [3, 4],
                "forbidden_folds": [0, 1, 2],
                "spec": spec,
                "strength": selected["strength"],
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
        residual=raw_residual, protocol_sha256=np.asarray(sha256(protocol_path)),
    )
    report = {
        "schema_version": 1,
        "scope": "folds-3/4 baseline-preserving causal residual cross-fit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_comparator": protocol["current_comparator"],
        "model": spec,
        "histories": histories,
        "candidates": candidates,
        "selected": selected,
        "artifact": artifact_record,
        "all_promotion_gates_pass": selected is not None,
        "decision": (
            "Freeze the causal residual ranker for separately preregistered fold-2 confirmation."
            if selected is not None
            else "Reject baseline-preserving causal residual ranking before fold-0/1/2 or external access."
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
                    "strength": selected["strength"],
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
