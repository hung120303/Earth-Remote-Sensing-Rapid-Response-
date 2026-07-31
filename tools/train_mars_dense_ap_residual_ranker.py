#!/usr/bin/env python3
"""Train an AP-focused residual scene ranker on frozen dense-Prithvi features."""

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
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from mars_paper_model import SENSOR_NAMES  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import comparison, metric_summary  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_dense_ap_residual_ranker_protocol.json")


class DenseAPResidualRanker(nn.Module):
    """Bounded logit residual whose initialization is the current score."""

    def __init__(self, input_features: int, hidden: tuple[int, int], dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden[1], 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.tanh(self.output(self.network(values)).squeeze(1))


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def fit_robust_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(values, dtype=np.float32).copy()
    local[np.abs(local) >= 1000] = np.nan
    center = np.nanmedian(local, axis=0).astype(np.float32)
    lower = np.nanpercentile(local, 25, axis=0).astype(np.float32)
    upper = np.nanpercentile(local, 75, axis=0).astype(np.float32)
    scale = ((upper - lower) / 1.349).astype(np.float32)
    invalid_center = ~np.isfinite(center)
    center[invalid_center] = 0.0
    scale[invalid_center | ~np.isfinite(scale) | (scale < 1e-4)] = 1.0
    return center, scale


def transform(
    values: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    local = np.asarray(values, dtype=np.float32).copy()
    missing = ~np.isfinite(local) | (np.abs(local) >= 1000)
    if missing.any():
        rows, columns = np.nonzero(missing)
        local[rows, columns] = center[columns]
    return np.clip((local - center) / scale, -8.0, 8.0).astype(np.float32)


def smooth_ap_loss(scores: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    positive = scores[labels > 0.5]
    if not positive.numel() or positive.numel() == scores.numel():
        return scores.sum() * 0.0
    all_higher = torch.sigmoid((scores[None, :] - positive[:, None]) / tau)
    positive_higher = torch.sigmoid(
        (positive[None, :] - positive[:, None]) / tau
    )
    rank = 0.5 + all_higher.sum(dim=1)
    positive_rank = 0.5 + positive_higher.sum(dim=1)
    return 1.0 - (positive_rank / rank.clamp_min(1.0)).mean()


def hard_pair_loss(
    scores: torch.Tensor,
    base_scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    positive = scores[labels > 0.5]
    negative = scores[labels < 0.5]
    base_positive = base_scores[labels > 0.5]
    base_negative = base_scores[labels < 0.5]
    if not positive.numel() or not negative.numel():
        return scores.sum() * 0.0
    difficulty = 0.25 + torch.sigmoid(
        base_negative[None, :] - base_positive[:, None]
    )
    losses = F.softplus(margin - positive[:, None] + negative[None, :])
    return (losses * difficulty).sum() / difficulty.sum().clamp_min(1e-6)


def fit_model(
    raw_features: np.ndarray,
    labels: np.ndarray,
    base_scores: np.ndarray,
    rows: np.ndarray,
    spec: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, dict[str, float]]:
    center, scale = fit_robust_scaler(raw_features[rows])
    values = torch.from_numpy(transform(raw_features[rows], center, scale)).to(device)
    target = torch.from_numpy(labels[rows].astype(np.float32)).to(device)
    base_logit = torch.from_numpy(logit(base_scores[rows]).astype(np.float32)).to(device)
    positive_indices = torch.nonzero(target > 0.5, as_tuple=False).flatten()
    negative_indices = torch.nonzero(target < 0.5, as_tuple=False).flatten()
    if not positive_indices.numel() or not negative_indices.numel():
        raise ValueError("AP residual training requires both labels")
    seed_everything(seed)
    model = DenseAPResidualRanker(
        values.shape[1],
        tuple(map(int, spec["hidden"])),
        float(spec["dropout"]),
    ).to(device)
    with torch.no_grad():
        if float(model(values[: min(32, len(values))]).abs().max()) != 0.0:
            raise ValueError("AP residual ranker is not exact identity at initialization")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    sums: dict[str, float] = {}
    model.train()
    for _step in range(int(spec["steps"])):
        positive_rows = positive_indices[
            torch.randint(
                positive_indices.numel(),
                (int(spec["positive_per_batch"]),),
                generator=generator,
                device=device,
            )
        ]
        negative_rows = negative_indices[
            torch.randint(
                negative_indices.numel(),
                (int(spec["negative_per_batch"]),),
                generator=generator,
                device=device,
            )
        ]
        batch_rows = torch.cat((positive_rows, negative_rows))
        batch_values = values[batch_rows]
        batch_target = target[batch_rows]
        batch_base = base_logit[batch_rows]
        optimizer.zero_grad(set_to_none=True)
        residual = model(batch_values)
        candidate = batch_base + residual
        ap = smooth_ap_loss(candidate, batch_target, float(spec["smooth_ap_tau"]))
        pair = hard_pair_loss(
            candidate,
            batch_base,
            batch_target,
            float(spec["pair_margin"]),
        )
        bce = F.binary_cross_entropy_with_logits(candidate, batch_target)
        residual_l2 = residual.square().mean()
        loss = (
            float(spec["smooth_ap_weight"]) * ap
            + float(spec["pair_weight"]) * pair
            + float(spec["bce_weight"]) * bce
            + float(spec["residual_l2_weight"]) * residual_l2
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(spec["gradient_clip"])
        )
        optimizer.step()
        for name, value in (
            ("loss", loss),
            ("smooth_ap", ap),
            ("pair", pair),
            ("bce", bce),
            ("residual_l2", residual_l2),
        ):
            sums[name] = sums.get(name, 0.0) + float(value.detach())
    history = {
        name: value / int(spec["steps"]) for name, value in sums.items()
    }
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return state, center, scale, history


@torch.no_grad()
def predict_model(
    raw_features: np.ndarray,
    rows: np.ndarray,
    state: dict[str, torch.Tensor],
    center: np.ndarray,
    scale: np.ndarray,
    spec: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    model = DenseAPResidualRanker(
        raw_features.shape[1],
        tuple(map(int, spec["hidden"])),
        float(spec["dropout"]),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    parts: list[np.ndarray] = []
    batch_size = int(spec["prediction_batch_size"])
    for start in range(0, rows.size, batch_size):
        local = rows[start : start + batch_size]
        values = torch.from_numpy(transform(raw_features[local], center, scale)).to(
            device
        )
        parts.append(model(values).cpu().numpy().astype(np.float64))
    return np.concatenate(parts)


def fold_ap_deltas(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    folds: np.ndarray,
) -> dict[str, float]:
    return {
        str(int(fold)): float(
            average_precision_score(labels[folds == fold], candidate[folds == fold])
            - average_precision_score(labels[folds == fold], baseline[folds == fold])
        )
        for fold in np.unique(folds)
    }


def evaluate_strength(
    labels: np.ndarray,
    sensors: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    base_scores: np.ndarray,
    residual: np.ndarray,
    strength: float,
    bootstrap: dict[str, Any],
    seed_offset: int,
    *,
    internal: bool,
) -> dict[str, Any]:
    scores = sigmoid(logit(base_scores) + float(strength) * residual)
    baseline_metrics = metric_summary(labels, base_scores, sensors)
    metrics = metric_summary(labels, scores, sensors)
    versus = comparison(metrics, baseline_metrics)
    interval = ap_group_bootstrap(
        labels,
        base_scores,
        scores,
        groups,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]) + seed_offset,
    )
    folds_delta = fold_ap_deltas(labels, base_scores, scores, folds)
    sensor_delta = versus["delta"]["sensor_average_precision"]
    passed = bool(
        versus["delta"]["average_precision"] >= float(
            bootstrap["minimum_ap_delta"]
        )
        and versus["delta"]["recall_at_fpr_0_0713"] >= 0.0
        and min(sensor_delta.values()) >= 0.0
        and min(folds_delta.values()) >= 0.0
        and interval["lower"] > 0.0
    )
    return {
        "strength": strength,
        "metrics": metrics,
        "versus_current": versus,
        "fold_ap_deltas": folds_delta,
        "paired_site_ap_delta": interval,
        "passed": passed,
        "rank": [
            int(passed),
            min(folds_delta.values()),
            min(sensor_delta.values()),
            interval["lower"],
            versus["delta"]["average_precision"],
            versus["delta"]["recall_at_fpr_0_0713"],
            -strength,
        ],
        "scope": "internal representation-fit folds" if internal else "held architecture fold",
    }


def train_ensemble(
    raw_features: np.ndarray,
    labels: np.ndarray,
    base_scores: np.ndarray,
    fit_rows: np.ndarray,
    predict_rows: np.ndarray,
    spec: dict[str, Any],
    seeds: list[int],
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = []
    artifacts = []
    histories = []
    for seed in seeds:
        state, center, scale, history = fit_model(
            raw_features,
            labels,
            base_scores,
            fit_rows,
            spec,
            seed,
            device,
        )
        predictions.append(
            predict_model(
                raw_features,
                predict_rows,
                state,
                center,
                scale,
                spec,
                device,
            )
        )
        artifacts.append(
            {
                "seed": seed,
                "state": state,
                "center": center,
                "scale": scale,
            }
        )
        histories.append({"seed": seed, **history})
    return np.mean(predictions, axis=0), artifacts, histories


def load_cache(
    feature_path: Path, metadata_path: Path
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    matrix = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    with np.load(metadata_path, allow_pickle=False) as values:
        metadata = {name: values[name] for name in values.files}
    if str(metadata["features_sha256"].item()) != sha256(feature_path):
        raise ValueError("Dense scene feature cache differs from its metadata")
    if matrix.shape != (metadata["sample_ids"].size, metadata["feature_names"].size):
        raise ValueError("Dense scene feature and metadata geometry differ")
    # The end-to-end BCE residual was rejected.  Keep the exact current score
    # (column 0) and every representation feature except rejected column 1.
    columns = np.concatenate((np.asarray([0]), np.arange(2, matrix.shape[1])))
    return matrix, metadata, columns


def load_exact_base_scores(
    score_path: Path, metadata: dict[str, np.ndarray]
) -> np.ndarray:
    scores = np.full(metadata["sample_ids"].size, np.nan, dtype=np.float64)
    with np.load(score_path, allow_pickle=False) as values:
        for prefix, rows in (
            ("fold0", metadata["folds"] == 0),
            ("fold1", metadata["folds"] == 1),
            ("inner", metadata["folds"] >= 2),
        ):
            if not np.array_equal(metadata["labels"][rows], values[f"{prefix}_labels"]):
                raise ValueError(f"{prefix} exact-score labels do not align")
            if not np.array_equal(metadata["sensors"][rows], values[f"{prefix}_sensors"]):
                raise ValueError(f"{prefix} exact-score sensors do not align")
            if not np.array_equal(
                metadata["groups"][rows].astype(str),
                values[f"{prefix}_groups"].astype(str),
            ):
                raise ValueError(f"{prefix} exact-score sites do not align")
            if prefix == "inner" and not np.array_equal(
                metadata["folds"][rows], values["inner_folds"]
            ):
                raise ValueError("Inner exact-score folds do not align")
            scores[rows] = values[f"{prefix}_new"].astype(np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Exact current scene scores are incomplete or outside [0,1]")
    return scores


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["held"]["selected"]
    delta = selected["versus_current"]["delta"]
    interval = selected["paired_site_ap_delta"]
    lines = [
        "# Dense AP residual scene ranker",
        "",
        "Development-only evaluation. The current cross-fitted scene score is the exact "
        "zero-residual floor.",
        "",
        f"- Selected residual strength: {selected['strength']:.3f}",
        f"- Held-fold AP delta: {delta['average_precision']:+.6f}",
        f"- Held-fold matched-FPR recall delta: {delta['recall_at_fpr_0_0713']:+.6f}",
        f"- Paired-site AP interval: [{interval['lower']:+.6f}, {interval['upper']:+.6f}]",
        "",
        report["decision"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--internal-only", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen dense AP ranker trainer hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen input mismatch: {name}")
        paths[name] = path
    matrix, metadata, columns = load_cache(paths["features"], paths["metadata"])
    raw_features = matrix[:, columns]
    labels = metadata["labels"].astype(np.uint8)
    sensors = metadata["sensors"].astype(np.uint8)
    groups = metadata["groups"].astype(str)
    folds = metadata["folds"].astype(np.uint8)
    base_scores = load_exact_base_scores(paths["score_cache"], metadata)
    if not np.allclose(
        base_scores,
        np.asarray(raw_features[:, 0], dtype=np.float64),
        atol=2.5e-4,
        rtol=5e-4,
    ):
        raise ValueError("Float16 current-score feature differs unexpectedly from exact floor")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Dense AP residual ranker requires CUDA")
    spec = protocol["training"]

    internal_rows = np.flatnonzero(np.isin(folds, protocol["folds"]["fit"]))
    internal_residual = np.empty(internal_rows.size, dtype=np.float64)
    internal_position = {int(row): index for index, row in enumerate(internal_rows)}
    internal_histories: dict[str, Any] = {}
    for held in sorted(map(int, protocol["folds"]["fit"])):
        fit_rows = np.flatnonzero(
            np.isin(folds, protocol["folds"]["fit"]) & (folds != held)
        )
        predict_rows = np.flatnonzero(folds == held)
        local, _artifacts, histories = train_ensemble(
            raw_features,
            labels,
            base_scores,
            fit_rows,
            predict_rows,
            spec,
            [
                int(seed) + held * 100
                for seed in protocol["internal_seeds"]
            ],
            device,
        )
        for row, value in zip(predict_rows, local):
            internal_residual[internal_position[int(row)]] = value
        internal_histories[str(held)] = histories
    if not np.isfinite(internal_residual).all():
        raise RuntimeError("Internal cross-fold residuals are incomplete")
    strengths = [float(value) for value in protocol["search"]["strengths"]]
    internal_candidates = [
        evaluate_strength(
            labels[internal_rows],
            sensors[internal_rows],
            groups[internal_rows],
            folds[internal_rows],
            base_scores[internal_rows],
            internal_residual,
            strength,
            protocol["bootstrap"],
            index,
            internal=True,
        )
        for index, strength in enumerate(strengths)
    ]
    internal_selected = max(
        internal_candidates, key=lambda row: tuple(row["rank"])
    )
    internal_report = {
        "rows": int(internal_rows.size),
        "folds": sorted(map(int, protocol["folds"]["fit"])),
        "histories": internal_histories,
        "candidates": internal_candidates,
        "selected": internal_selected,
    }
    if args.internal_only:
        print(json.dumps({"ok": internal_selected["passed"], "internal": internal_report}, indent=2))
        return 0 if internal_selected["passed"] else 2

    frozen_selection = protocol.get("frozen_internal_selection")
    if frozen_selection is None:
        raise ValueError("Held-fold evaluation requires a frozen internal selection")
    if (
        not internal_selected["passed"]
        or float(frozen_selection["strength"]) != float(internal_selected["strength"])
    ):
        raise ValueError("Recomputed internal selection differs from the frozen protocol")

    held_fold = int(protocol["folds"]["held"])
    held_rows = np.flatnonzero(folds == held_fold)
    final_residual, final_artifacts, final_histories = train_ensemble(
        raw_features,
        labels,
        base_scores,
        internal_rows,
        held_rows,
        spec,
        [int(seed) for seed in protocol["final_seeds"]],
        device,
    )
    held_selected = evaluate_strength(
        labels[held_rows],
        sensors[held_rows],
        groups[held_rows],
        folds[held_rows],
        base_scores[held_rows],
        final_residual,
        float(frozen_selection["strength"]),
        protocol["bootstrap"],
        100,
        internal=False,
    )
    passed = bool(held_selected["passed"])
    artifact = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        if artifact_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "kind": "mars_dense_ap_residual_ranker",
                "models": final_artifacts,
                "feature_columns": columns,
                "feature_names": metadata["feature_names"][columns],
                "residual_strength": float(frozen_selection["strength"]),
                "protocol_sha256": sha256(protocol_path),
            },
            artifact_path,
        )
        artifact = {
            "path": artifact_path.relative_to(ROOT).as_posix(),
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
        }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "scope": "AP-focused residual ranker on frozen dense-Prithvi development features",
        "all_promotion_gates_pass": passed,
        "decision": (
            "Authorize independent full-development cross-fitting of the dense AP ranker."
            if passed
            else "Reject the dense AP residual before full cross-fit or external scoring."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "trainer_sha256": sha256(Path(__file__).resolve()),
            "features_sha256": sha256(paths["features"]),
            "metadata_sha256": sha256(paths["metadata"]),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "input_features": int(columns.size),
        "excluded_feature": str(metadata["feature_names"][1]),
        "internal": internal_report,
        "held": {
            "fold": held_fold,
            "rows": int(held_rows.size),
            "histories": final_histories,
            "selected": held_selected,
        },
        "artifact": artifact,
        "external_inputs_accessed": False,
    }
    output_json = (ROOT / protocol["outputs"]["json"]).resolve()
    write_json(output_json, report)
    write_markdown((ROOT / protocol["outputs"]["markdown"]).resolve(), report)
    print(
        json.dumps(
            {
                "ok": passed,
                "strength": held_selected["strength"],
                "ap_delta": held_selected["versus_current"]["delta"][
                    "average_precision"
                ],
                "ap_lower": held_selected["paired_site_ap_delta"]["lower"],
                "recall_delta": held_selected["versus_current"]["delta"][
                    "recall_at_fpr_0_0713"
                ],
                "sensor_deltas": held_selected["versus_current"]["delta"][
                    "sensor_average_precision"
                ],
                "artifact": artifact,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
