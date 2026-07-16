#!/usr/bin/env python3
"""Pilot end-to-end MethaneS2CM v5.1 transfer on one inner MARS site fold."""

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
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from extract_mars_methanes2cm_context_features import mars_to_v5_input  # noqa: E402
from mars_paper_model import SENSOR_NAMES  # noqa: E402
from methanes2cm_v5_model import MethaneS2CMV5Model  # noqa: E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    sampling_weights,
    verify_acquisition_receipt,
)
from train_mars_scene_ranker import blend_scores, comparison, metric_summary  # noqa: E402
from train_mars_spatial_scene_classifier import (  # noqa: E402
    DEFAULT_FOLD0_CACHE,
    DEFAULT_FOLD0_SHA256,
    DEFAULT_FOLD1_CACHE,
    DEFAULT_FOLD1_SHA256,
    DEFAULT_IMAGES,
    DEFAULT_IMAGES_SHA256,
    DEFAULT_INNER_CACHE,
    DEFAULT_INNER_SHA256,
    DEFAULT_METADATA,
    DEFAULT_METADATA_SHA256,
    DEFAULT_SCORE_CACHE,
    DEFAULT_SCORE_SHA256,
    load_partitions,
)
from train_methanes2cm_v5 import segmentation_first_loss  # noqa: E402

DEFAULT_SOURCE_CHECKPOINT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/methanes2cm_v5_1_seed1101.pt"
)
DEFAULT_SOURCE_CHECKPOINT_SHA256 = (
    "7b648548cc62ca3f6d428df2cf427e373fba5a7bdcf03aabada68bf6f1cfc446"
)
DEFAULT_SOURCE_REPORT = Path(
    "reports/experiments/methanes2cm_v5_1_seed1101_validation.json"
)
DEFAULT_SOURCE_REPORT_SHA256 = (
    "a4a830619d217859c586b78d23d06ca6c54a91800a7d6497e1ed70ff9628997c"
)
DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_v5_transfer_pilot.pt"
)
DEFAULT_JSON = Path("reports/experiments/mars_v5_transfer_pilot.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_V5_TRANSFER_PILOT.md")
FIT_FOLDS = (3, 4)
HELD_FOLD = 2
INPUT_SIZE = 64
EPOCHS = 3
SEED = 20262200
BLENDS = (0.05, 0.1, 0.2, 0.3)


def resize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    values = dict(batch)
    values["inputs"] = F.interpolate(
        mars_to_v5_input(batch["inputs"]),
        size=(INPUT_SIZE, INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    values["observable"] = F.interpolate(
        batch["observable"].float(), size=(INPUT_SIZE, INPUT_SIZE), mode="nearest"
    )
    values["mask"] = F.interpolate(
        batch["mask"].float(), size=(INPUT_SIZE, INPUT_SIZE), mode="nearest"
    )
    return values


def load_source_model(checkpoint: Path, report_path: Path, device: torch.device) -> MethaneS2CMV5Model:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["model"].get("context_scene_weight") != 0.65:
        raise ValueError("Expected the frozen v5.1 context model")
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload["model_metadata"] != report["model"]:
        raise ValueError("V5.1 source checkpoint metadata differs from its report")
    model = MethaneS2CMV5Model(context_scene_weight=0.65).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model


def train_model(
    model: MethaneS2CMV5Model,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    history = []
    for epoch in range(EPOCHS):
        model.train()
        losses: list[float] = []
        scene_losses: list[float] = []
        for batch in loader:
            batch = resize_batch(move_batch(batch, device))
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch["inputs"], batch["observable"])
                loss, parts = segmentation_first_loss(output, batch, scene_weight=0.5)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            scene_losses.append(float(parts["scene_bce"]))
        row = {
            "epoch": epoch + 1,
            "mean_loss": float(np.mean(losses)),
            "mean_scene_bce": float(np.mean(scene_losses)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


@torch.no_grad()
def predict(
    model: MethaneS2CMV5Model,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sensors: list[np.ndarray] = []
    groups: list[str] = []
    sample_ids: list[str] = []
    for batch in loader:
        batch = resize_batch(move_batch(batch, device))
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(batch["inputs"], batch["observable"])
        scores.append(torch.sigmoid(output["scene_logit"]).float().cpu().numpy())
        labels.append(batch["presence"].cpu().numpy())
        sensors.append(batch["sensor_index"].cpu().numpy())
        groups.extend(str(value) for value in batch["group_id"])
        sample_ids.extend(str(value) for value in batch["sample_id"])
    return (
        np.concatenate(scores).astype(np.float64),
        np.concatenate(labels).astype(np.uint8),
        np.concatenate(sensors).astype(np.uint8),
        np.asarray(groups),
        np.asarray(sample_ids),
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=Path("EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L").as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--acquisition-receipt", default=DEFAULT_ACQUISITION_RECEIPT.as_posix())
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE_CHECKPOINT.as_posix())
    parser.add_argument("--source-checkpoint-sha256", default=DEFAULT_SOURCE_CHECKPOINT_SHA256)
    parser.add_argument("--source-report", default=DEFAULT_SOURCE_REPORT.as_posix())
    parser.add_argument("--source-report-sha256", default=DEFAULT_SOURCE_REPORT_SHA256)
    parser.add_argument("--images", default=DEFAULT_IMAGES.as_posix())
    parser.add_argument("--images-sha256", default=DEFAULT_IMAGES_SHA256)
    parser.add_argument("--metadata", default=DEFAULT_METADATA.as_posix())
    parser.add_argument("--metadata-sha256", default=DEFAULT_METADATA_SHA256)
    parser.add_argument("--score-cache", default=DEFAULT_SCORE_CACHE.as_posix())
    parser.add_argument("--score-sha256", default=DEFAULT_SCORE_SHA256)
    parser.add_argument("--inner-cache", default=DEFAULT_INNER_CACHE.as_posix())
    parser.add_argument("--inner-sha256", default=DEFAULT_INNER_SHA256)
    parser.add_argument("--fold0-cache", default=DEFAULT_FOLD0_CACHE.as_posix())
    parser.add_argument("--fold0-sha256", default=DEFAULT_FOLD0_SHA256)
    parser.add_argument("--fold1-cache", default=DEFAULT_FOLD1_CACHE.as_posix())
    parser.add_argument("--fold1-sha256", default=DEFAULT_FOLD1_SHA256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")
    paths = {
        "source_checkpoint": (root / args.source_checkpoint).resolve(),
        "source_report": (root / args.source_report).resolve(),
        "images": (root / args.images).resolve(),
        "metadata": (root / args.metadata).resolve(),
        "score": (root / args.score_cache).resolve(),
        "inner": (root / args.inner_cache).resolve(),
        "fold0": (root / args.fold0_cache).resolve(),
        "fold1": (root / args.fold1_cache).resolve(),
    }
    expected = {
        "source_checkpoint": args.source_checkpoint_sha256,
        "source_report": args.source_report_sha256,
        "images": args.images_sha256,
        "metadata": args.metadata_sha256,
        "score": args.score_sha256,
        "inner": args.inner_sha256,
        "fold0": args.fold0_sha256,
        "fold1": args.fold1_sha256,
    }
    for name, digest in expected.items():
        if sha256(paths[name]) != digest:
            raise ValueError(f"Frozen {name} hash mismatch")

    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen protocol")
    verify_acquisition_receipt((root / args.acquisition_receipt).resolve(), manifest_hash)
    group_to_fold = {
        str(item["group_id"]): int(item["fold"]) for item in protocol["assignments"]
    }
    records = list(iter_development_manifest(manifest))
    fit_records = [
        record for record in records if group_to_fold[str(record["group_id"])] in FIT_FOLDS
    ]
    held_records = [
        record for record in records if group_to_fold[str(record["group_id"])] == HELD_FOLD
    ]
    metadata_dir = (root / args.metadata_dir).resolve()
    train_dataset = MarsPaperDataset(metadata_dir, fit_records, augment=True, seed=SEED)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            sampling_weights(fit_records),
            num_samples=len(fit_records),
            replacement=True,
            generator=generator,
        ),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    held_loader = DataLoader(
        MarsPaperDataset(metadata_dir, held_records, augment=False, seed=0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_source_model(paths["source_checkpoint"], paths["source_report"], device)
    history = train_model(model, train_loader, device)
    raw, labels, sensors, groups, sample_ids = predict(model, held_loader, device)

    parts = load_partitions(
        paths["metadata"], paths["score"], {name: paths[name] for name in ("inner", "fold0", "fold1")}
    )
    inner = parts["inner"]
    rows = inner["folds"] == HELD_FOLD
    expected_ids = np.load(paths["inner"], allow_pickle=False)["sample_ids"].astype(str)[rows]
    if not np.array_equal(sample_ids, expected_ids):
        raise ValueError("Held transfer predictions do not align with fold-2 scores")
    values = {
        "labels": labels,
        "sensors": sensors,
        "groups": groups,
        "current": inner["new"][rows],
        "primary": inner["primary"][rows],
    }
    candidates = []
    for blend in BLENDS:
        scores = blend_scores(values["current"], raw, blend)
        candidate = metric_summary(labels, scores, sensors)
        versus_current = comparison(
            candidate, metric_summary(labels, values["current"], sensors)
        )
        versus_primary = comparison(
            candidate, metric_summary(labels, values["primary"], sensors)
        )
        candidates.append(
            {
                "blend_weight": blend,
                "versus_current": versus_current,
                "versus_primary": versus_primary,
                "rank": [
                    int(
                        versus_current["delta"]["average_precision"] >= 0.003
                        and versus_current["delta"]["recall_at_fpr_0_0713"] >= 0
                        and min(versus_current["delta"]["sensor_average_precision"].values()) >= 0
                    ),
                    versus_current["delta"]["recall_at_fpr_0_0713"],
                    versus_current["delta"]["average_precision"],
                    -blend,
                ],
            }
        )
    selected = max(candidates, key=lambda value: tuple(value["rank"]))
    scores = blend_scores(values["current"], raw, selected["blend_weight"])
    selected["paired_group_bootstrap_ap_delta_vs_current"] = ap_group_bootstrap(
        labels, values["current"], scores, groups, replicates=10_000, seed=20262280
    )
    passed = bool(
        selected["rank"][0] == 1
        and selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"] > 0
    )
    artifact = (root / args.artifact).resolve()
    artifact_hash = None
    if passed:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_suffix(artifact.suffix + ".tmp")
        torch.save(
            {
                "schema_version": 1,
                "kind": "mars_v5_transfer_pilot",
                "state_dict": model.state_dict(),
                "blend_weight": selected["blend_weight"],
                "fit_folds": FIT_FOLDS,
                "held_fold": HELD_FOLD,
            },
            temporary,
        )
        os.replace(temporary, artifact)
        artifact_hash = sha256(artifact)
    report = {
        "schema_version": 1,
        "scope": "MethaneS2CM-v5.1 end-to-end MARS transfer pilot; fit folds 3/4, held fold 2; paper cache not loaded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "fit_folds": list(FIT_FOLDS),
            "held_fold": HELD_FOLD,
            "input_size": INPUT_SIZE,
            "epochs": EPOCHS,
            "seed": SEED,
            "blends": list(BLENDS),
        },
        "history": history,
        "selected": selected,
        "pilot_gate_pass": passed,
        "decision": (
            "Authorize a full cross-fitted v5.1 transfer campaign."
            if passed
            else "Reject end-to-end v5.1 transfer before a full campaign."
        ),
        "provenance": {
            **{f"{name}_sha256": digest for name, digest in expected.items()},
            "artifact_sha256": artifact_hash,
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    output_json = (root / args.output_json).resolve()
    write_json(output_json, report)
    delta = selected["versus_current"]["delta"]
    interval = selected["paired_group_bootstrap_ap_delta_vs_current"]
    markdown = (root / args.output_markdown).resolve()
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(
        "\n".join(
            [
                "# MethaneS2CM v5.1 to MARS transfer pilot",
                "",
                f"- Blend: {selected['blend_weight']:.2f}",
                f"- Fold-2 AP delta vs current: {delta['average_precision']:+.5f}",
                f"- Fold-2 recall delta: {delta['recall_at_fpr_0_0713']:+.5f}",
                f"- Paired-site AP interval: [{interval['lower']:+.5f}, {interval['upper']:+.5f}]",
                "",
                report["decision"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": passed,
                "blend": selected["blend_weight"],
                "ap_delta": delta["average_precision"],
                "recall_delta": delta["recall_at_fpr_0_0713"],
                "ap_lower": interval["lower"],
                "artifact_sha256": artifact_hash,
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
