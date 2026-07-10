#!/usr/bin/env python3
"""Train ERSRR MARS joint model v2 with top-k multiple-instance presence pooling."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from mars_joint_mil_model import MarsJointMILModel  # noqa: E402
from mars_s2l_adapter import iter_manifest  # noqa: E402

from acquire_mars_metadata import DEFAULT_OUTPUT, REVISION, checked_output_dir, repo_root, sha256  # noqa: E402
from build_mars_dev_cohort import DEV_SAMPLES, DEFAULT_JSON as DEV_REPORT_JSON  # noqa: E402
from run_mars_dev_pixel_baselines import evaluate_rule  # noqa: E402
from run_mars_dev_scene_baselines import (  # noqa: E402
    bootstrap_ci,
    choose_lower_threshold,
    choose_upper_threshold,
    metrics,
    role_weights,
)
from train_mars_joint_model import (  # noqa: E402
    QUALITY_OBSERVABLE_FRACTION,
    MarsJointDataset,
    choose_quality_threshold,
    collect_predictions,
    git_commit,
    masked_segmentation_loss,
    move_batch,
    quality_loss,
    safe_output,
    seed_everything,
    select_segmentation_by_dice,
    selective_quality_metrics,
    tracked_dirty,
    write_json,
)

DEFAULT_JSON = Path("reports/experiments/mars_joint_mil_development.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_JOINT_MIL_DEVELOPMENT.md")
DEFAULT_CHECKPOINT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_joint_mil_v2_seed101.pt"
)
PRESENCE_LOSS_WEIGHT = 2.0
QUALITY_LOSS_WEIGHT = 0.10
DEFAULT_SEED = 101


def total_loss_v2(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    segmentation, segmentation_bce, dice = masked_segmentation_loss(
        outputs["segmentation_logits"], batch["mask"], batch["observable"]
    )
    presence = F.binary_cross_entropy_with_logits(
        outputs["presence_logit"], batch["presence"]
    )
    quality = quality_loss(outputs["quality_logit"], batch["quality"])
    total = segmentation + PRESENCE_LOSS_WEIGHT * presence + QUALITY_LOSS_WEIGHT * quality
    return total, {
        "total": float(total.detach()),
        "segmentation_bce": float(segmentation_bce.detach()),
        "dice_loss": float(dice.detach()),
        "presence_bce": float(presence.detach()),
        "quality_bce": float(quality.detach()),
    }


def forward_batch(model: MarsJointMILModel, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return model(batch["target"], batch["reference"], batch["mbmp"], batch["observable"])


@torch.no_grad()
def validation_summary(
    model: MarsJointMILModel,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    labels: list[np.ndarray] = []
    presence: list[np.ndarray] = []
    quality_labels: list[np.ndarray] = []
    quality: list[np.ndarray] = []
    losses: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            outputs = forward_batch(model, batch)
            loss, _ = total_loss_v2(outputs, batch)
        losses.append(float(loss))
        labels.append(batch["presence"].cpu().numpy())
        presence.append(torch.sigmoid(outputs["presence_logit"]).float().cpu().numpy())
        quality_labels.append(batch["quality"].cpu().numpy())
        quality.append(torch.sigmoid(outputs["quality_logit"]).float().cpu().numpy())
    y = np.concatenate(labels).astype(np.uint8)
    p = np.concatenate(presence)
    qy = np.concatenate(quality_labels).astype(np.uint8)
    qp = np.concatenate(quality)
    threshold, operating = choose_upper_threshold(y, p)
    return {
        "loss": float(np.mean(losses)),
        "presence_average_precision": float(average_precision_score(y, p)),
        "presence_auroc": float(roc_auc_score(y, p)),
        "recall_at_fpr5": operating["recall"],
        "observed_fpr": operating["false_positive_rate"],
        "operating_threshold": threshold,
        "quality_auroc": float(roc_auc_score(qy, qp)),
    }


def train(
    model: MarsJointMILModel,
    train_loader: DataLoader[dict[str, Any]],
    val_loader: DataLoader[dict[str, Any]],
    device: torch.device,
    checkpoint: Path,
    *,
    epochs: int,
    learning_rate: float,
    patience: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_rank = (-math.inf, -math.inf)
    best_epoch = -1
    stale = 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[dict[str, float]] = []
        started = time.perf_counter()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                outputs = forward_batch(model, batch)
                loss, parts = total_loss_v2(outputs, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(parts)
        scheduler.step()
        validation = validation_summary(model, val_loader, device)
        record = {
            "epoch": epoch,
            "seconds": round(time.perf_counter() - started, 3),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training": {
                key: float(np.mean([item[key] for item in losses])) for key in losses[0]
            },
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        rank = (
            float(validation["recall_at_fpr5"] or 0.0),
            float(validation["presence_average_precision"]),
        )
        if rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            stale = 0
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_metadata": model.artifact_metadata(),
                    "seed": seed,
                    "epoch": epoch,
                    "validation": validation,
                },
                temporary,
            )
            os.replace(temporary, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    if best_epoch < 0:
        raise RuntimeError("MIL training produced no checkpoint")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return history, best_epoch


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    validation = report["validation"]["scene"]
    lines = [
        "# ERSRR MARS joint MIL v2 development result",
        "",
        "Validation-first architecture experiment; strict-spatial results appear only when the explicitly frozen test flag was used.",
        "",
        f"- Model: `{report['model']['model_name']}` / {report['model']['parameter_count']:,} parameters",
        f"- Best epoch: {report['training']['best_epoch']} / {len(report['training']['history'])}",
        f"- Validation recall at FPR <= 0.05: {validation['recall']:.3f} (observed FPR {validation['false_positive_rate']:.3f})",
        f"- Validation AUROC/AP: {validation['auroc']:.3f} / {validation['average_precision']:.3f}",
        f"- Validation mask Dice: {report['validation']['segmentation']['pixel_dice']:.3f}",
        f"- Checkpoint SHA-256: `{report['artifact']['sha256']}`",
    ]
    if "test" in report:
        test = report["test"]["scene_unweighted"]
        segmentation = report["test"]["segmentation"]["pixel"]
        lines.extend(
            [
                "",
                "## Frozen strict-spatial evaluation",
                "",
                f"- Scene recall/specificity/FPR: {test['recall']:.3f} / {test['specificity']:.3f} / {test['false_positive_rate']:.3f}",
                f"- Pixel AP/IoU/Dice: {segmentation['average_precision']:.4f} / {segmentation['intersection_over_union']:.4f} / {segmentation['dice']:.4f}",
            ]
        )
    lines.extend(["", "## Decision", "", report["decision"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the strict-spatial benchmark after validation rules are frozen",
    )
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the MIL development experiment")
        seed_everything(args.seed)
        metadata_dir = checked_output_dir(root, args.metadata_dir)
        manifest_path = metadata_dir / DEV_SAMPLES
        dev_report = json.loads((root / DEV_REPORT_JSON).read_text(encoding="utf-8"))
        if sha256(manifest_path) != dev_report["identities"]["sample_manifest_sha256"]:
            raise ValueError("Development manifest identity mismatch")
        records = list(iter_manifest(manifest_path))
        by_role = {
            role: [record for record in records if record["research_role"] == role]
            for role in ("internal_training", "internal_validation", "strict_spatial_test")
        }
        train_dataset = MarsJointDataset(
            metadata_dir, by_role["internal_training"], augment=True, seed=args.seed
        )
        val_dataset = MarsJointDataset(
            metadata_dir, by_role["internal_validation"], augment=False, seed=args.seed
        )
        test_dataset = MarsJointDataset(
            metadata_dir, by_role["strict_spatial_test"], augment=False, seed=args.seed
        )
        sample_weights = [
            2.0 if record["label_state"] == "PLUME" else 1.0
            for record in by_role["internal_training"]
        ]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        options = {
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "pin_memory": True,
            "persistent_workers": args.workers > 0,
        }
        train_loader = DataLoader(train_dataset, sampler=sampler, **options)
        val_loader = DataLoader(val_dataset, shuffle=False, **options)
        test_loader = DataLoader(test_dataset, shuffle=False, **options)
        device = torch.device("cuda")
        model = MarsJointMILModel().to(device)
        checkpoint = safe_output(root, args.checkpoint)
        output_json = safe_output(root, args.output_json)
        output_markdown = safe_output(root, args.output_markdown)
        if args.evaluate_only:
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(payload["state_dict"])
            best_epoch = int(payload["epoch"])
            history = (
                json.loads(output_json.read_text(encoding="utf-8"))["training"]["history"]
                if output_json.is_file()
                else []
            )
        else:
            history, best_epoch = train(
                model,
                train_loader,
                val_loader,
                device,
                checkpoint,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                patience=args.patience,
                seed=args.seed,
            )
        validation = collect_predictions(model, val_loader, device)
        upper, validation_scene = choose_upper_threshold(
            validation["labels"], validation["presence"]
        )
        validation_weights = role_weights(validation["labels"], "internal_validation")
        lower, lower_selection = choose_lower_threshold(
            validation["labels"], validation["presence"], upper, validation_weights
        )
        quality_threshold, quality_selection = choose_quality_threshold(
            validation["quality_labels"], validation["quality"]
        )
        segmentation_rule, validation_segmentation = select_segmentation_by_dice(
            validation["segmentation"],
            validation["observable"],
            validation["truth"],
            validation["labels"],
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "validation_first_joint_mil_development",
            "source": {
                "repository": "UNEP-IMEO/MARS-S2L",
                "revision": REVISION,
                "development_manifest_sha256": sha256(manifest_path),
            },
            "runtime": {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
            },
            "model": model.artifact_metadata(),
            "training": {
                "seed": args.seed,
                "best_epoch": best_epoch,
                "epochs_requested": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "patience": args.patience,
                "presence_loss_weight": PRESENCE_LOSS_WEIGHT,
                "quality_loss_weight": QUALITY_LOSS_WEIGHT,
                "history": history,
            },
            "operating_rule": {
                "selected_on": "internal_validation_only",
                "upper_plume_threshold": upper,
                "lower_no_plume_threshold": lower,
                "lower_selection": lower_selection,
                "quality_threshold": quality_threshold,
                "quality_selection": quality_selection,
                "segmentation": segmentation_rule,
            },
            "validation": {
                "scene": validation_scene,
                "segmentation": validation_segmentation,
                "quality_auroc": float(
                    roc_auc_score(validation["quality_labels"], validation["quality"])
                ),
            },
            "artifact": {
                "path": checkpoint.relative_to(root).as_posix(),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
                "tracked": False,
            },
            "provenance": {
                "git_commit": git_commit(root),
                "git_tracked_worktree_dirty_at_start": tracked_dirty(root),
                "script": "tools/train_mars_joint_mil.py",
                "script_sha256": sha256(Path(__file__)),
                "model_source": "EarthRemoteSensingRapidResponse/mars_joint_mil_model.py",
                "model_source_sha256": sha256(MODEL_ROOT / "mars_joint_mil_model.py"),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "rasterio": rasterio.__version__,
                "sklearn": sklearn.__version__,
            },
        }
        validation_recall = float(validation_scene["recall"] or 0.0)
        if args.evaluate_test:
            test = collect_predictions(model, test_loader, device)
            test_weights = role_weights(test["labels"], "strict_spatial_test")
            scene_unweighted = metrics(test["labels"], test["presence"], upper)
            segmentation = evaluate_rule(
                test["segmentation"],
                test["observable"],
                test["truth"],
                test["labels"],
                test["groups"].astype(str),
                segmentation_rule,
            )
            ci = bootstrap_ci(
                test["labels"], test["presence"], test["groups"].astype(str), upper, args.seed
            )
            selective = selective_quality_metrics(
                test["labels"],
                test["presence"],
                test["quality"],
                lower,
                upper,
                quality_threshold,
                test_weights,
            )
            report["test"] = {
                "scene_unweighted": scene_unweighted,
                "scene_representative_weighted": metrics(
                    test["labels"], test["presence"], upper, weights=test_weights
                ),
                "segmentation": segmentation,
                "group_bootstrap": ci,
                "selective_with_quality": selective,
                "quality_auroc": float(
                    roc_auc_score(test["quality_labels"], test["quality"])
                ),
            }
            gate = (
                float(ci["recall_95ci"][0]) >= 0.75
                and float(scene_unweighted["false_positive_rate"] or 1.0) <= 0.05
                and float(scene_unweighted["specificity"] or 0.0) >= 0.95
            )
            report["promotion_gate_passed_on_development_tranche"] = gate
            report["decision"] = (
                "MIL v2 clears the provisional gate; run the remaining four fixed seeds and released baselines before promotion."
                if gate
                else "MIL v2 does not clear the strict-spatial gate. Freeze this result and use only validation/external data for the next decision."
            )
        else:
            report["decision"] = (
                "MIL v2 materially improves validation recall at controlled FPR; freeze the checkpoint and authorize its one-time strict-spatial evaluation."
                if validation_recall >= 0.25
                else "MIL v2 does not materially improve controlled-FPR validation recall; do not consume another strict-spatial evaluation."
            )
        write_json(output_json, report)
        write_markdown(output_markdown, report)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, rasterio.errors.RasterioError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=None if args.compact else 2))
        return 2
    payload = {
        "ok": True,
        "best_epoch": best_epoch,
        "validation_recall": validation_scene["recall"],
        "validation_fpr": validation_scene["false_positive_rate"],
        "validation_ap": validation_scene["average_precision"],
        "evaluated_test": bool(args.evaluate_test),
        "checkpoint_sha256": report["artifact"]["sha256"],
        "output_json": output_json.relative_to(root).as_posix(),
        "output_markdown": output_markdown.relative_to(root).as_posix(),
    }
    print(json.dumps(payload, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
