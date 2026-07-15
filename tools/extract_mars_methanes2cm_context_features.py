#!/usr/bin/env python3
"""Extract frozen MethaneS2CM v5.1 context features for MARS development scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import DEFAULT_OUTPUT, repo_root, sha256  # noqa: E402
from extract_mars_scene_features import atomic_savez  # noqa: E402
from methanes2cm_v5_model import MethaneS2CMV5Model  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    DEFAULT_ACQUISITION_RECEIPT,
    DEFAULT_MANIFEST,
    DEFAULT_PROTOCOL,
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)

DEFAULT_CHECKPOINT = Path("EarthRemoteSensingRapidResponse/artifacts/methanes2cm_v5_1_seed1101.pt")
DEFAULT_SOURCE_REPORT = Path("reports/experiments/methanes2cm_v5_1_seed1101_validation.json")
DEFAULT_OUTPUT_CACHE = Path("outputs/mars_methanes2cm_v5_1_context_features_all_folds.npz")
CONTEXT_WIDTH = 390
MASK_STATISTICS = ("mean", "std", "max", "top0_1pct", "top0_5pct", "top1pct", "top2pct", "top5pct")


def feature_names() -> list[str]:
    return [
        *(f"methanes2cm_context_{index}" for index in range(CONTEXT_WIDTH)),
        "methanes2cm_mask_scene_logit",
        "methanes2cm_context_scene_logit",
        "methanes2cm_fused_scene_logit",
        *(f"methanes2cm_mask_probability_{name}" for name in MASK_STATISTICS),
    ]


def masked_probability_statistics(
    probability: torch.Tensor, observable: torch.Tensor
) -> torch.Tensor:
    flat = probability.float().flatten(1)
    valid = observable.flatten(1) > 0.5
    weight = valid.float()
    count = weight.sum(dim=1).clamp_min(1.0)
    mean = (flat * weight).sum(dim=1) / count
    variance = ((flat - mean[:, None]).square() * weight).sum(dim=1) / count
    masked = flat.masked_fill(~valid, -torch.inf)
    maximum = masked.amax(dim=1)
    parts = [mean, variance.sqrt(), maximum]
    for fraction in (0.001, 0.005, 0.01, 0.02, 0.05):
        width = max(1, int(flat.shape[1] * fraction))
        parts.append(torch.topk(masked, width, dim=1).values.mean(dim=1))
    result = torch.stack(parts, dim=1)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def mars_to_v5_input(inputs: torch.Tensor) -> torch.Tensor:
    """Map the two-frame MARS contract to v5.1 by repeating its reference frame."""
    if inputs.ndim != 4 or inputs.shape[1] != 16:
        raise ValueError("Expected the frozen 16-channel MARS input")
    mbmp = inputs[:, 0:1]
    target = inputs[:, 1:7]
    reference = inputs[:, 7:13]
    return torch.cat([mbmp, mbmp, target, reference, reference], dim=1)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--acquisition-receipt", default=DEFAULT_ACQUISITION_RECEIPT.as_posix())
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--source-report", default=DEFAULT_SOURCE_REPORT.as_posix())
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CACHE.as_posix())
    args = parser.parse_args()
    folds = tuple(sorted(set(args.folds)))
    if not folds or any(fold not in range(5) for fold in folds):
        parser.error("folds must be a non-empty subset of 0..4")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")

    root = repo_root()
    manifest = (root / args.manifest).resolve()
    protocol_path = (root / args.protocol).resolve()
    receipt_path = (root / args.acquisition_receipt).resolve()
    checkpoint = (root / args.checkpoint).resolve()
    source_report_path = (root / args.source_report).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest)
    if manifest_hash != protocol["development_manifest_sha256"]:
        raise ValueError("Development manifest differs from the frozen protocol")
    verify_acquisition_receipt(receipt_path, manifest_hash)
    if sha256(checkpoint) != source_report["checkpoint"]["sha256"]:
        raise ValueError("MethaneS2CM checkpoint differs from its validation report")
    if source_report["model"].get("context_scene_weight") != 0.65:
        raise ValueError("Expected a frozen v5.1 context checkpoint")
    group_to_fold = {
        str(item["group_id"]): int(item["fold"]) for item in protocol["assignments"]
    }
    records = [
        record
        for record in iter_development_manifest(manifest)
        if group_to_fold[str(record["group_id"])] in folds
    ]
    loader = DataLoader(
        MarsPaperDataset((root / args.metadata_dir).resolve(), records, augment=False, seed=0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MethaneS2CMV5Model(context_scene_weight=0.65).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload["model_metadata"] != source_report["model"]:
        raise ValueError("MethaneS2CM checkpoint metadata mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    captured: list[torch.Tensor] = []

    def capture_context(_module: torch.nn.Module, values: tuple[torch.Tensor, ...]) -> None:
        captured.append(values[0])

    assert model.context_scene is not None
    handle = model.context_scene.register_forward_pre_hook(capture_context)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    sensors: list[int] = []
    sample_ids: list[str] = []
    groups: list[str] = []
    row_folds: list[int] = []
    try:
        for batch_index, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            captured.clear()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(mars_to_v5_input(batch["inputs"]), batch["observable"])
            if len(captured) != 1 or captured[0].shape[1] != CONTEXT_WIDTH:
                raise RuntimeError("MethaneS2CM context hook violated its frozen schema")
            probability = torch.sigmoid(output["segmentation_logits"]) * batch["observable"]
            mask_statistics = masked_probability_statistics(probability, batch["observable"])
            local = torch.cat(
                [
                    captured[0].float(),
                    output["mask_scene_logit"].float()[:, None],
                    output["context_scene_logit"].float()[:, None],
                    output["scene_logit"].float()[:, None],
                    mask_statistics,
                ],
                dim=1,
            )
            if local.shape[1] != len(feature_names()) or not torch.isfinite(local).all():
                raise RuntimeError("MethaneS2CM transfer feature schema or finiteness failure")
            rows.append(local.cpu().numpy().astype(np.float16))
            labels.extend(int(value) for value in batch["presence"].cpu().numpy())
            sensors.extend(int(value) for value in batch["sensor_index"].cpu().numpy())
            for sample_id, group in zip(batch["sample_id"], batch["group_id"]):
                sample_ids.append(str(sample_id))
                groups.append(str(group))
                row_folds.append(group_to_fold[str(group)])
            if batch_index % 100 == 0:
                print(json.dumps({"batches": batch_index, "rows": len(labels)}), flush=True)
    finally:
        handle.remove()

    output_path = (root / args.output).resolve()
    atomic_savez(
        output_path,
        features=np.concatenate(rows),
        feature_names=np.asarray(feature_names()),
        labels=np.asarray(labels, dtype=np.uint8),
        sensors=np.asarray(sensors, dtype=np.uint8),
        sample_ids=np.asarray(sample_ids),
        groups=np.asarray(groups),
        folds=np.asarray(row_folds, dtype=np.uint8),
        checkpoint_sha256=np.asarray(sha256(checkpoint)),
        source_report_sha256=np.asarray(sha256(source_report_path)),
        manifest_sha256=np.asarray(manifest_hash),
        protocol_sha256=np.asarray(sha256(protocol_path)),
        mars_to_v5_mapping=np.asarray("repeat MARS reference as T-90 and T-365; repeat MBMP"),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "rows": len(labels),
                "features": len(feature_names()),
                "folds": list(folds),
                "output": args.output,
                "sha256": sha256(output_path),
                "checkpoint_sha256": sha256(checkpoint),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
