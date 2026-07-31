#!/usr/bin/env python3
"""Extract frozen dense-Prithvi adapter embeddings for development-only scene ranking."""

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
import torch
from numpy.lib.format import open_memmap
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from extract_mars_scene_features import atomic_savez  # noqa: E402
from mars_dense_prithvi_teacher import DensePrithviTeacherAdapter  # noqa: E402
from mars_paper_model import released_state  # noqa: E402
from train_mars_dense_prithvi_teacher_pilot import (  # noqa: E402
    DensePrithviDataset,
    PRITHVI_GRID_SIZE,
    load_feature_contract,
)
from train_mars_paper_residual import (  # noqa: E402
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)


DEFAULT_PROTOCOL = Path("configs/mars_dense_prithvi_scene_feature_protocol.json")


def feature_names() -> list[str]:
    names = ["base_scene_score", "rejected_scene_delta_logit"]
    names.extend(f"scene_head_input_{index}" for index in range(585))
    names.extend(f"scene_embedding_{index}" for index in range(64))
    for prefix in ("released_probability", "adapted_probability", "logit_correction"):
        names.extend(
            f"{prefix}_{statistic}"
            for statistic in ("mean", "std", "max", "top10_mean", "top100_mean")
        )
    names.extend(
        f"patch_logit_{statistic}"
        for statistic in ("mean", "std", "max", "top4_mean", "top16_mean")
    )
    return names


def masked_statistics(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    top_counts: tuple[int, int],
) -> torch.Tensor:
    if values.ndim != 4 or values.shape[1] != 1 or valid.shape != values.shape:
        raise ValueError("Statistics require matching Bx1xHxW values and masks")
    flat = values.float().flatten(1)
    mask = valid.bool().flatten(1)
    weights = mask.to(flat.dtype)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (flat * weights).sum(dim=1) / count
    variance = ((flat - mean[:, None]).square() * weights).sum(dim=1) / count
    masked = flat.masked_fill(~mask, -1e4)
    maximum = masked.amax(dim=1)
    top_values = []
    for requested in top_counts:
        k = min(int(requested), flat.shape[1])
        selected = torch.topk(masked, k=k, dim=1).values
        selected_valid = selected > -1e3
        top_values.append(
            selected.masked_fill(~selected_valid, 0.0).sum(dim=1)
            / selected_valid.sum(dim=1).clamp_min(1)
        )
    return torch.stack((mean, variance.clamp_min(0).sqrt(), maximum, *top_values), dim=1)


def load_adapter_state(
    model: DensePrithviTeacherAdapter, artifact: dict[str, Any]
) -> None:
    state = artifact["adapter_state"]
    current = model.state_dict()
    expected = {name for name in current if not name.startswith("teacher.")}
    if set(state) != expected:
        raise ValueError(
            f"Dense adapter state differs: missing={sorted(expected - set(state))}, "
            f"extra={sorted(set(state) - expected)}"
        )
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["extractor"]["sha256"]:
        raise ValueError("Frozen dense scene-feature extractor hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen input mismatch: {name}")
        paths[name] = path
    if not paths["metadata_root"].is_dir():
        raise ValueError("MARS metadata root is unavailable")
    verify_acquisition_receipt(paths["acquisition_receipt"], sha256(paths["manifest"]))
    features, row_by_id, base_scores, feature_identity = load_feature_contract(
        paths["prithvi_features"],
        paths["prithvi_metadata"],
        paths["score_cache"],
        str(protocol["inputs"]["prithvi_features"]["sha256"]),
    )
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(row["group_id"]): int(row["fold"]) for row in fold_protocol["assignments"]
    }
    records = list(iter_development_manifest(paths["manifest"]))
    selected_folds = set(map(int, protocol["folds"]))
    records = [
        row for row in records if group_to_fold[str(row["group_id"])] in selected_folds
    ]
    if args.smoke:
        records = records[:16]
    elif args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        records = records[: args.max_rows]
    if not records:
        raise ValueError("No records selected for dense scene-feature extraction")

    dataset = DensePrithviDataset(
        paths["metadata_root"],
        records,
        features=features,
        row_by_id=row_by_id,
        base_scores=base_scores,
        augment=False,
        seed=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(protocol["runtime"]["batch_size"]),
        shuffle=False,
        num_workers=0 if args.smoke else int(protocol["runtime"]["loader_workers"]),
        pin_memory=True,
        persistent_workers=(
            not args.smoke and int(protocol["runtime"]["loader_workers"]) > 0
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Dense scene-feature extraction requires CUDA")
    model = DensePrithviTeacherAdapter().to(device)
    model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
    artifact = torch.load(paths["adapter"], map_location="cpu", weights_only=True)
    if artifact.get("kind") != "mars_dense_prithvi_mask_adapter":
        raise ValueError("Dense adapter artifact kind differs from the frozen schema")
    if float(artifact["mask_strength"]) != float(protocol["mask_strength"]):
        raise ValueError("Dense adapter mask strength differs from the protocol")
    if float(artifact["scene_strength"]) != 0.0:
        raise ValueError("Dense adapter artifact does not preserve scene-score identity")
    load_adapter_state(model, artifact)
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def capture_input(
        _module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]
    ) -> None:
        captured["scene_input"] = arguments[0].detach()

    def capture_output(
        _module: torch.nn.Module,
        _arguments: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["scene_embedding"] = output.detach()

    pre_handle = model.scene_hidden.register_forward_pre_hook(capture_input)
    post_handle = model.scene_hidden.register_forward_hook(capture_output)
    names = feature_names()
    output_features = (ROOT / protocol["outputs"]["features"]).resolve()
    output_metadata = (ROOT / protocol["outputs"]["metadata"]).resolve()
    output_receipt = (ROOT / protocol["outputs"]["receipt"]).resolve()
    if not args.smoke:
        for path in (output_features, output_metadata, output_receipt):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        output_features.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_features.with_suffix(".tmp.npy")
        if temporary.exists():
            raise FileExistsError(f"Stale temporary feature cache exists: {temporary}")
        matrix = open_memmap(
            temporary,
            mode="w+",
            dtype=np.float16,
            shape=(len(records), len(names)),
        )
    else:
        temporary = None
        matrix = np.empty((len(records), len(names)), dtype=np.float16)

    cursor = 0
    sample_ids: list[str] = []
    labels: list[int] = []
    sensors: list[int] = []
    groups: list[str] = []
    folds: list[int] = []
    mask_strength = float(protocol["mask_strength"])
    for batch_index, batch in enumerate(loader, start=1):
        local_ids = [str(value) for value in batch["sample_id"]]
        local_groups = [str(value) for value in batch["group_id"]]
        batch = move_batch(batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model(
                batch["inputs"],
                batch["observable"],
                batch["sensor_index"],
                batch["prithvi_tokens"],
                batch["base_scene_score"],
            )
        scene_input = captured.pop("scene_input").float()
        scene_embedding = captured.pop("scene_embedding").float()
        if scene_input.shape[1] != 585 or scene_embedding.shape[1] != 64:
            raise ValueError("Dense scene hook geometry differs from the frozen schema")
        observable = batch["observable"] > 0.5
        baseline_probability = torch.sigmoid(output["baseline_logits"].float())
        adapted_probability = torch.sigmoid(
            output["baseline_logits"].float()
            + mask_strength * output["correction_logits"].float()
        )
        full_stats = [
            masked_statistics(
                baseline_probability, observable, top_counts=(10, 100)
            ),
            masked_statistics(
                adapted_probability, observable, top_counts=(10, 100)
            ),
            masked_statistics(
                output["correction_logits"], observable, top_counts=(10, 100)
            ),
        ]
        patch_valid = (
            F.adaptive_avg_pool2d(
                batch["observable"].float(),
                (PRITHVI_GRID_SIZE, PRITHVI_GRID_SIZE),
            )
            >= 0.9
        )
        patch_stats = masked_statistics(
            output["patch_logits"], patch_valid, top_counts=(4, 16)
        )
        values = torch.cat(
            (
                batch["base_scene_score"].float()[:, None],
                output["scene_delta_logit"].float()[:, None],
                scene_input,
                scene_embedding,
                *full_stats,
                patch_stats,
            ),
            dim=1,
        )
        if values.shape[1] != len(names) or not torch.isfinite(values).all():
            raise RuntimeError("Dense scene feature schema or finiteness failure")
        local = values.cpu().numpy().astype(np.float16)
        end = cursor + local.shape[0]
        matrix[cursor:end] = local
        cursor = end
        sample_ids.extend(local_ids)
        labels.extend(int(value) for value in batch["presence"].cpu().numpy())
        sensors.extend(int(value) for value in batch["sensor_index"].cpu().numpy())
        groups.extend(local_groups)
        folds.extend(group_to_fold[group] for group in local_groups)
        if batch_index % 250 == 0 or cursor == len(records):
            if hasattr(matrix, "flush"):
                matrix.flush()
            print(json.dumps({"batches": batch_index, "rows": cursor}), flush=True)
    pre_handle.remove()
    post_handle.remove()
    if cursor != len(records):
        raise RuntimeError("Dense scene-feature extraction row count is incomplete")
    if args.smoke:
        print(
            json.dumps(
                {
                    "ok": True,
                    "rows": cursor,
                    "shape": list(matrix.shape),
                    "feature_names": names[:5] + ["..."] + names[-5:],
                    "finite": bool(np.isfinite(matrix).all()),
                    "feature_identity": feature_identity,
                }
            )
        )
        return 0

    matrix.flush()
    del matrix
    feature_hash = sha256(temporary)
    os.replace(temporary, output_features)
    atomic_savez(
        output_metadata,
        feature_names=np.asarray(names),
        sample_ids=np.asarray(sample_ids),
        labels=np.asarray(labels, dtype=np.uint8),
        sensors=np.asarray(sensors, dtype=np.uint8),
        groups=np.asarray(groups),
        folds=np.asarray(folds, dtype=np.uint8),
        features_sha256=np.asarray(feature_hash),
        adapter_sha256=np.asarray(sha256(paths["adapter"])),
        manifest_sha256=np.asarray(sha256(paths["manifest"])),
        fold_protocol_sha256=np.asarray(sha256(paths["fold_protocol"])),
        sample_id_sha256=np.asarray(
            hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
        ),
        input_contract=np.asarray(
            "frozen fold2 mask adapter; scene residual retained only as a feature; current scene score unchanged"
        ),
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = {
        "schema_version": 1,
        "scope": "development-only frozen dense-Prithvi representation cache",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": cursor,
        "feature_count": len(names),
        "folds": sorted(selected_folds),
        "features": {
            "path": output_features.relative_to(ROOT).as_posix(),
            "bytes": output_features.stat().st_size,
            "sha256": feature_hash,
        },
        "metadata": {
            "path": output_metadata.relative_to(ROOT).as_posix(),
            "bytes": output_metadata.stat().st_size,
            "sha256": sha256(output_metadata),
        },
        "adapter_sha256": sha256(paths["adapter"]),
        "protocol_sha256": sha256(protocol_path),
        "extractor_sha256": sha256(Path(__file__).resolve()),
        "git_commit": commit,
        "external_inputs_accessed": False,
    }
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = output_receipt.with_suffix(output_receipt.suffix + ".tmp")
    temporary_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_receipt, output_receipt)
    print(json.dumps({"ok": True, **receipt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
