#!/usr/bin/env python3
"""Extract honest dense-Prithvi crossfit scene representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
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
from extract_mars_dense_prithvi_scene_features import (  # noqa: E402
    feature_names,
    load_adapter_state,
    masked_statistics,
)
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


DEFAULT_PROTOCOL = Path(
    "configs/mars_dense_prithvi_crossfit_scene_feature_protocol.json"
)


def stable_stat(path: Path, attempts: int = 20) -> os.stat_result:
    """Tolerate the short DrvFs visibility delay observed after atomic rename."""
    failure: FileNotFoundError | None = None
    for _ in range(attempts):
        try:
            return path.stat()
        except FileNotFoundError as error:
            failure = error
            time.sleep(0.05)
    if failure is not None:
        raise failure
    raise RuntimeError("stable_stat exhausted without a result")


def validate_artifact(
    artifact: dict[str, Any],
    *,
    held_fold: int,
    fit_folds: list[int],
    mask_strength: float,
    training_protocol_sha256: str,
) -> None:
    if artifact.get("kind") != "mars_dense_prithvi_crossfit_representation":
        raise ValueError("Crossfit adapter artifact kind differs from the frozen schema")
    if int(artifact["held_fold"]) != held_fold:
        raise ValueError("Crossfit adapter held fold differs from the extraction job")
    artifact_fit = sorted(map(int, artifact["fit_folds"]))
    if artifact_fit != sorted(map(int, fit_folds)) or held_fold in artifact_fit:
        raise ValueError("Crossfit adapter fit folds differ from the extraction job")
    if float(artifact["mask_strength"]) != mask_strength:
        raise ValueError("Crossfit adapter mask strength differs from the protocol")
    if float(artifact["scene_strength"]) != 0.0:
        raise ValueError("Crossfit adapter does not preserve the scene-score floor")
    if str(artifact["protocol_sha256"]) != training_protocol_sha256:
        raise ValueError("Crossfit adapter training protocol identity differs")


@torch.no_grad()
def extract_job(
    *,
    job: dict[str, Any],
    records: list[dict[str, Any]],
    group_to_fold: dict[str, int],
    paths: dict[str, Path],
    features: np.ndarray,
    row_by_id: dict[str, int],
    base_scores: np.ndarray,
    output_row_by_id: dict[str, int],
    matrix: np.ndarray,
    written: np.ndarray,
    labels: np.ndarray,
    sensors: np.ndarray,
    spec: dict[str, Any],
    device: torch.device,
    smoke: bool,
) -> dict[str, Any]:
    held_fold = int(job["held_fold"])
    held_records = [
        row
        for row in records
        if group_to_fold[str(row["group_id"])] == held_fold
    ]
    if smoke:
        held_records = held_records[:8]
    if not held_records:
        raise ValueError(f"No records selected for held fold {held_fold}")
    dataset = DensePrithviDataset(
        paths["metadata_root"],
        held_records,
        features=features,
        row_by_id=row_by_id,
        base_scores=base_scores,
        augment=False,
        seed=int(job["seed"]),
    )
    workers = 0 if smoke else int(spec["loader_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(spec["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    model = DensePrithviTeacherAdapter().to(device)
    model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
    artifact_path = (ROOT / job["artifact"]["path"]).resolve()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    validate_artifact(
        artifact,
        held_fold=held_fold,
        fit_folds=list(map(int, job["fit_folds"])),
        mask_strength=float(spec["mask_strength"]),
        training_protocol_sha256=sha256(paths["crossfit_protocol"]),
    )
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
    cursor = 0
    mask_strength = float(spec["mask_strength"])
    names = feature_names()
    try:
        for batch_index, batch in enumerate(loader, start=1):
            local_ids = [str(value) for value in batch["sample_id"]]
            positions = np.asarray(
                [output_row_by_id[sample_id] for sample_id in local_ids],
                dtype=np.int64,
            )
            if written[positions].any():
                raise ValueError("Crossfit scene row was written more than once")
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
                raise ValueError("Dense scene hook geometry differs from the schema")
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
                raise RuntimeError("Crossfit scene feature schema or finiteness failure")
            matrix[positions] = values.cpu().numpy().astype(np.float16)
            labels[positions] = batch["presence"].cpu().numpy().astype(np.uint8)
            sensors[positions] = batch["sensor_index"].cpu().numpy().astype(np.uint8)
            written[positions] = True
            cursor += len(local_ids)
            if batch_index % 250 == 0 or cursor == len(held_records):
                if hasattr(matrix, "flush"):
                    matrix.flush()
                print(
                    json.dumps(
                        {
                            "held_fold": held_fold,
                            "batches": batch_index,
                            "rows": cursor,
                        }
                    ),
                    flush=True,
                )
    finally:
        pre_handle.remove()
        post_handle.remove()
    return {
        "held_fold": held_fold,
        "fit_folds": sorted(map(int, job["fit_folds"])),
        "rows": cursor,
        "artifact_sha256": sha256(artifact_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["extractor"]["sha256"]:
        raise ValueError("Frozen crossfit scene-feature extractor hash mismatch")
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
    all_records = list(iter_development_manifest(paths["manifest"]))
    selected_folds = {int(job["held_fold"]) for job in protocol["jobs"]}
    if len(selected_folds) != len(protocol["jobs"]):
        raise ValueError("Crossfit extraction jobs repeat a held fold")
    records = [
        row
        for row in all_records
        if group_to_fold[str(row["group_id"])] in selected_folds
    ]
    if args.smoke:
        smoke_ids = {
            str(row["sample_id"])
            for job in protocol["jobs"]
            for row in [
                item
                for item in records
                if group_to_fold[str(item["group_id"])] == int(job["held_fold"])
            ][:8]
        }
        records = [row for row in records if str(row["sample_id"]) in smoke_ids]
    if not records:
        raise ValueError("No records selected for crossfit feature extraction")
    sample_ids = [str(row["sample_id"]) for row in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Crossfit extraction selected duplicate sample identifiers")
    output_row_by_id = {
        sample_id: index for index, sample_id in enumerate(sample_ids)
    }
    exact_base_scores = np.asarray(
        [base_scores[row_by_id[sample_id]] for sample_id in sample_ids],
        dtype=np.float64,
    )
    if (
        not np.isfinite(exact_base_scores).all()
        or np.any((exact_base_scores < 0) | (exact_base_scores > 1))
    ):
        raise ValueError("Selected exact base scores are incomplete or outside [0,1]")
    groups = np.asarray([str(row["group_id"]) for row in records])
    folds = np.asarray(
        [group_to_fold[str(row["group_id"])] for row in records], dtype=np.uint8
    )
    labels = np.full(len(records), 255, dtype=np.uint8)
    sensors = np.full(len(records), 255, dtype=np.uint8)
    names = feature_names()
    output_features = (ROOT / protocol["outputs"]["features"]).resolve()
    output_metadata = (ROOT / protocol["outputs"]["metadata"]).resolve()
    output_receipt = (ROOT / protocol["outputs"]["receipt"]).resolve()
    if args.smoke:
        temporary = None
        matrix = np.empty((len(records), len(names)), dtype=np.float16)
    else:
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
    written = np.zeros(len(records), dtype=bool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Crossfit scene-feature extraction requires CUDA")
    jobs = []
    for job in protocol["jobs"]:
        artifact_path = (ROOT / job["artifact"]["path"]).resolve()
        if sha256(artifact_path) != job["artifact"]["sha256"]:
            raise ValueError(f"Frozen crossfit artifact mismatch: {artifact_path}")
        jobs.append(
            extract_job(
                job=job,
                records=records,
                group_to_fold=group_to_fold,
                paths=paths,
                features=features,
                row_by_id=row_by_id,
                base_scores=base_scores,
                output_row_by_id=output_row_by_id,
                matrix=matrix,
                written=written,
                labels=labels,
                sensors=sensors,
                spec=protocol["runtime"],
                device=device,
                smoke=args.smoke,
            )
        )
    if not written.all() or np.any(labels > 1) or np.any(sensors > 1):
        raise RuntimeError("Crossfit scene-feature extraction is incomplete")
    if args.smoke:
        floor_matches = bool(
            np.allclose(
                exact_base_scores,
                np.asarray(matrix[:, 0], dtype=np.float64),
                atol=2.5e-4,
                rtol=5e-4,
            )
        )
        print(
            json.dumps(
                {
                    "ok": bool(np.isfinite(matrix).all()) and floor_matches,
                    "rows": len(records),
                    "shape": list(matrix.shape),
                    "exact_floor_matches_float16_feature": floor_matches,
                    "jobs": jobs,
                    "feature_identity": feature_identity,
                }
            )
        )
        return 0

    matrix.flush()
    del matrix
    if temporary is None:
        raise RuntimeError("Crossfit feature temporary path is unavailable")
    feature_hash = sha256(temporary)
    os.replace(temporary, output_features)
    stable_stat(output_features)
    if sha256(output_features) != feature_hash:
        raise RuntimeError("Crossfit feature hash changed across atomic promotion")
    artifact_hashes = {
        str(job["held_fold"]): str(job["artifact_sha256"]) for job in jobs
    }
    atomic_savez(
        output_metadata,
        feature_names=np.asarray(names),
        sample_ids=np.asarray(sample_ids),
        labels=labels,
        sensors=sensors,
        groups=groups,
        folds=folds,
        exact_base_scores=exact_base_scores,
        features_sha256=np.asarray(feature_hash),
        adapter_sha256_by_fold=np.asarray(
            json.dumps(artifact_hashes, sort_keys=True)
        ),
        manifest_sha256=np.asarray(sha256(paths["manifest"])),
        fold_protocol_sha256=np.asarray(sha256(paths["fold_protocol"])),
        sample_id_sha256=np.asarray(
            hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
        ),
        input_contract=np.asarray(
            json.dumps(
                {
                    str(job["held_fold"]): list(map(int, job["fit_folds"]))
                    for job in jobs
                },
                sort_keys=True,
            )
        ),
    )
    stable_stat(output_metadata)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = {
        "schema_version": 1,
        "scope": "honest dense-Prithvi crossfit representation cache",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(records),
        "feature_count": len(names),
        "folds": sorted(selected_folds),
        "features": {
            "path": output_features.relative_to(ROOT).as_posix(),
            "bytes": stable_stat(output_features).st_size,
            "sha256": feature_hash,
        },
        "metadata": {
            "path": output_metadata.relative_to(ROOT).as_posix(),
            "bytes": stable_stat(output_metadata).st_size,
            "sha256": sha256(output_metadata),
        },
        "jobs": jobs,
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
