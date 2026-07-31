#!/usr/bin/env python3
"""Train honest fold-3/4 dense-Prithvi representation endpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from mars_dense_prithvi_teacher import DensePrithviTeacherAdapter  # noqa: E402
from mars_paper_model import released_state  # noqa: E402
from preserve_mars_dense_prithvi_mask_pilot import evaluate_mask_only  # noqa: E402
from train_mars_dense_prithvi_teacher_pilot import (  # noqa: E402
    DensePrithviDataset,
    load_feature_contract,
    train_endpoint,
)
from train_mars_paper_residual import (  # noqa: E402
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)
from train_mars_physics_guided_teacher_balanced_pilot import (  # noqa: E402
    balanced_request_weights,
)
from train_mars_physics_guided_teacher_pilot import seed_everything  # noqa: E402


DEFAULT_PROTOCOL = Path(
    "configs/mars_dense_prithvi_crossfit_representation_protocol.json"
)


def smoke_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    for row in records:
        key = (str(row["label_state"]), str(row["sensor_family"]))
        if counts[key] < 4:
            selected.append(row)
            counts[key] += 1
        if len(counts) == 4 and min(counts.values()) >= 4:
            break
    if len(counts) != 4:
        raise ValueError("Crossfit smoke lacks a complete label x sensor grid")
    return selected


def run_job(
    job: dict[str, Any],
    *,
    all_records: list[dict[str, Any]],
    group_to_fold: dict[str, int],
    paths: dict[str, Path],
    features: np.ndarray,
    row_by_id: dict[str, int],
    base_scores: np.ndarray,
    spec: dict[str, Any],
    bootstrap: dict[str, Any],
    device: torch.device,
    smoke: bool,
    protocol_path: Path,
) -> dict[str, Any]:
    fit_folds = set(map(int, job["fit_folds"]))
    held_fold = int(job["held_fold"])
    if held_fold in fit_folds:
        raise ValueError("Crossfit held fold is present in its fit folds")
    fit_records = [
        row
        for row in all_records
        if group_to_fold[str(row["group_id"])] in fit_folds
    ]
    held_records = [
        row
        for row in all_records
        if group_to_fold[str(row["group_id"])] == held_fold
    ]
    if smoke:
        fit_records = smoke_records(fit_records)
        held_records = smoke_records(held_records)
    weights, request_mass = balanced_request_weights(fit_records)
    if not np.isclose(
        sum(value for key, value in request_mass.items() if key.startswith("PLUME|")),
        0.5,
        atol=1e-12,
    ):
        raise ValueError("Crossfit sampler does not assign 0.5 plume request mass")
    train_dataset = DensePrithviDataset(
        paths["metadata_root"],
        fit_records,
        features=features,
        row_by_id=row_by_id,
        base_scores=base_scores,
        augment=True,
        seed=int(job["seed"]),
    )
    held_dataset = DensePrithviDataset(
        paths["metadata_root"],
        held_records,
        features=features,
        row_by_id=row_by_id,
        base_scores=base_scores,
        augment=False,
        seed=int(job["seed"]),
    )
    samples = 32 if smoke else int(spec["samples_per_epoch"])
    workers = 0 if smoke else int(spec["loader_workers"])
    sampler = WeightedRandomSampler(
        weights,
        num_samples=samples,
        replacement=True,
        generator=torch.Generator().manual_seed(int(job["seed"])),
    )
    options = {
        "batch_size": int(spec["batch_size"]),
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **options)
    held_loader = DataLoader(held_dataset, shuffle=False, **options)
    seed_everything(int(job["seed"]))
    model = DensePrithviTeacherAdapter().to(device)
    model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
    with torch.no_grad():
        first = move_batch(next(iter(held_loader)), device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            initial = model(
                first["inputs"],
                first["observable"],
                first["sensor_index"],
                first["prithvi_tokens"],
                first["base_scene_score"],
            )
        identity_pixel = float(initial["correction_logits"].abs().max())
        identity_scene = float(initial["scene_delta_logit"].abs().max())
    if identity_pixel != 0.0 or identity_scene != 0.0:
        raise ValueError("Crossfit representation is not exact identity at initialization")
    print(
        json.dumps(
            {
                "job": job["name"],
                "phase": "train",
                "fit_rows": len(fit_records),
                "held_rows": len(held_records),
            }
        ),
        flush=True,
    )
    local_spec = {**spec, "seed": int(job["seed"])}
    history = train_endpoint(
        model,
        train_loader,
        local_spec,
        device,
        1 if smoke else int(spec["epochs"]),
    )
    finite = all(
        torch.isfinite(value).all() for value in model.trainable_state().values()
    )
    if not finite:
        raise ValueError("Crossfit representation endpoint is non-finite")
    if smoke:
        return {
            "name": job["name"],
            "fit_folds": sorted(fit_folds),
            "held_fold": held_fold,
            "fit_rows": len(fit_records),
            "held_rows": len(held_records),
            "request_mass": request_mass,
            "identity_pixel_max_abs": identity_pixel,
            "identity_scene_max_abs": identity_scene,
            "history": history,
            "finite": finite,
            "artifact": None,
        }

    print(
        json.dumps({"job": job["name"], "phase": "held-mask-evaluation"}),
        flush=True,
    )
    mask_result, evaluation_identity = evaluate_mask_only(
        model,
        held_loader,
        strength=float(job["mask_strength"]),
        device=device,
        bootstrap={
            **bootstrap,
            "seed": int(bootstrap["seed"]) + held_fold,
        },
    )
    artifact_path = (ROOT / job["artifact"]).resolve()
    if artifact_path.exists():
        raise FileExistsError(f"Refusing to overwrite crossfit artifact: {artifact_path}")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "mars_dense_prithvi_crossfit_representation",
            "adapter_state": model.trainable_state(),
            "fit_folds": sorted(fit_folds),
            "held_fold": held_fold,
            "seed": int(job["seed"]),
            "mask_strength": float(job["mask_strength"]),
            "scene_strength": 0.0,
            "protocol_sha256": sha256(protocol_path),
        },
        artifact_path,
    )
    artifact = {
        "path": artifact_path.relative_to(ROOT).as_posix(),
        "bytes": artifact_path.stat().st_size,
        "sha256": sha256(artifact_path),
    }
    return {
        "name": job["name"],
        "fit_folds": sorted(fit_folds),
        "held_fold": held_fold,
        "fit_rows": len(fit_records),
        "held_rows": len(held_records),
        "request_mass": request_mass,
        "identity_pixel_max_abs": identity_pixel,
        "identity_scene_max_abs": identity_scene,
        "history": history,
        "finite": finite,
        "evaluation_identity": evaluation_identity,
        "mask_result_descriptive": mask_result,
        "artifact": artifact,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen crossfit representation trainer hash mismatch")
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
        paths["features"],
        paths["feature_metadata"],
        paths["score_cache"],
        str(protocol["inputs"]["features"]["sha256"]),
    )
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(row["group_id"]): int(row["fold"]) for row in fold_protocol["assignments"]
    }
    records = list(iter_development_manifest(paths["manifest"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Crossfit representation training requires CUDA")
    results = []
    for job in protocol["jobs"]:
        results.append(
            run_job(
                job,
                all_records=records,
                group_to_fold=group_to_fold,
                paths=paths,
                features=features,
                row_by_id=row_by_id,
                base_scores=base_scores,
                spec=protocol["training"],
                bootstrap=protocol["bootstrap"],
                device=device,
                smoke=args.smoke,
                protocol_path=protocol_path,
            )
        )
    if args.smoke:
        print(
            json.dumps(
                {
                    "ok": all(row["finite"] for row in results),
                    "feature_identity": feature_identity,
                    "jobs": results,
                }
            )
        )
        return 0

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "scope": "honest fold-3/4 crossfit dense-Prithvi representation endpoints",
        "status": "intermediate representation artifacts; not promoted scene or mask models",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "trainer_sha256": sha256(Path(__file__).resolve()),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "feature_identity": feature_identity,
        "jobs": results,
        "external_inputs_accessed": False,
    }
    output = (ROOT / protocol["output_report"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "ok": True,
                "jobs": [
                    {
                        "name": row["name"],
                        "held_fold": row["held_fold"],
                        "artifact": row["artifact"],
                        "mask_iou_delta": row["mask_result_descriptive"]["iou_delta"],
                        "mask_iou_lower": row["mask_result_descriptive"][
                            "paired_site_iou_delta"
                        ]["lower"],
                    }
                    for row in results
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
