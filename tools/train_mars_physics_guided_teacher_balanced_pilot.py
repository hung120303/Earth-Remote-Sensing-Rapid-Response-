#!/usr/bin/env python3
"""Run the paper-aligned balanced-sampling physics-guided teacher pilot."""

from __future__ import annotations

import argparse
import json
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
from mars_paper_model import released_state  # noqa: E402
from mars_physics_guided_teacher import PhysicsGuidedTeacherAdapter  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    MarsPaperDataset,
    iter_development_manifest,
    move_batch,
    verify_acquisition_receipt,
)
from train_mars_physics_guided_teacher_pilot import evaluate, seed_everything, train_endpoint  # noqa: E402
from train_mars_source_aligned_residual import MarsSourceAlignedDataset, offshore_flags  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_physics_guided_teacher_balanced_pilot_protocol.json")


def balanced_request_weights(records: list[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, float]]:
    cells = [
        (str(row["group_id"]), str(row["label_state"]), str(row["sensor_family"]))
        for row in records
    ]
    row_counts = Counter(cells)
    unique_cells = set(cells)
    stratum_cells = Counter((label, sensor) for _, label, sensor in unique_cells)
    weights = torch.tensor(
        [
            1.0 / row_counts[cell] / stratum_cells[(cell[1], cell[2])]
            for cell in cells
        ],
        dtype=torch.double,
    )
    weights /= weights.sum()
    mass: dict[str, float] = {}
    for row, weight in zip(records, weights.tolist()):
        key = f"{row['label_state']}|{row['sensor_family']}"
        mass[key] = mass.get(key, 0.0) + float(weight)
    return weights, mass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen balanced-pilot trainer hash mismatch")
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
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {str(row["group_id"]): int(row["fold"]) for row in fold_protocol["assignments"]}
    records = list(iter_development_manifest(paths["manifest"]))
    fit_folds = set(map(int, protocol["folds"]["fit"]))
    held_fold = int(protocol["folds"]["held"])
    fit_records = [row for row in records if group_to_fold[str(row["group_id"])] in fit_folds]
    held_records = [row for row in records if group_to_fold[str(row["group_id"])] == held_fold]
    spec = protocol["training"]
    if args.smoke:
        selected = []
        counts: Counter[tuple[str, str]] = Counter()
        for row in fit_records:
            key = (str(row["label_state"]), str(row["sensor_family"]))
            if counts[key] < 4:
                selected.append(row)
                counts[key] += 1
            if len(counts) == 4 and min(counts.values()) >= 4:
                break
        fit_records = selected
        held_records = selected
    weights, request_mass = balanced_request_weights(fit_records)
    positive_mass = sum(value for key, value in request_mass.items() if key.startswith("PLUME|"))
    if not np.isclose(positive_mass, 0.5, atol=1e-12):
        raise ValueError(f"Balanced sampler positive request mass is {positive_mass}, expected 0.5")
    flags = offshore_flags(paths["metadata_csv"], {str(row["sample_id"]) for row in fit_records})
    dataset = MarsSourceAlignedDataset(
        paths["metadata_root"],
        fit_records,
        flags,
        lut_path=paths["lut"],
        augment=True,
        simulation_fraction=float(spec["simulation_fraction"]),
        crop_size=int(spec["crop_size"]),
        seed=int(spec["seed"]),
    )
    samples = 64 if args.smoke else int(spec["samples_per_epoch"])
    workers = 2 if args.smoke else int(spec["loader_workers"])
    batch_size = 4 if args.smoke else int(spec["batch_size"])
    sampler = WeightedRandomSampler(
        weights,
        num_samples=samples,
        replacement=True,
        generator=torch.Generator().manual_seed(int(spec["seed"])),
    )
    options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(dataset, sampler=sampler, **options)
    evaluation_loader = DataLoader(
        MarsPaperDataset(paths["metadata_root"], held_records, augment=False, seed=int(spec["seed"])),
        shuffle=False,
        **options,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Balanced physics-guided teacher pilot requires CUDA")
    model = PhysicsGuidedTeacherAdapter().to(device)
    model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
    with torch.no_grad():
        first = move_batch(next(iter(evaluation_loader)), device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            initial = model(first["inputs"], first["observable"], first["sensor_index"])
        identity_max = float(initial["correction_logits"].abs().max())
    if identity_max != 0.0:
        raise ValueError(f"Adapter initialization is not exact identity: {identity_max}")
    history = train_endpoint(
        model,
        train_loader,
        spec,
        device,
        int(spec["seed"]),
        1 if args.smoke else int(spec["epochs"]),
    )
    if args.smoke:
        finite = all(torch.isfinite(value).all() for value in model.trainable_state().values())
        print(json.dumps({"ok": finite, "identity_max_abs": identity_max, "request_mass": request_mass, "trainable_parameters": model.trainable_parameter_count(), "history": history}))
        return 0 if finite else 1
    candidates, identity = evaluate(
        model,
        evaluation_loader,
        [float(value) for value in protocol["search"]["strengths"]],
        device,
        protocol["bootstrap"],
    )
    selected = candidates[0]
    passed = bool(selected["passed"])
    artifact = None
    if passed:
        artifact_path = (ROOT / protocol["outputs"]["artifact"]).resolve()
        if artifact_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact_path}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "adapter_state": model.trainable_state(),
                "selected_strength": selected["strength"],
                "protocol_sha256": sha256(protocol_path),
            },
            artifact_path,
        )
        artifact = {"path": artifact_path.relative_to(ROOT).as_posix(), "bytes": artifact_path.stat().st_size, "sha256": sha256(artifact_path)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    report = {
        "schema_version": 1,
        "scope": "balanced-request physics-guided released-U-Net adapter v2 pilot",
        "all_promotion_gates_pass": passed,
        "decision": "Authorize preregistered multi-seed cross-fit." if passed else "Reject balanced adapter pilot before full cross-fit or external scoring.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "trainer_sha256": sha256(Path(__file__).resolve()),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "fit_folds": sorted(fit_folds),
        "held_fold": held_fold,
        "fit_rows": len(fit_records),
        "held_rows": len(held_records),
        "request_mass": request_mass,
        "positive_request_mass": positive_mass,
        "trainable_parameters": model.trainable_parameter_count(),
        "identity_max_abs": identity_max,
        "history": history,
        "released_identity": identity,
        "candidate": selected,
        "artifact": artifact,
        "external_inputs_accessed": False,
    }
    output = (ROOT / protocol["outputs"]["json"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": passed, "strength": selected["strength"], "ap_delta": selected["versus_released"]["delta"]["average_precision"], "ap_lower": selected["bootstrap"]["lower"], "recall_delta": selected["versus_released"]["delta"]["recall_at_fpr_0_0713"], "iou_delta": selected["pixel_iou_delta"], "realized_simulation": [row["simulated_fraction"] for row in history]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
