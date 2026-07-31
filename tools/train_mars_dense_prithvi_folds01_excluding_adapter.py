#!/usr/bin/env python3
"""Train the fixed dense-Prithvi adapter without opening folds 0, 1, or 2."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

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
from train_mars_dense_prithvi_crossfit_representations import (  # noqa: E402
    smoke_records,
)
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
    "configs/mars_dense_prithvi_folds01_excluding_adapter_protocol.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen folds-0/1-excluding trainer hash mismatch")
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
    verify_acquisition_receipt(
        paths["acquisition_receipt"],
        sha256(paths["manifest"]),
    )
    features, row_by_id, base_scores, feature_identity = load_feature_contract(
        paths["features"],
        paths["feature_metadata"],
        paths["score_cache"],
        str(protocol["inputs"]["features"]["sha256"]),
    )
    fold_protocol = json.loads(
        paths["fold_protocol"].read_text(encoding="utf-8")
    )
    group_to_fold = {
        str(row["group_id"]): int(row["fold"])
        for row in fold_protocol["assignments"]
    }
    fit_folds = set(map(int, protocol["job"]["fit_folds"]))
    held_folds = set(map(int, protocol["job"]["held_folds"]))
    forbidden_folds = set(map(int, protocol["job"]["forbidden_folds"]))
    if fit_folds & held_folds:
        raise ValueError("Held fold is present in folds-0/1-excluding fit folds")
    if fit_folds & forbidden_folds:
        raise ValueError("Forbidden fold is present in adapter fit folds")
    if held_folds != {0, 1} or forbidden_folds != {0, 1, 2}:
        raise ValueError("Unexpected independent-confirmation fold contract")
    records = [
        row
        for row in iter_development_manifest(paths["manifest"])
        if group_to_fold[str(row["group_id"])] in fit_folds
    ]
    if args.smoke:
        records = smoke_records(records)
    if not records:
        raise ValueError("Folds-0/1-excluding training selection is empty")
    record_folds = {
        group_to_fold[str(row["group_id"])]
        for row in records
    }
    if record_folds & forbidden_folds:
        raise ValueError("A forbidden fold leaked into adapter training records")

    weights, request_mass = balanced_request_weights(records)
    plume_mass = sum(
        value
        for key, value in request_mass.items()
        if key.startswith("PLUME|")
    )
    if not np.isclose(plume_mass, 0.5, atol=1e-12):
        raise ValueError(
            "Folds-0/1-excluding sampler lacks 0.5 plume request mass"
        )
    spec = protocol["training"]
    seed = int(protocol["job"]["seed"])
    dataset = DensePrithviDataset(
        paths["metadata_root"],
        records,
        features=features,
        row_by_id=row_by_id,
        base_scores=base_scores,
        augment=True,
        seed=seed,
    )
    samples = 32 if args.smoke else int(spec["samples_per_epoch"])
    workers = 0 if args.smoke else int(spec["loader_workers"])
    sampler = WeightedRandomSampler(
        weights,
        num_samples=samples,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=int(spec["batch_size"]),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Folds-0/1-excluding adapter training requires CUDA")
    seed_everything(seed)
    model = DensePrithviTeacherAdapter().to(device)
    model.load_released_checkpoint(released_state(paths["released_checkpoint"]))
    with torch.no_grad():
        first = move_batch(next(iter(loader)), device)
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
        raise ValueError(
            "Folds-0/1-excluding adapter is not exact identity at initialization"
        )

    local_spec = {**spec, "seed": seed}
    history = train_endpoint(
        model,
        loader,
        local_spec,
        device,
        1 if args.smoke else int(spec["epochs"]),
    )
    finite = all(
        torch.isfinite(value).all()
        for value in model.trainable_state().values()
    )
    if not finite:
        raise ValueError("Folds-0/1-excluding adapter endpoint is non-finite")
    smoke_summary = {
        "ok": finite,
        "fit_folds": sorted(fit_folds),
        "held_folds": sorted(held_folds),
        "forbidden_folds": sorted(forbidden_folds),
        "fit_rows": len(records),
        "request_mass": request_mass,
        "identity_pixel_max_abs": identity_pixel,
        "identity_scene_max_abs": identity_scene,
        "history": history,
        "feature_identity": feature_identity,
    }
    if args.smoke:
        print(json.dumps(smoke_summary))
        return 0

    artifact_path = (ROOT / protocol["job"]["artifact"]).resolve()
    if artifact_path.exists():
        raise FileExistsError(
            "Refusing to overwrite folds-0/1-excluding adapter: "
            f"{artifact_path}"
        )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "kind": "mars_dense_prithvi_crossfit_representation",
            "adapter_state": model.trainable_state(),
            "fit_folds": sorted(fit_folds),
            "held_folds": sorted(held_folds),
            "forbidden_folds": sorted(forbidden_folds),
            "seed": seed,
            "mask_strength": float(protocol["job"]["mask_strength"]),
            "scene_strength": 0.0,
            "protocol_sha256": sha256(protocol_path),
        },
        artifact_path,
    )
    artifact = {
        "path": artifact_path.relative_to(ROOT).as_posix(),
        "bytes": artifact_path.stat().st_size,
        "sha256": sha256(artifact_path),
        "tracked": False,
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
        "scope": (
            "fit-only dense-Prithvi adapter on folds 3+4 for one-shot "
            "folds-0/1 feature extraction"
        ),
        "status": "intermediate folds-0/1-excluding representation endpoint",
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
        "held_folds": sorted(held_folds),
        "forbidden_folds": sorted(forbidden_folds),
        "fit_rows": len(records),
        "held_records_loaded": 0,
        "request_mass": request_mass,
        "identity_pixel_max_abs": identity_pixel,
        "identity_scene_max_abs": identity_scene,
        "history": history,
        "finite": finite,
        "feature_identity": feature_identity,
        "artifact": artifact,
        "folds01_labels_accessed": False,
        "fold2_inputs_or_labels_accessed": False,
        "fresh_inputs_accessed": False,
        "exact_paper_inputs_accessed": False,
    }
    output = (ROOT / protocol["output_report"]).resolve()
    if output.exists():
        raise FileExistsError(
            "Refusing to repeat or overwrite folds-0/1-excluding report"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({"ok": True, "artifact": artifact}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
