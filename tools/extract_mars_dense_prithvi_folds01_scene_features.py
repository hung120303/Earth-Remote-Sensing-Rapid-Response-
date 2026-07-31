#!/usr/bin/env python3
"""Extract one-shot dense-Prithvi scene features for held folds 0 and 1."""

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

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
import extract_mars_dense_prithvi_crossfit_scene_features as crossfit  # noqa: E402
from extract_mars_dense_prithvi_scene_features import feature_names  # noqa: E402
from extract_mars_scene_features import atomic_savez  # noqa: E402
from train_mars_dense_prithvi_teacher_pilot import (  # noqa: E402
    load_feature_contract,
)
from train_mars_paper_residual import (  # noqa: E402
    iter_development_manifest,
    verify_acquisition_receipt,
)


DEFAULT_PROTOCOL = Path(
    "configs/mars_dense_prithvi_folds01_scene_feature_protocol.json"
)


def validate_multifold_artifact(
    artifact: dict[str, Any],
    *,
    held_fold: int,
    fit_folds: list[int],
    mask_strength: float,
    training_protocol_sha256: str,
) -> None:
    if artifact.get("kind") != "mars_dense_prithvi_crossfit_representation":
        raise ValueError("Folds-0/1 adapter kind differs from the frozen schema")
    artifact_held = sorted(map(int, artifact["held_folds"]))
    artifact_forbidden = sorted(map(int, artifact["forbidden_folds"]))
    if artifact_held != [0, 1] or artifact_forbidden != [0, 1, 2]:
        raise ValueError("Folds-0/1 adapter holdout contract differs")
    if held_fold not in artifact_held:
        raise ValueError("Extraction fold is absent from adapter held folds")
    artifact_fit = sorted(map(int, artifact["fit_folds"]))
    if (
        artifact_fit != sorted(map(int, fit_folds))
        or set(artifact_fit) & set(artifact_forbidden)
    ):
        raise ValueError("Folds-0/1 adapter fit folds differ")
    if float(artifact["mask_strength"]) != mask_strength:
        raise ValueError("Folds-0/1 adapter mask strength differs")
    if float(artifact["scene_strength"]) != 0.0:
        raise ValueError("Folds-0/1 adapter does not preserve the scene floor")
    if str(artifact["protocol_sha256"]) != training_protocol_sha256:
        raise ValueError("Folds-0/1 adapter training protocol identity differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["extractor"]["sha256"]:
        raise ValueError("Frozen folds-0/1 extractor hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen folds-0/1 input mismatch: {name}")
        paths[name] = path
    if not paths["metadata_root"].is_dir():
        raise ValueError("MARS metadata root is unavailable")
    verify_acquisition_receipt(
        paths["acquisition_receipt"],
        sha256(paths["manifest"]),
    )
    selected_folds = sorted(int(job["held_fold"]) for job in protocol["jobs"])
    if selected_folds != [0, 1]:
        raise ValueError("Folds-0/1 extraction jobs differ from [0,1]")
    artifact_hashes: set[str] = set()
    for job in protocol["jobs"]:
        artifact_path = (ROOT / job["artifact"]["path"]).resolve()
        artifact_hash = sha256(artifact_path)
        if artifact_hash != str(job["artifact"]["sha256"]):
            raise ValueError(f"Frozen folds-0/1 artifact mismatch: {artifact_path}")
        artifact = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
        )
        validate_multifold_artifact(
            artifact,
            held_fold=int(job["held_fold"]),
            fit_folds=list(map(int, job["fit_folds"])),
            mask_strength=float(protocol["runtime"]["mask_strength"]),
            training_protocol_sha256=sha256(paths["crossfit_protocol"]),
        )
        artifact_hashes.add(artifact_hash)
    if len(artifact_hashes) != 1:
        raise ValueError("Folds 0 and 1 do not share one frozen adapter")
    if args.smoke:
        print(
            json.dumps(
                {
                    "ok": True,
                    "held_folds": selected_folds,
                    "fit_folds": protocol["jobs"][0]["fit_folds"],
                    "artifact_sha256": next(iter(artifact_hashes)),
                    "feature_count": len(feature_names()),
                    "held_records_loaded": 0,
                    "held_labels_accessed": False,
                }
            )
        )
        return 0

    features, row_by_id, base_scores, feature_identity = load_feature_contract(
        paths["prithvi_features"],
        paths["prithvi_metadata"],
        paths["score_cache"],
        str(protocol["inputs"]["prithvi_features"]["sha256"]),
    )
    fold_protocol = json.loads(
        paths["fold_protocol"].read_text(encoding="utf-8")
    )
    group_to_fold = {
        str(row["group_id"]): int(row["fold"])
        for row in fold_protocol["assignments"]
    }
    records = [
        row
        for row in iter_development_manifest(paths["manifest"])
        if group_to_fold[str(row["group_id"])] in {0, 1}
    ]
    if not records:
        raise ValueError("No records selected for folds-0/1 extraction")
    sample_ids = [str(row["sample_id"]) for row in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Folds-0/1 extraction selected duplicate sample IDs")
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
        raise ValueError("Folds-0/1 exact base scores are invalid")
    groups = np.asarray([str(row["group_id"]) for row in records])
    folds = np.asarray(
        [group_to_fold[str(row["group_id"])] for row in records],
        dtype=np.uint8,
    )
    labels = np.full(len(records), 255, dtype=np.uint8)
    sensors = np.full(len(records), 255, dtype=np.uint8)
    names = feature_names()
    output_features = (ROOT / protocol["outputs"]["features"]).resolve()
    output_metadata = (ROOT / protocol["outputs"]["metadata"]).resolve()
    output_receipt = (ROOT / protocol["outputs"]["receipt"]).resolve()
    for path in (output_features, output_metadata, output_receipt):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    output_features.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_features.with_suffix(".tmp.npy")
    if temporary.exists():
        raise FileExistsError(
            f"Stale temporary folds-0/1 feature cache exists: {temporary}"
        )
    matrix = open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(len(records), len(names)),
    )
    written = np.zeros(len(records), dtype=bool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Folds-0/1 feature extraction requires CUDA")

    # Reuse the already exercised extraction kernel while substituting only the
    # explicit multi-holdout artifact validator defined above.
    crossfit.validate_artifact = validate_multifold_artifact
    jobs: list[dict[str, Any]] = []
    for job in protocol["jobs"]:
        jobs.append(
            crossfit.extract_job(
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
                smoke=False,
            )
        )
    if not written.all() or np.any(labels > 1) or np.any(sensors > 1):
        raise RuntimeError("Folds-0/1 feature extraction is incomplete")
    matrix.flush()
    del matrix
    feature_hash = sha256(temporary)
    os.replace(temporary, output_features)
    crossfit.stable_stat(output_features)
    if sha256(output_features) != feature_hash:
        raise RuntimeError("Folds-0/1 feature hash changed on promotion")
    adapter_by_fold = {
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
            json.dumps(adapter_by_fold, sort_keys=True)
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
    crossfit.stable_stat(output_metadata)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = {
        "schema_version": 1,
        "scope": "one-shot folds-0/1 dense-Prithvi representation cache",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(records),
        "feature_count": len(names),
        "folds": selected_folds,
        "features": {
            "path": output_features.relative_to(ROOT).as_posix(),
            "bytes": crossfit.stable_stat(output_features).st_size,
            "sha256": feature_hash,
        },
        "metadata": {
            "path": output_metadata.relative_to(ROOT).as_posix(),
            "bytes": crossfit.stable_stat(output_metadata).st_size,
            "sha256": sha256(output_metadata),
        },
        "jobs": jobs,
        "protocol_sha256": sha256(protocol_path),
        "extractor_sha256": sha256(Path(__file__).resolve()),
        "git_commit": commit,
        "feature_identity": feature_identity,
        "fold2_accessed": False,
        "fresh_inputs_accessed": False,
        "exact_paper_inputs_accessed": False,
    }
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = output_receipt.with_suffix(
        output_receipt.suffix + ".tmp"
    )
    temporary_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_receipt, output_receipt)
    print(json.dumps({"ok": True, **receipt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
