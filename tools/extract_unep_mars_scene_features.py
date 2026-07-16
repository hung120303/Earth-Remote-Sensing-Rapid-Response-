#!/usr/bin/env python3
"""Extract the frozen MARS scene-feature schema for nonsealed UNEP positives."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from evaluate_mars_residual_endpoint_blend import (  # noqa: E402
    load_residual_model,
    trust_region_logits,
)
from evaluate_released_marss2l import connected_scene_score  # noqa: E402
from extract_mars_scene_features import (  # noqa: E402
    atomic_savez,
    pooled_scene_features,
    tensor_feature_names,
)
from mars_s2l_adapter import iter_manifest  # noqa: E402
from train_mars_paper_residual import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    MarsPaperDataset,
    move_batch,
)


DEFAULT_ARTIFACT = Path(
    "EarthRemoteSensingRapidResponse/artifacts/mars_paper_residual_fold0_seed606.pt"
)
DEFAULT_ARTIFACT_SHA256 = (
    "b94880d858e1e7791591eeb5f7d0da9be84b99a324e980437ebe83cfae6c7d49"
)
ROLE_CONTRACTS = {
    "auxiliary_training": {
        "manifest": Path(".research/unep_mars_post2024/model_auxiliary_training.jsonl"),
        "manifest_sha256": "2971ebad317c7f709a677e5c6431a75804fce4026bbcf28fa50ce5bb4cc89300",
        "rows": 135,
        "output": Path("outputs/unep_mars_post2024_scene_features_auxiliary.npz"),
    },
    "development": {
        "manifest": Path(".research/unep_mars_post2024/model_development.jsonl"),
        "manifest_sha256": "5a17c24cb7a78941ed8586cbf99813c3f4772040d5c4bfb9377b2d3a675e1741",
        "rows": 4,
        "output": Path("outputs/unep_mars_post2024_scene_features_development.npz"),
    },
}
DEFAULT_JSON = Path("reports/acquisition/unep_mars_post2024_scene_features.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/UNEP_MARS_POST2024_SCENE_FEATURES.md")


def validate_records(records: list[dict[str, Any]], role: str, expected_rows: int) -> None:
    if len(records) != expected_rows:
        raise ValueError(f"{role} row count differs from the frozen contract")
    if any(record.get("research_role") != role for record in records):
        raise ValueError(f"{role} manifest contains a different research role")
    if any(record.get("label_state") != "PLUME" for record in records):
        raise ValueError(f"{role} feature extraction is positive-only")
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError(f"{role} manifest contains duplicate samples")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# UNEP MARS post-2024 scene features",
        "",
        "Frozen released/residual scene features were extracted without fitting or model selection.",
        "",
        "| Role | Rows | Groups | Features | Cache SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for role, result in report["roles"].items():
        lines.append(
            f"| {role} | {result['rows']} | {result['groups']} | "
            f"{result['features']} | `{result['output_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "The auxiliary and development caches remain physically separate. Sealed-external rows and paper-test imagery were not accessed.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT.as_posix())
    parser.add_argument("--artifact-sha256", default=DEFAULT_ARTIFACT_SHA256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")

    artifact_path = (ROOT / args.artifact).resolve()
    if sha256(artifact_path) != args.artifact_sha256:
        raise ValueError("Residual artifact hash mismatch")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    checkpoint = (ROOT / args.released_checkpoint).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_residual_model(checkpoint, artifact, device)
    feature_names = [
        "primary_connected_score",
        "released_connected_score",
        *tensor_feature_names(),
    ]
    results: dict[str, Any] = {}
    for role, contract in ROLE_CONTRACTS.items():
        manifest = (ROOT / contract["manifest"]).resolve()
        if sha256(manifest) != contract["manifest_sha256"]:
            raise ValueError(f"{role} manifest hash mismatch")
        records = list(iter_manifest(manifest))
        validate_records(records, role, int(contract["rows"]))
        loader = DataLoader(
            MarsPaperDataset(ROOT, records, augment=False, seed=0),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        rows: list[np.ndarray] = []
        sample_ids: list[str] = []
        groups: list[str] = []
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model(
                    batch["inputs"], batch["observable"], batch["sensor_index"]
                )
            primary_logits = trust_region_logits(
                output["baseline_logits"], output["segmentation_logits"], 0.5
            )
            tensor = pooled_scene_features(
                batch["inputs"],
                primary_logits,
                output["baseline_logits"],
                batch["clear"],
                batch["observable"],
            ).cpu().numpy()
            primary_probability = torch.sigmoid(primary_logits).float().masked_fill(
                batch["clear"] <= 0.5, 0.0
            ).cpu().numpy()
            released_probability = torch.sigmoid(output["baseline_logits"]).float().masked_fill(
                batch["clear"] <= 0.5, 0.0
            ).cpu().numpy()
            for index in range(tensor.shape[0]):
                connected = np.asarray(
                    [
                        connected_scene_score(primary_probability[index, 0]),
                        connected_scene_score(released_probability[index, 0]),
                    ],
                    dtype=np.float32,
                )
                rows.append(np.concatenate((connected, tensor[index])).astype(np.float32))
                sample_ids.append(str(batch["sample_id"][index]))
                groups.append(str(batch["group_id"][index]))
        output = (ROOT / contract["output"]).resolve()
        atomic_savez(
            output,
            features=np.stack(rows),
            feature_names=np.asarray(feature_names),
            labels=np.ones(len(rows), dtype=np.uint8),
            sensors=np.zeros(len(rows), dtype=np.uint8),
            sample_ids=np.asarray(sample_ids),
            groups=np.asarray(groups),
            research_role=np.asarray(role),
            artifact_sha256=np.asarray(args.artifact_sha256),
            checkpoint_sha256=np.asarray(sha256(checkpoint)),
            manifest_sha256=np.asarray(contract["manifest_sha256"]),
        )
        results[role] = {
            "rows": len(rows),
            "groups": len(set(groups)),
            "features": len(feature_names),
            "manifest_sha256": contract["manifest_sha256"],
            "output": output.relative_to(ROOT).as_posix(),
            "output_bytes": output.stat().st_size,
            "output_sha256": sha256(output),
        }
        print(json.dumps({"role": role, **results[role]}), flush=True)

    report = {
        "schema_version": 1,
        "status": "nonsealed frozen scene features extracted",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "roles": results,
        "contract": {
            "feature_schema": feature_names,
            "artifact_sha256": args.artifact_sha256,
            "checkpoint_sha256": sha256(checkpoint),
            "trust_region_strength": 0.5,
            "sealed_external_accessed": False,
            "paper_test_accessed": False,
        },
        "provenance": {
            "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "device": str(
                torch.cuda.get_device_name(device) if device.type == "cuda" else device
            ),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    output_json = (ROOT / args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown((ROOT / args.output_markdown).resolve(), report)
    print(json.dumps({"ok": True, "roles": list(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
