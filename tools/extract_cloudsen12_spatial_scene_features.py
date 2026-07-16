#!/usr/bin/env python3
"""Extract frozen model-aligned scene features for CloudSEN12+ pilot negatives."""

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
    MarsPaperDataset,
    move_batch,
)


DEFAULT_PROTOCOL = Path("configs/mars_cloudsen12_spatial_scene_features_protocol.json")
DEFAULT_JSON = Path("reports/acquisition/cloudsen12_spatial_scene_features.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/CLOUDSEN12_SPATIAL_SCENE_FEATURES.md")


def validate_records(records: list[dict[str, Any]], role: str, expected_rows: int) -> None:
    if len(records) != expected_rows:
        raise ValueError(f"{role} row count differs from the frozen contract")
    if any(record.get("research_role") != role for record in records):
        raise ValueError(f"{role} manifest contains a different research role")
    if any(record.get("label_state") != "NO_PLUME" for record in records):
        raise ValueError(f"{role} feature extraction is negative-only")
    if any(record.get("sensor_family") != "Sentinel-2" for record in records):
        raise ValueError(f"{role} feature extraction is Sentinel-2-only")
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError(f"{role} manifest contains duplicate samples")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    fresh_test = set(report["roles"]) == {"fresh_external_test"}
    lines = [
        (
            "# CloudSEN12+ fresh-test scene features"
            if fresh_test
            else "# CloudSEN12+ spatial-pilot scene features"
        ),
        "",
        "The released detector and frozen residual representation extracted scene features without fitting or model selection.",
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
            (
                "The fixed representation was applied once to the fresh published test rows; the exact MARS paper-test cache was not accessed."
                if fresh_test
                else "Auxiliary and development caches remain physically separate. Published CloudSEN12+ test rows and MARS paper-test imagery were not accessed."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")

    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["extractor"]["sha256"]:
        raise ValueError("Scene-feature extractor hash mismatch")
    source_receipt = (ROOT / protocol["source_receipt"]["path"]).resolve()
    if sha256(source_receipt) != protocol["source_receipt"]["sha256"]:
        raise ValueError("Source receipt hash mismatch")
    model_contract = protocol["model_contract"]
    artifact_path = (ROOT / model_contract["residual_artifact"]["path"]).resolve()
    if sha256(artifact_path) != model_contract["residual_artifact"]["sha256"]:
        raise ValueError("Residual artifact hash mismatch")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    checkpoint = (ROOT / model_contract["released_checkpoint"]["path"]).resolve()
    if sha256(checkpoint) != model_contract["released_checkpoint"]["sha256"]:
        raise ValueError("Released checkpoint hash mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_residual_model(checkpoint, artifact, device)
    model.eval()
    trust_region_strength = float(model_contract["trust_region_strength"])
    feature_names = [
        "primary_connected_score",
        "released_connected_score",
        *tensor_feature_names(),
    ]
    results: dict[str, Any] = {}
    roles = list(protocol["manifests"])
    for role in roles:
        contract = protocol["manifests"][role]
        manifest = (ROOT / contract["path"]).resolve()
        if sha256(manifest) != contract["sha256"]:
            raise ValueError(f"{role} manifest hash mismatch")
        records = list(iter_manifest(manifest))
        validate_records(records, role, int(contract["rows"]))
        record_by_id = {record["sample_id"]: record for record in records}
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
                output["baseline_logits"],
                output["segmentation_logits"],
                trust_region_strength,
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
            released_probability = (
                torch.sigmoid(output["baseline_logits"])
                .float()
                .masked_fill(batch["clear"] <= 0.5, 0.0)
                .cpu()
                .numpy()
            )
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
        output_path = (ROOT / contract["feature_output"]).resolve()
        atomic_savez(
            output_path,
            features=np.stack(rows),
            feature_names=np.asarray(feature_names),
            labels=np.zeros(len(rows), dtype=np.uint8),
            sensors=np.zeros(len(rows), dtype=np.uint8),
            sample_ids=np.asarray(sample_ids),
            groups=np.asarray(groups),
            published_all_clear=np.asarray(
                [record_by_id[sample_id].get("published_all_clear", True) for sample_id in sample_ids],
                dtype=np.bool_,
            ),
            published_nonclear_pixels=np.asarray(
                [record_by_id[sample_id].get("published_nonclear_pixels", 0) for sample_id in sample_ids],
                dtype=np.int32,
            ),
            research_role=np.asarray(role),
            artifact_sha256=np.asarray(model_contract["residual_artifact"]["sha256"]),
            checkpoint_sha256=np.asarray(model_contract["released_checkpoint"]["sha256"]),
            manifest_sha256=np.asarray(contract["sha256"]),
        )
        results[role] = {
            "rows": len(rows),
            "groups": len(set(groups)),
            "features": len(feature_names),
            "manifest_sha256": contract["sha256"],
            "output": output_path.relative_to(ROOT).as_posix(),
            "output_bytes": output_path.stat().st_size,
            "output_sha256": sha256(output_path),
        }
        print(json.dumps({"role": role, **results[role]}), flush=True)

    report = {
        "schema_version": 1,
        "status": (
            "fresh external-test frozen negative scene features extracted"
            if set(roles) == {"fresh_external_test"}
            else "nonsealed frozen negative scene features extracted"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "roles": results,
        "contract": {
            "feature_schema": feature_names,
            "residual_artifact_sha256": model_contract["residual_artifact"]["sha256"],
            "checkpoint_sha256": model_contract["released_checkpoint"]["sha256"],
            "trust_region_strength": trust_region_strength,
            "cloudsen12_test_accessed": set(roles) == {"fresh_external_test"},
            "paper_test_accessed": False,
        },
        "provenance": {
            "protocol": args.protocol,
            "protocol_sha256": sha256(protocol_path),
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
