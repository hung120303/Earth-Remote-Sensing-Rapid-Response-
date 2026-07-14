#!/usr/bin/env python3
"""Reconstruct and freeze the exact MARS-S2L paper-v3 benchmark identity.

The paper's headline metrics splice the released onshore model with the
fine-tuned offshore model.  The Hub contains both per-scene prediction files
at an historical revision, even though one filename uses ``th100`` rather
than ``thr100``.  The prediction files are the authoritative identity for the
43,529 evaluated scenes and their paper-era labels; later public metadata
revisions contain 43,524 scenes and 1,832 positives instead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score

from acquire_mars_metadata import repo_root, sha256

PAPER_URL = "https://arxiv.org/html/2511.21777v3"
PAPER_REVISION_DATE = "2026-04-24"
UPSTREAM_CODE_REVISION = "f7d264c2c845dfba1cb27f76ef6026275f8d8758"
ARCHIVE_REVISION = "1a722445d7e344a1581ade9741aa41fb707c6d0e"
PUBLIC_METADATA_REVISION = "8ebb807c5b055ee98ff1039cd9e8f4b3e92c6e73"

DATA_ROOT = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
    "publication-v1/external/MARS-S2L-paper-source"
)
DEFAULT_METADATA = DATA_ROOT / "validated_images_all_20250704.csv"
DEFAULT_ONSHORE = DATA_ROOT / "mars_onshore_preds_test_2023th100.csv"
DEFAULT_OFFSHORE = DATA_ROOT / "mars_offshore_preds_test_2023thr100.csv"
DEFAULT_CONFIG = Path(
    "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/"
    "publication-v1/external/MARS-S2L/trained_models/"
    "MARSS2L_20250326/config_experiment.json"
)
DEFAULT_ASSIGNMENTS = DATA_ROOT / "paper_v3_benchmark_assignments.jsonl"
DEFAULT_JSON = Path("reports/acquisition/mars_s2l_paper_v3_benchmark.json")
DEFAULT_MARKDOWN = Path("reports/acquisition/MARS_S2L_PAPER_V3_BENCHMARK.md")

EXPECTED_HASHES = {
    "metadata": "1ab3beb83d9c062fa5b6e5c07fb2cd9ceac54b353149428e71ec43457539891f",
    "onshore_predictions": "396c6af6d7ae4afa122eac271347889e6137d11e09ff434a426d435c23874c7e",
    "offshore_predictions": "42b6fa72fa5825c00355baeb129c2a74dd15f40a5b11cb5e8e1b31a7bf042525",
    "released_config": "abeb92d01313fbb2939e6c5fc1c6281846b8102ea5edd7081668fe0db05bf79f",
}

PUBLISHED = {
    "full": {
        "rows": 43_529,
        "positive": 1_813,
        "negative": 41_716,
        "sites": 1_289,
        "average_precision": 0.6408,
        "precision": 0.3253,
        "recall": 0.7915,
        "false_positive_rate": 0.0713,
        "pixel_iou": 0.3224,
    },
    "test_only_sites": {
        "rows": 15_655,
        "positive": 227,
        "negative": 15_428,
        "sites": 697,
        "average_precision": 0.4496,
        "precision": 0.1301,
        "recall": 0.7753,
        "false_positive_rate": 0.0763,
    },
}


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def index_unique(rows: Iterable[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = row
    return result


def scene_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([int(row["target"]) for row in rows], dtype=np.uint8)
    scores = np.asarray([float(row["scene_pred"]) for row in rows], dtype=np.float64)
    predictions = scores > 0.5
    tp = int(np.sum((labels == 1) & predictions))
    fp = int(np.sum((labels == 0) & predictions))
    tn = int(np.sum((labels == 0) & ~predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    pixel_tp = int(sum(float(row["TP"]) for row in rows))
    pixel_fp = int(sum(float(row["FP"]) for row in rows))
    pixel_fn = int(sum(float(row["FN"]) for row in rows))
    return {
        "rows": len(rows),
        "positive": int(np.sum(labels)),
        "negative": int(np.sum(labels == 0)),
        "sites": len({str(row["location_name"]).strip() for row in rows}),
        "average_precision": float(average_precision_score(labels, scores)),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "accuracy": (tp + tn) / len(rows),
        "false_positive_rate": fp / (fp + tn),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "pixel_iou": pixel_tp / (pixel_tp + pixel_fp + pixel_fn),
        "pixel_tp": pixel_tp,
        "pixel_fp": pixel_fp,
        "pixel_fn": pixel_fn,
    }


def metric_delta(observed: dict[str, Any], published: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(observed[key]) - float(value)
        for key, value in published.items()
        if key in observed and isinstance(value, float)
    }


def reconstruct(
    metadata_path: Path,
    onshore_path: Path,
    offshore_path: Path,
    config_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for name, path in {
        "metadata": metadata_path,
        "onshore_predictions": onshore_path,
        "offshore_predictions": offshore_path,
        "released_config": config_path,
    }.items():
        observed = sha256(path)
        if observed != EXPECTED_HASHES[name]:
            raise ValueError(f"{name} hash mismatch: expected {EXPECTED_HASHES[name]}, got {observed}")

    metadata = index_unique(read_csv(metadata_path), "id_loc_image")
    onshore = read_csv(onshore_path)
    offshore = index_unique(read_csv(offshore_path), "id_loc_image")
    if len(onshore) != PUBLISHED["full"]["rows"] or len(offshore) != len(onshore):
        raise ValueError("Archived prediction files do not contain the paper's 43,529 rows")
    if set(row["id_loc_image"] for row in onshore) != set(offshore):
        raise ValueError("Onshore and offshore prediction archives contain different scene IDs")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    training_sites = {str(value).strip() for value in config["all_locs_train"]}
    combined: list[dict[str, Any]] = []
    missing_metadata: list[str] = []
    metadata_label_disagreements = 0
    offshore_replacements = 0
    for base in onshore:
        sample_id = base["id_loc_image"]
        meta = metadata.get(sample_id)
        is_offshore = bool(meta and parse_bool(meta["offshore"]))
        selected = offshore[sample_id] if is_offshore else base
        if is_offshore:
            offshore_replacements += 1
        if meta is None:
            missing_metadata.append(sample_id)
        elif int(selected["target"]) != int(parse_bool(meta["isplume"])):
            metadata_label_disagreements += 1
        location = str(selected["location_name"]).strip()
        combined.append(
            {
                **selected,
                "location_name": location,
                "target": int(selected["target"]),
                "is_offshore": is_offshore,
                "test_only_site": location not in training_sites,
                "baseline_source": "offshore_finetune" if is_offshore else "general_model",
                "public_metadata_available": meta is not None,
            }
        )
    combined.sort(key=lambda row: str(row["id_loc_image"]))

    full = scene_metrics(combined)
    test_only = scene_metrics([row for row in combined if row["test_only_site"]])
    required_exact = {
        "full_rows": full["rows"] == PUBLISHED["full"]["rows"],
        "full_positive": full["positive"] == PUBLISHED["full"]["positive"],
        "full_sites": full["sites"] == PUBLISHED["full"]["sites"],
        "test_only_rows": test_only["rows"] == PUBLISHED["test_only_sites"]["rows"],
        "test_only_positive": test_only["positive"] == PUBLISHED["test_only_sites"]["positive"],
        "test_only_sites": test_only["sites"] == PUBLISHED["test_only_sites"]["sites"],
        "full_recall_rounds_to_paper": round(full["recall"], 4) == PUBLISHED["full"]["recall"],
        "test_only_recall_rounds_to_paper": round(test_only["recall"], 4)
        == PUBLISHED["test_only_sites"]["recall"],
    }
    if not all(required_exact.values()):
        raise ValueError(f"Paper benchmark identity failed: {required_exact}")
    audit = {
        "full": full,
        "test_only_sites": test_only,
        "published": PUBLISHED,
        "delta_vs_published": {
            "full": metric_delta(full, PUBLISHED["full"]),
            "test_only_sites": metric_delta(test_only, PUBLISHED["test_only_sites"]),
        },
        "identity_checks": required_exact,
        "offshore_replacements": offshore_replacements,
        "public_metadata_rows": len(combined) - len(missing_metadata),
        "missing_public_metadata_rows": len(missing_metadata),
        "missing_public_metadata_ids": sorted(missing_metadata),
        "paper_target_vs_public_metadata_label_disagreements": metadata_label_disagreements,
        "training_site_count_from_released_config": len(training_sites),
    }
    return combined, audit


def assignment_line(row: dict[str, Any]) -> str:
    compact = {
        "sample_id": row["id_loc_image"],
        "location_name": row["location_name"],
        "tile": row["tile"],
        "target": row["target"],
        "test_only_site": row["test_only_site"],
        "is_offshore": row["is_offshore"],
        "public_metadata_available": row["public_metadata_available"],
        "baseline_source": row["baseline_source"],
        "baseline_scene_score": float(row["scene_pred"]),
        "baseline_pixel_tp": int(float(row["TP"])),
        "baseline_pixel_fp": int(float(row["FP"])),
        "baseline_pixel_tn": int(float(row["TN"])),
        "baseline_pixel_fn": int(float(row["FN"])),
    }
    return json.dumps(compact, sort_keys=True, separators=(",", ":")) + "\n"


def write_assignments(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(assignment_line(row))
    os.replace(temporary, path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    audit = report["reconstruction"]
    full = audit["full"]
    unseen = audit["test_only_sites"]
    lines = [
        "# MARS-S2L paper-v3 benchmark lock",
        "",
        "The paper's general model is used on onshore scenes and its fine-tuned model on offshore scenes. The historical per-scene archives, not the later public metadata labels, define the evaluated cohort.",
        "",
        "| View | Scenes | Plume | Sites | AP | Recall | FPR | Pixel IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Reconstructed full | {full['rows']:,} | {full['positive']:,} | {full['sites']:,} | {full['average_precision']:.4f} | {full['recall']:.4f} | {full['false_positive_rate']:.4f} | {full['pixel_iou']:.4f} |",
        f"| Paper Table S6 | 43,529 | 1,813 | 1,289 | 0.6408 | 0.7915 | 0.0713 | 0.3224 |",
        f"| Reconstructed test-only | {unseen['rows']:,} | {unseen['positive']:,} | {unseen['sites']:,} | {unseen['average_precision']:.4f} | {unseen['recall']:.4f} | {unseen['false_positive_rate']:.4f} | {unseen['pixel_iou']:.4f} |",
        f"| Paper Table S5 | 15,655 | 227 | 697 | 0.4496 | 0.7753 | 0.0763 | not reported |",
        "",
        f"- Assignment SHA-256: `{report['artifacts']['assignment_sha256']}`",
        f"- Public raster coverage: {audit['public_metadata_rows']:,} / {full['rows']:,}; {audit['missing_public_metadata_rows']} historical rows have predictions but no released metadata row.",
        f"- Paper-era targets disagree with the July 2025 public metadata label on {audit['paper_target_vs_public_metadata_label_disagreements']} available scenes.",
        "- Exact scene, positive, site, unseen-site, and recall counts reproduce. Small residual AP/FPR/IoU differences are retained explicitly and never rounded into an exact reproduction claim.",
        "",
        "## Superiority gate",
        "",
        "A successor must beat both the paper table and the reconstructed per-scene comparator. On full and test-only views, paired site-bootstrap 95% confidence intervals must show AP and IoU improvements; recall must improve while FPR is no worse. The two historical scenes without public rasters are scored adversarially for the candidate (positive as a miss, negative as a false alarm, and worst-case pixel error). Test outputs remain sealed until architecture, ensemble, and thresholds are frozen.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=DEFAULT_METADATA.as_posix())
    parser.add_argument("--onshore-predictions", default=DEFAULT_ONSHORE.as_posix())
    parser.add_argument("--offshore-predictions", default=DEFAULT_OFFSHORE.as_posix())
    parser.add_argument("--config", default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--assignments", default=DEFAULT_ASSIGNMENTS.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    paths = {
        name: (root / value).resolve()
        for name, value in {
            "metadata": args.metadata,
            "onshore": args.onshore_predictions,
            "offshore": args.offshore_predictions,
            "config": args.config,
            "assignments": args.assignments,
            "json": args.output_json,
            "markdown": args.output_markdown,
        }.items()
    }
    rows, audit = reconstruct(paths["metadata"], paths["onshore"], paths["offshore"], paths["config"])
    expected_assignment_hash = hashlib.sha256(
        "".join(assignment_line(row) for row in rows).encode("utf-8")
    ).hexdigest()
    if args.verify_only:
        if not paths["assignments"].is_file() or not paths["json"].is_file():
            raise FileNotFoundError("Verification requires frozen assignments and tracked JSON")
        report = json.loads(paths["json"].read_text(encoding="utf-8"))
        if sha256(paths["assignments"]) != expected_assignment_hash:
            raise ValueError("Frozen paper assignment identity changed")
        if report["artifacts"]["assignment_sha256"] != expected_assignment_hash:
            raise ValueError("Tracked benchmark references a different assignment identity")
        print(json.dumps({"ok": True, "assignment_sha256": expected_assignment_hash, "identity_checks": audit["identity_checks"]}, sort_keys=True))
        return 0

    write_assignments(paths["assignments"], rows)
    report = {
        "schema_version": 1,
        "status": "frozen_before_exact_mixed_sensor_training",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper": {"url": PAPER_URL, "revision": "v3", "revision_date": PAPER_REVISION_DATE},
        "source": {
            "dataset": "UNEP-IMEO/MARS-S2L",
            "public_metadata_revision": PUBLIC_METADATA_REVISION,
            "prediction_archive_revision": ARCHIVE_REVISION,
            "upstream_code_revision": UPSTREAM_CODE_REVISION,
            "hashes": EXPECTED_HASHES,
        },
        "reconstruction": audit,
        "superiority_gate": {
            "views": ["full_43529", "test_only_sites_15655"],
            "must_exceed": ["paper_published_table", "paired_reconstructed_archive"],
            "primary": ["average_precision", "pixel_iou"],
            "operating_point": "recall higher and false_positive_rate no worse",
            "uncertainty": "paired site-block bootstrap, 10000 replicates, two-sided 95% intervals",
            "missing_raster_policy": "candidate adversarial: positive miss, negative false alarm, worst-case pixel error",
            "selection_data": "training and development/validation only",
            "test_tuning": "prohibited; one-shot after model, ensemble, calibration, threshold, and postprocessing hashes are frozen",
        },
        "artifacts": {
            "assignment_path": paths["assignments"].relative_to(root).as_posix(),
            "assignment_sha256": sha256(paths["assignments"]),
            "assignment_rows": len(rows),
            "assignment_tracked": False,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "provenance": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "script": "tools/audit_mars_paper_benchmark.py",
            "script_sha256": sha256(Path(__file__)),
        },
    }
    write_json(paths["json"], report)
    write_markdown(paths["markdown"], report)
    print(json.dumps({"ok": True, "assignment_sha256": expected_assignment_hash, "full": audit["full"], "test_only_sites": audit["test_only_sites"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
