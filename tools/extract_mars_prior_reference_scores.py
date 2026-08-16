#!/usr/bin/env python3
"""Extract label-free released-MARS evidence for frozen prior references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from numpy.lib.format import open_memmap
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_released_marss2l import connected_scene_score  # noqa: E402
from mars_paper_model import ReleasedMarsUNet, released_state  # noqa: E402
from mars_s2l_adapter import (  # noqa: E402
    CLOUD_CLASSES,
    MARS_IMAGE_BANDS,
    compute_mbmp,
    safe_asset_path,
    validate_image_band_order,
)

DEFAULT_PROTOCOL = Path("configs/mars_prior_reference_score_extraction_protocol.json")
VIEW_NAMES = (
    "original_reference",
    "selected_reference_1",
    "selected_reference_2",
    "selected_reference_3",
    "selected_reference_4",
    "selected_reference_5",
)
VIEW_FEATURE_NAMES = (
    "connected_score",
    "top_25_mean",
    "top_100_mean",
    "top_500_mean",
    "clear_mean",
    "clear_std",
    "clear_area_above_0.1",
    "clear_area_above_0.3",
    "clear_area_above_0.5",
    "clear_area_above_0.7",
    "clear_area_above_0.9",
)
SAFE_RECORD_FIELDS = {
    "band_order",
    "group_id",
    "sample_id",
    "sensor_family",
    "target_datetime",
    "target_scene_id",
    "wind_u",
    "wind_v",
}
SAFE_ASSET_ROLES = {"image", "cloud_mask"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def safe_manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    """Retain only non-outcome metadata and the two permitted asset paths."""
    result = {key: record[key] for key in SAFE_RECORD_FIELDS}
    assets: dict[str, str] = {}
    for item in record["assets"]:
        role = str(item["role"])
        if role not in SAFE_ASSET_ROLES:
            continue
        if role in assets:
            raise ValueError(
                f"Duplicate safe asset role for {record['sample_id']}: {role}"
            )
        assets[role] = str(item["path"])
    if set(assets) != SAFE_ASSET_ROLES:
        raise ValueError(f"Missing safe input asset for {record['sample_id']}")
    result["assets"] = assets
    return result


def load_label_free_records(
    manifest_path: Path, group_to_fold: dict[str, int], selected_folds: set[int]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            group = str(raw["group_id"])
            if group not in group_to_fold:
                raise ValueError(f"Unknown group at manifest line {line_number}")
            fold = int(group_to_fold[group])
            if fold not in selected_folds:
                continue
            record = safe_manifest_record(raw)
            record["fold"] = fold
            records.append(record)
    return records


def load_selection_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("Selection manifest sample IDs are not unique")
    return rows


def grid_signature(source: rasterio.io.DatasetReader) -> tuple[Any, ...]:
    return (source.width, source.height, source.crs, tuple(source.transform)[:6])


@lru_cache(maxsize=512)
def read_image(
    path_value: str, declared_band_order: tuple[str, ...]
) -> tuple[np.ndarray, tuple[Any, ...]]:
    path = Path(path_value)
    with rasterio.open(path) as source:
        if source.count != 12 or set(source.dtypes) != {"uint16"}:
            raise ValueError(f"MARS image contract differs: {path}")
        validate_image_band_order(
            {"sample_id": path.name, "band_order": declared_band_order},
            tuple(source.descriptions),
        )
        values = source.read()
        grid = grid_signature(source)
    values.setflags(write=False)
    return values, grid


def read_cloud(path: Path, expected_grid: tuple[Any, ...]) -> np.ndarray:
    with rasterio.open(path) as source:
        if source.count != 1 or grid_signature(source) != expected_grid:
            raise ValueError(f"MARS cloud grid differs: {path}")
        cloud = source.read(1)
    unknown = set(map(int, np.unique(cloud))) - set(CLOUD_CLASSES)
    if unknown:
        raise ValueError(f"Unknown MARS cloud classes in {path}: {sorted(unknown)}")
    return cloud


def released_input(
    raw_target: np.ndarray,
    raw_reference: np.ndarray,
    wind: tuple[float, float],
    cloud: np.ndarray,
) -> np.ndarray:
    if (
        raw_target.shape != raw_reference.shape
        or raw_target.ndim != 3
        or raw_target.shape[0] != 6
        or cloud.shape != raw_target.shape[1:]
    ):
        raise ValueError("Released input arrays have incompatible shapes")
    spectral = np.clip(
        np.concatenate((raw_target, raw_reference), axis=0).astype(np.float32) / 5000.0,
        0.0,
        2.0,
    )
    spectral[~np.isfinite(spectral)] = 0.0
    mbmp = compute_mbmp(spectral[:6], spectral[6:])
    height, width = cloud.shape
    wind_values = np.asarray(wind, dtype=np.float32)
    if not np.isfinite(wind_values).all():
        raise ValueError("Wind values must be finite")
    wind_channels = np.broadcast_to(
        wind_values[:, None, None] / 8.0, (2, height, width)
    ).copy()
    result = np.concatenate(
        (mbmp[None], spectral, wind_channels, (cloud > 0).astype(np.float32)[None]),
        axis=0,
    ).astype(np.float32)
    if result.shape != (16, height, width) or not np.isfinite(result).all():
        raise ValueError("Released input is invalid")
    return result


def build_scene_views(
    row: dict[str, Any],
    selection: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    metadata_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    sample_id = str(row["sample_id"])
    if str(selection["sample_id"]) != sample_id:
        raise ValueError("Selection and target row are misaligned")
    target_path = safe_asset_path(metadata_root, row["assets"]["image"])
    pair, target_grid = read_image(
        str(target_path), tuple(row.get("band_order") or MARS_IMAGE_BANDS)
    )
    cloud = read_cloud(
        safe_asset_path(metadata_root, row["assets"]["cloud_mask"]), target_grid
    )
    target = pair[:6]
    references = [pair[6:]]
    for candidate_id in selection["selected_sample_ids"]:
        candidate = records_by_id[str(candidate_id)]
        candidate_path = safe_asset_path(metadata_root, candidate["assets"]["image"])
        candidate_pair, candidate_grid = read_image(
            str(candidate_path),
            tuple(candidate.get("band_order") or MARS_IMAGE_BANDS),
        )
        if candidate_grid != target_grid:
            raise ValueError(f"Selected reference grid drifted for {sample_id}")
        references.append(candidate_pair[:6])
    if len(references) > len(VIEW_NAMES):
        raise ValueError("Selection contains too many references")
    view_mask = np.zeros(len(VIEW_NAMES), dtype=bool)
    view_mask[: len(references)] = True
    references.extend([pair[6:]] * (len(VIEW_NAMES) - len(references)))
    inputs = np.stack(
        [
            released_input(
                target,
                reference,
                (float(row["wind_u"]), float(row["wind_v"])),
                cloud,
            )
            for reference in references
        ]
    )
    return inputs, view_mask


def probability_features(probability: torch.Tensor, clear: torch.Tensor) -> np.ndarray:
    if probability.ndim != 4 or probability.shape[1] != 1:
        raise ValueError("Probability tensor must be Bx1xHxW")
    if clear.shape != probability.shape or clear.dtype != torch.bool:
        raise ValueError("Clear mask must be a boolean probability-aligned tensor")
    flat = probability[:, 0].flatten(1)
    features: list[torch.Tensor] = []
    for count in (25, 100, 500):
        features.append(
            torch.topk(flat, min(count, flat.shape[1]), dim=1).values.mean(1)
        )
    count = clear.flatten(1).sum(1).clamp_min(1)
    mean = probability.flatten(1).sum(1) / count
    centered = torch.where(clear, probability - mean[:, None, None, None], 0.0)
    std = torch.sqrt(centered.square().flatten(1).sum(1) / count)
    areas = [
        ((probability > threshold) & clear).flatten(1).sum(1) / count
        for threshold in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    return (
        torch.stack([*features, mean, std, *areas], dim=1)
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def score_batch(
    model: ReleasedMarsUNet, inputs: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    rows, views, channels, height, width = inputs.shape
    tensor = torch.from_numpy(inputs.reshape(-1, channels, height, width)).to(
        device, non_blocking=True
    )
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(tensor)
    probability = torch.sigmoid(logits.float())
    clear = tensor[:, 15:16] <= 0.5
    probability = probability.masked_fill(~clear, 0.0)
    scalar = probability_features(probability, clear)
    connected = np.asarray(
        [connected_scene_score(value[0]) for value in probability.cpu().numpy()],
        dtype=np.float32,
    )
    scalar = np.concatenate((connected[:, None], scalar), axis=1)
    pooled = torch.stack(
        (
            F.adaptive_avg_pool2d(probability, (32, 32)),
            F.adaptive_max_pool2d(probability, (32, 32)),
        ),
        dim=1,
    )[:, :, 0]
    return (
        scalar.reshape(rows, views, len(VIEW_FEATURE_NAMES)),
        pooled.reshape(rows, views, 2, 32, 32).cpu().numpy().astype(np.float16),
    )


def load_released_score_parity(
    path: Path, selected_folds: set[int]
) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as cache:
        names = cache["feature_names"].astype(str)
        matches = np.flatnonzero(names == "released_connected_score")
        if matches.size != 1:
            raise ValueError("Released score feature is missing or ambiguous")
        ids = cache["sample_ids"].astype(str)
        folds = cache["folds"].astype(np.uint8)
        scores = cache["features"][:, int(matches[0])].astype(np.float64)
    selected = np.isin(folds, sorted(selected_folds))
    ids = ids[selected]
    scores = scores[selected]
    if len(set(ids.tolist())) != ids.size or not np.isfinite(scores).all():
        raise ValueError("Released parity scores are invalid")
    return dict(zip(ids.tolist(), scores.tolist(), strict=True))


def parity_summary(
    sample_ids: list[str], observed: np.ndarray, expected: dict[str, float]
) -> dict[str, Any]:
    target = np.asarray([expected[value] for value in sample_ids], dtype=np.float64)
    difference = np.abs(np.asarray(observed, dtype=np.float64) - target)
    return {
        "rows": len(sample_ids),
        "maximum_absolute_difference": float(np.max(difference)),
        "mean_absolute_difference": float(np.mean(difference)),
        "paper_rule_decisions_equal": bool(
            np.array_equal(np.asarray(observed) > 0.5, target > 0.5)
        ),
    }


def choose_smoke_indices(
    records: list[dict[str, Any]], selections: list[dict[str, Any]]
) -> list[int]:
    wanted = {"sentinel_five": 2, "sentinel_fallback": 1, "landsat": 1}
    chosen: list[int] = []
    counts = {key: 0 for key in wanted}
    for index, (record, selection) in enumerate(zip(records, selections, strict=True)):
        if record["sensor_family"] == "Landsat":
            key = "landsat"
        elif len(selection["selected_sample_ids"]) == 5:
            key = "sentinel_five"
        elif not selection["selected_sample_ids"]:
            key = "sentinel_fallback"
        else:
            continue
        if counts[key] < wanted[key]:
            chosen.append(index)
            counts[key] += 1
        if counts == wanted:
            break
    if counts != wanted:
        raise ValueError(f"Smoke strata are unavailable: {counts}")
    return chosen


def verify_protocol(protocol_path: Path, protocol: dict[str, Any]) -> dict[str, Path]:
    expected_script_hash = protocol["implementation"]["script"]["sha256"]
    if (
        expected_script_hash != "DRAFT"
        and sha256(Path(__file__).resolve()) != expected_script_hash
    ):
        raise ValueError("Frozen extractor hash mismatch")
    for dependency in protocol["implementation"]["code_dependencies"]:
        if sha256(repo_path(dependency["path"])) != dependency["sha256"]:
            raise ValueError(f"Frozen code dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = repo_path(contract["path"])
        if contract["sha256"] != "directory_verified_by_acquisition_receipt":
            if sha256(path) != contract["sha256"]:
                raise ValueError(f"Frozen extraction input hash mismatch: {name}")
        paths[name] = path
    return paths


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = repo_path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    paths = verify_protocol(protocol_path, protocol)
    selected_folds = set(map(int, protocol["scope"]["folds"]))
    fold_protocol = json.loads(paths["fold_protocol"].read_text(encoding="utf-8"))
    group_to_fold = {
        str(item["group_id"]): int(item["fold"])
        for item in fold_protocol["assignments"]
    }
    records = load_label_free_records(
        paths["development_manifest"], group_to_fold, selected_folds
    )
    selections = load_selection_rows(paths["selection_manifest"])
    if [row["sample_id"] for row in records] != [
        row["sample_id"] for row in selections
    ]:
        raise ValueError("Selection manifest order differs from development rows")
    records_by_id = {str(row["sample_id"]): row for row in records}
    if len(records_by_id) != len(records):
        raise ValueError("Development sample IDs are not unique")
    parity_scores = load_released_score_parity(
        paths["released_score_cache"], selected_folds
    )
    if set(records_by_id) != set(parity_scores):
        raise ValueError("Released parity cache differs from selected row identities")

    if args.smoke:
        smoke_indices = choose_smoke_indices(records, selections)
        records = [records[index] for index in smoke_indices]
        selections = [selections[index] for index in smoke_indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not bool(protocol["runtime"]["allow_cpu"]):
        raise RuntimeError("Prior-reference extraction requires CUDA")
    model = ReleasedMarsUNet().to(device)
    incompatible = model.load_state_dict(
        released_state(paths["released_checkpoint"]), strict=False
    )
    if incompatible.missing_keys or any(
        not key.startswith("out_mlp.") for key in incompatible.unexpected_keys
    ):
        raise ValueError(f"Released checkpoint compatibility differs: {incompatible}")
    model.eval()

    rows = len(records)
    batch_size = rows if args.smoke else int(protocol["runtime"]["target_batch_size"])
    workers = int(protocol["runtime"]["raster_workers"])
    output_maps = repo_path(protocol["outputs"]["pooled_probability_maps"])
    temporary_maps = output_maps.with_suffix(".tmp.npy")
    maps = None
    if not args.smoke:
        output_maps.parent.mkdir(parents=True, exist_ok=True)
        maps = open_memmap(
            temporary_maps,
            mode="w+",
            dtype=np.float16,
            shape=(rows, len(VIEW_NAMES), 2, 32, 32),
        )
    all_features = np.zeros(
        (rows, len(VIEW_NAMES), len(VIEW_FEATURE_NAMES)), dtype=np.float32
    )
    all_masks = np.zeros((rows, len(VIEW_NAMES)), dtype=bool)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, rows, batch_size):
            end = min(start + batch_size, rows)
            extracted = list(
                executor.map(
                    lambda pair: build_scene_views(
                        pair[0], pair[1], records_by_id, paths["metadata_root"]
                    ),
                    zip(records[start:end], selections[start:end], strict=True),
                )
            )
            input_batch = np.stack([value[0] for value in extracted])
            mask_batch = np.stack([value[1] for value in extracted])
            feature_batch, map_batch = score_batch(model, input_batch, device)
            feature_batch[~mask_batch] = 0.0
            map_batch[~mask_batch] = 0.0
            all_features[start:end] = feature_batch
            all_masks[start:end] = mask_batch
            if maps is not None:
                maps[start:end] = map_batch
            if (
                end % int(protocol["runtime"]["progress_every_rows"]) == 0
                or end == rows
            ):
                if maps is not None:
                    maps.flush()
                print(json.dumps({"rows": end}), flush=True)

    sample_ids = [str(row["sample_id"]) for row in records]
    parity = parity_summary(sample_ids, all_features[:, 0, 0], parity_scores)
    gates = protocol["original_view_parity"]
    parity["pass"] = bool(
        parity["maximum_absolute_difference"]
        <= float(gates["maximum_absolute_difference"])
        and parity["mean_absolute_difference"]
        <= float(gates["maximum_mean_absolute_difference"])
        and parity["paper_rule_decisions_equal"]
    )
    if not parity["pass"]:
        if maps is not None:
            del maps
            temporary_maps.unlink(missing_ok=True)
        raise RuntimeError(f"Original-view parity failed: {parity}")
    if args.smoke:
        print(
            json.dumps(
                {
                    "ok": True,
                    "smoke": True,
                    "rows": rows,
                    "device": torch.cuda.get_device_name(device),
                    "parity": parity,
                    "view_masks": all_masks.astype(int).tolist(),
                },
                indent=2,
            )
        )
        return 0

    assert maps is not None
    maps.flush()
    del maps
    os.replace(temporary_maps, output_maps)
    output_scores = repo_path(protocol["outputs"]["scores"])
    selected_ids = np.full((rows, len(VIEW_NAMES) - 1), "", dtype="<U36")
    selected_distances = np.full((rows, len(VIEW_NAMES) - 1), np.nan, dtype=np.float32)
    for index, selection in enumerate(selections):
        count = len(selection["selected_sample_ids"])
        selected_ids[index, :count] = selection["selected_sample_ids"]
        selected_distances[index, :count] = selection["selected_distances"]
    atomic_savez(
        output_scores,
        schema_version=np.asarray(1, dtype=np.uint8),
        sample_ids=np.asarray(sample_ids),
        folds=np.asarray([int(row["fold"]) for row in records], dtype=np.uint8),
        sensors=np.asarray([str(row["sensor_family"]) for row in records]),
        view_names=np.asarray(VIEW_NAMES),
        view_mask=all_masks,
        view_feature_names=np.asarray(VIEW_FEATURE_NAMES),
        view_features=all_features,
        selected_sample_ids=selected_ids,
        selected_distances=selected_distances,
        original_view_parity_scores=np.asarray(
            [parity_scores[value] for value in sample_ids], dtype=np.float32
        ),
        protocol_sha256=np.asarray(sha256(protocol_path)),
        selection_manifest_sha256=np.asarray(sha256(paths["selection_manifest"])),
        released_checkpoint_sha256=np.asarray(sha256(paths["released_checkpoint"])),
        pooled_probability_maps_sha256=np.asarray(sha256(output_maps)),
    )
    receipt_path = repo_path(protocol["outputs"]["receipt"])
    receipt = {
        "schema_version": 1,
        "status": "complete_label_free_prior_reference_released_score_extraction",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "folds": sorted(selected_folds),
        "views": list(VIEW_NAMES),
        "view_features": list(VIEW_FEATURE_NAMES),
        "reference_coverage": {
            "rows_with_alternate_reference": int(np.sum(all_masks[:, 1:].any(axis=1))),
            "rows_with_five_alternate_references": int(
                np.sum(all_masks[:, 1:].sum(axis=1) == 5)
            ),
        },
        "original_view_parity": parity,
        "outcome_access": {
            "labels_accessed": False,
            "plume_masks_accessed": False,
            "methane_enhancement_accessed": False,
            "model_selection_performed": False,
            "permitted_mixed_cache_fields_accessed": [
                "feature_names",
                "features[released_connected_score only]",
                "folds",
                "sample_ids",
            ],
        },
        "outputs": {
            "scores": {
                "path": output_scores.relative_to(ROOT).as_posix(),
                "bytes": output_scores.stat().st_size,
                "sha256": sha256(output_scores),
                "tracked": False,
            },
            "pooled_probability_maps": {
                "path": output_maps.relative_to(ROOT).as_posix(),
                "bytes": output_maps.stat().st_size,
                "sha256": sha256(output_maps),
                "tracked": False,
            },
        },
        "provenance": {
            "protocol": {
                "path": protocol_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(protocol_path),
            },
            "script": {
                "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, **receipt["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
