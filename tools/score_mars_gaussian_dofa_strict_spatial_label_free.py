#!/usr/bin/env python3
"""Candidate-specific label-free Gaussian+DOFA replay on MARS strict spatial rows.

This scorer deliberately never opens the labeled strict sample manifest, plume
masks, methane enhancement rasters, or the exact-paper diagnostic cache. It
selects the complete strict image/cloud corpus from the hash-bound remote asset
catalog, aligns it to the already materialized label-free paper feature cache,
and applies the unchanged Gaussian and DOFA deployment endpoints.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for _path in (ROOT / "tools", MODEL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

DEFAULT_PROTOCOL = Path("configs/mars_gaussian_dofa_strict_spatial_scoring_protocol.json")
EXPECTED_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B11",
    "B12",
    "B02_bg",
    "B03_bg",
    "B04_bg",
    "B08_bg",
    "B11_bg",
    "B12_bg",
)
ALLOWED_ASSET_ROLES = {"image", "cloud_mask"}
FORBIDDEN_SCORING_PATH_TOKENS = (
    "publication_v3_strict_samples",
    "validated_images_all",
    "plume_mask",
    "methane_enhancement",
    "diagnostic_cache",
    "outcome",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode())
    digest.update(data)
    return digest.hexdigest()


def _binding_records(value: Any, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            records.append((prefix.rstrip("."), value))
        else:
            for key, child in value.items():
                records.extend(_binding_records(child, f"{prefix}{key}."))
    return records


def validate_bindings(root: Path, dependencies: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for name, binding in _binding_records(dependencies):
        path = (root / str(binding["path"])).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"Binding escapes repository root: {name}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen dependency: {name}: {path}")
        observed = sha256(path)
        if observed != str(binding["sha256"]):
            raise ValueError(f"Frozen dependency hash mismatch: {name}")
        result.append({"name": name, "path": path.relative_to(root).as_posix(), "sha256": observed})
    return result


def assert_scoring_dependency_safety(dependencies: dict[str, Any]) -> None:
    for name, binding in _binding_records(dependencies):
        lowered = str(binding["path"]).lower()
        blocked = [token for token in FORBIDDEN_SCORING_PATH_TOKENS if token in lowered]
        if blocked:
            raise ValueError(f"Forbidden candidate-scoring dependency {name}: {blocked}")


def _sample_id_from_asset_path(path: str, role: str) -> str:
    name = Path(path).name
    suffix = "_s2.tif" if role == "image" else "_cloudmask.tif"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected {role} filename: {name}")
    base = name[: -len(suffix)]
    if "_" not in base:
        raise ValueError(f"Cannot recover sample id from {name}")
    sample_id = base.rsplit("_", 1)[1]
    if len(sample_id) != 36 or sample_id.count("-") != 4:
        raise ValueError(f"Recovered sample id is not a UUID: {sample_id}")
    return sample_id


def load_label_free_asset_catalog(path: Path) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    ignored_roles: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            role = str(record.get("role", ""))
            if role not in ALLOWED_ASSET_ROLES:
                ignored_roles[role] = ignored_roles.get(role, 0) + 1
                continue
            sample_id = _sample_id_from_asset_path(str(record["path"]), role)
            local = by_id.setdefault(sample_id, {"sample_id": sample_id, "assets": {}})
            if role in local["assets"]:
                raise ValueError(f"Duplicate {role} for {sample_id} at catalog line {line_number}")
            local["assets"][role] = {
                "path": str(record["path"]),
                "size": int(record["size"]),
                "remote_oid": str(record["remote_oid"]),
                "remote_oid_type": str(record["remote_oid_type"]),
            }
    rows = [by_id[key] for key in sorted(by_id)]
    if ignored_roles not in ({"plume_mask": 67}, {}):
        raise ValueError(f"Unexpected non-input roles in strict catalog: {ignored_roles}")
    for row in rows:
        if set(row["assets"]) != ALLOWED_ASSET_ROLES:
            raise ValueError(f"Incomplete label-free assets for {row['sample_id']}")
    return rows


def load_label_free_base_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        expected = {
            "sample_ids",
            "groups",
            "base_features",
            "base_feature_names",
            "current_v3_scores",
        }
        if set(source.files) != expected:
            raise ValueError(f"Label-free paper cache schema differs: {source.files}")
        result = {name: source[name].copy() for name in source.files}
    rows = result["sample_ids"].size
    if rows <= 0 or any(result[name].shape[0] != rows for name in ("groups", "base_features", "current_v3_scores")):
        raise ValueError("Label-free cache arrays do not align")
    if len(set(result["sample_ids"].astype(str).tolist())) != rows:
        raise ValueError("Label-free cache sample identities are not unique")
    return result


def align_strict_inputs(
    catalog_rows: list[dict[str, Any]], cache: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    ids = cache["sample_ids"].astype(str)
    lookup = {sample_id: index for index, sample_id in enumerate(ids)}
    selected_ids = [str(row["sample_id"]) for row in catalog_rows]
    missing = sorted(set(selected_ids) - set(lookup))
    if missing:
        raise ValueError(f"Strict assets are absent from label-free cache: {len(missing)}")
    indices = np.asarray([lookup[sample_id] for sample_id in selected_ids], dtype=np.int64)
    names = cache["base_feature_names"].astype(str).tolist()
    required_names = ("released_connected_score", "input_13_mean", "input_14_mean")
    if any(name not in names for name in required_names):
        raise ValueError("Label-free base feature cache lacks released score or wind channels")
    features = np.asarray(cache["base_features"][indices], dtype=np.float64)
    aligned = {
        "sample_ids": np.asarray(selected_ids),
        "groups": cache["groups"][indices].astype(str),
        "released_scores": features[:, names.index("released_connected_score")],
        "current_scores": np.asarray(cache["current_v3_scores"][indices], dtype=np.float64),
        "wind_u": features[:, names.index("input_13_mean")] * 8.0,
        "wind_v": features[:, names.index("input_14_mean")] * 8.0,
    }
    if not all(np.isfinite(value).all() for key, value in aligned.items() if key not in {"sample_ids", "groups"}):
        raise ValueError("Aligned label-free score or wind arrays contain non-finite values")
    if np.max(np.abs(np.concatenate((aligned["wind_u"], aligned["wind_v"])))) > 20.0001:
        raise ValueError("Reconstructed wind exceeds the frozen clipping contract")
    return catalog_rows, aligned


def verify_asset(path: Path, record: dict[str, Any], *, full_digest: bool) -> None:
    if not path.is_file() or path.stat().st_size != int(record["size"]):
        raise ValueError(f"Strict asset size mismatch: {path}")
    if not full_digest:
        return
    kind = str(record["remote_oid_type"])
    observed = sha256(path) if kind == "sha256_lfs" else git_blob_sha1(path)
    if kind not in {"sha256_lfs", "git_blob_sha1"}:
        raise ValueError(f"Unknown strict asset digest type: {kind}")
    if observed != str(record["remote_oid"]):
        raise ValueError(f"Strict asset digest mismatch: {path}")


def input_preflight(
    metadata_dir: Path,
    rows: list[dict[str, Any]],
    download_receipt_path: Path,
) -> dict[str, Any]:
    receipt = json.loads(download_receipt_path.read_text(encoding="utf-8"))
    result = receipt.get("result", {})
    verification = receipt.get("verification", {})
    if not (
        result.get("ok") is True
        and result.get("partial_scope") is False
        and int(result.get("remaining_bytes", -1)) == 0
        and verification.get("all_selected_assets_verified") is True
        and verification.get("size_checked") is True
    ):
        raise ValueError("Frozen strict download receipt is not a complete verified acquisition")
    bytes_total = sum(
        int(row["assets"][role]["size"])
        for row in rows
        for role in sorted(ALLOWED_ASSET_ROLES)
    )
    spot_indices = np.linspace(0, len(rows) - 1, num=min(32, len(rows)), dtype=np.int64)
    for index in np.unique(spot_indices):
        row = rows[int(index)]
        for role in sorted(ALLOWED_ASSET_ROLES):
            record = row["assets"][role]
            path = (metadata_dir / record["path"]).resolve()
            try:
                path.relative_to(metadata_dir.resolve())
            except ValueError as exc:
                raise ValueError(f"Strict asset escapes metadata root: {path}") from exc
            verify_asset(path, record, full_digest=False)
    return {
        "rows": len(rows),
        "input_assets": len(rows) * 2,
        "input_bytes_from_catalog": bytes_total,
        "spot_checked_assets": int(np.unique(spot_indices).size) * 2,
        "full_acquisition_assets_verified_by_bound_receipt": int(
            result["complete_size_match_count"]
        ),
        "full_digests_replayed": False,
    }


def load_input_batch(
    metadata_dir: Path,
    rows: list[dict[str, Any]],
    wind_u: np.ndarray,
    wind_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import rasterio
    from mars_s2l_adapter import CLOUD_CLASSES, compute_mbmp

    inputs: list[np.ndarray] = []
    observables: list[np.ndarray] = []
    for index, row in enumerate(rows):
        image_record = row["assets"]["image"]
        cloud_record = row["assets"]["cloud_mask"]
        image_path = (metadata_dir / image_record["path"]).resolve()
        cloud_path = (metadata_dir / cloud_record["path"]).resolve()
        with rasterio.open(image_path) as image_source:
            if image_source.count != 12 or set(image_source.dtypes) != {"uint16"}:
                raise ValueError(f"Strict image contract differs: {row['sample_id']}")
            if tuple(image_source.descriptions) != EXPECTED_BANDS:
                raise ValueError(f"Strict band ordering differs: {row['sample_id']}")
            raw_pair = image_source.read()
            image_shape = (image_source.height, image_source.width)
            image_crs = image_source.crs
            image_transform = image_source.transform
        with rasterio.open(cloud_path) as cloud_source:
            if cloud_source.count != 1 or (cloud_source.height, cloud_source.width) != image_shape:
                raise ValueError(f"Strict cloud geometry differs: {row['sample_id']}")
            if cloud_source.crs != image_crs or cloud_source.transform != image_transform:
                raise ValueError(f"Strict image/cloud grid differs: {row['sample_id']}")
            cloud = cloud_source.read(1)
        if raw_pair.shape != (12, 200, 200):
            raise ValueError(f"Strict image dimensions differ: {row['sample_id']}: {raw_pair.shape}")
        unknown_cloud = set(map(int, np.unique(cloud))) - set(CLOUD_CLASSES)
        if unknown_cloud:
            raise ValueError(f"Unknown cloud classes for {row['sample_id']}: {unknown_cloud}")
        spectral = np.clip(raw_pair.astype(np.float32) / 5000.0, 0.0, 2.0)
        radiometric = np.all(raw_pair[:6] != 0, axis=0) & np.all(raw_pair[6:] != 0, axis=0)
        clear = cloud == 0
        observable = radiometric & clear
        mbmp = compute_mbmp(spectral[:6], spectral[6:])
        wind = np.broadcast_to(
            np.asarray([wind_u[index], wind_v[index]], dtype=np.float32)[:, None, None] / 8.0,
            (2, 200, 200),
        ).copy()
        model_input = np.concatenate(
            (mbmp[None], spectral, wind, (cloud > 0).astype(np.float32)[None]), axis=0
        ).astype(np.float32)
        if model_input.shape != (16, 200, 200) or not np.isfinite(model_input).all():
            raise ValueError(f"Invalid strict model input: {row['sample_id']}")
        inputs.append(model_input)
        observables.append(observable[None])
    return np.stack(inputs), np.stack(observables).astype(bool)


def _batch_ranges(rows: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, rows, batch_size):
        yield start, min(start + batch_size, rows)


def gaussian_logits(
    *,
    metadata_dir: Path,
    rows: list[dict[str, Any]],
    aligned: dict[str, np.ndarray],
    protocol: dict[str, Any],
    batch_size: int,
    device: Any,
) -> np.ndarray:
    import torch
    from train_mars_gaussian_contrast_crossfit import TransferGaussianContrastViTUNet

    dependencies = protocol["scoring_dependencies"]
    architecture = json.loads(
        (ROOT / dependencies["gaussian_protocol"]["path"]).read_text(encoding="utf-8")
    )["architecture"]["model"]
    state_payload = torch.load(
        ROOT / dependencies["gaussian_state"]["path"], map_location="cpu", weights_only=False
    )
    states = state_payload.get("states_by_held_fold", {})
    if set(states) != {"3", "4"}:
        raise ValueError("Gaussian state must contain exactly opposite-fold endpoints 3 and 4")
    endpoint_logits: list[np.ndarray] = []
    for held_fold in (3, 4):
        model = TransferGaussianContrastViTUNet(**architecture).to(device)
        model.load_state_dict(states[str(held_fold)], strict=True)
        model.eval()
        parts: list[np.ndarray] = []
        for start, end in _batch_ranges(len(rows), batch_size):
            inputs, observable = load_input_batch(
                metadata_dir,
                rows[start:end],
                aligned["wind_u"][start:end],
                aligned["wind_v"][start:end],
            )
            with torch.inference_mode(), torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model(
                    torch.from_numpy(inputs).to(device),
                    torch.from_numpy(observable).float().to(device),
                    torch.zeros(end - start, dtype=torch.long, device=device),
                )
            parts.append(output["scene_logit"].float().reshape(-1).cpu().numpy())
            print(json.dumps({"stage": "gaussian", "held_fold": held_fold, "rows": end}), flush=True)
        endpoint_logits.append(np.concatenate(parts).astype(np.float64))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = np.mean(np.stack(endpoint_logits), axis=0)
    if result.shape != (len(rows),) or not np.isfinite(result).all():
        raise RuntimeError("Gaussian strict logits are incomplete or non-finite")
    return result


def dofa_scores(
    *,
    metadata_dir: Path,
    rows: list[dict[str, Any]],
    aligned: dict[str, np.ndarray],
    protocol: dict[str, Any],
    batch_size: int,
    device: Any,
) -> np.ndarray:
    import joblib
    import torch
    from dofa_v2_backbone import vit_base_patch14
    from extract_mars_dofa_v2_scene_features import (
        build_dofa_frames,
        feature_names,
        sensor_wavelengths,
        temporal_scene_features,
    )
    from materialize_mars_dofa_v2_external_deployment_state import external_mean_logit_score
    from train_mars_dofa_v2_scene_probe import select_features

    dependencies = protocol["scoring_dependencies"]
    model = vit_base_patch14()
    model.load_state_dict(
        torch.load(ROOT / dependencies["dofa_checkpoint"]["path"], map_location="cpu", weights_only=True),
        strict=True,
    )
    model = model.to(device).eval()
    parts: list[np.ndarray] = []
    for start, end in _batch_ranges(len(rows), batch_size):
        inputs, observable = load_input_batch(
            metadata_dir,
            rows[start:end],
            aligned["wind_u"][start:end],
            aligned["wind_v"][start:end],
        )
        batch = {
            "inputs": torch.from_numpy(inputs).to(device),
            "observable": torch.from_numpy(observable).to(device),
        }
        frames = build_dofa_frames(batch).flatten(0, 1)
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model.forward_features(frames, sensor_wavelengths(0))
        features = temporal_scene_features(
            [value[0::2] for value in output], [value[1::2] for value in output]
        )
        parts.append(features.cpu().numpy().astype(np.float32))
        print(json.dumps({"stage": "dofa", "rows": end}), flush=True)
    full = np.concatenate(parts)
    selected, names = select_features(full, np.asarray(feature_names()), "change_extreme")
    deployment = joblib.load(ROOT / dependencies["dofa_deployment_state"]["path"])
    if list(deployment["feature_names"]) != names.astype(str).tolist():
        raise ValueError("DOFA strict feature schema differs from frozen deployment state")
    result = np.asarray(external_mean_logit_score(deployment, selected), dtype=np.float64)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if result.shape != (len(rows),) or not np.isfinite(result).all():
        raise RuntimeError("DOFA strict scores are incomplete or non-finite")
    return result


def score_rows(
    *,
    metadata_dir: Path,
    rows: list[dict[str, Any]],
    aligned: dict[str, np.ndarray],
    protocol: dict[str, Any],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    from score_stanford_large_controlled_release_label_free import compose_gaussian_dofa_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaussian = gaussian_logits(
        metadata_dir=metadata_dir,
        rows=rows,
        aligned=aligned,
        protocol=protocol,
        batch_size=batch_size,
        device=device,
    )
    dofa = dofa_scores(
        metadata_dir=metadata_dir,
        rows=rows,
        aligned=aligned,
        protocol=protocol,
        batch_size=batch_size,
        device=device,
    )
    rule = protocol["candidate"]
    combined = compose_gaussian_dofa_score(
        aligned["current_scores"],
        gaussian,
        dofa,
        gaussian_strength=float(rule["gaussian_strength"]),
        final_gate=float(rule["final_protection_gate"]),
        dofa_gate=float(rule["dofa_gate"]),
        dofa_weight=float(rule["dofa_weight"]),
    )
    arrays = {
        "sample_ids": aligned["sample_ids"].astype(str),
        "groups": aligned["groups"].astype(str),
        "released_mars_v3_scores": aligned["released_scores"].astype(np.float64),
        "current_v3_scores": aligned["current_scores"].astype(np.float64),
        "gaussian_raw_logits": gaussian,
        "dofa_raw_scores": dofa,
        "gaussian_dofa_scores": combined,
        "released_mars_v3_decisions": (aligned["released_scores"] > float(protocol["released"]["threshold"])).astype(np.uint8),
        "gaussian_dofa_decisions": (combined >= float(rule["operational_threshold"])).astype(np.uint8),
    }
    expected = len(rows)
    for name, values in arrays.items():
        if np.asarray(values).shape != (expected,):
            raise RuntimeError(f"Incomplete strict score array: {name}")
        if name not in {"sample_ids", "groups"} and not np.isfinite(np.asarray(values)).all():
            raise RuntimeError(f"Non-finite strict score array: {name}")
    return arrays, {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch": torch.__version__,
        "rows": expected,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_outputs(
    *,
    root: Path,
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    protocol_path: Path,
    score_path: Path,
    manifest_path: Path,
    receipt_path: Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    for path in (score_path, manifest_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite strict one-shot score output: {path}")
    score_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = score_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, schema_version=np.asarray(1, dtype=np.int64), **arrays)
    os.replace(temporary, score_path)
    samples = [
        {
            "sample_id": row["sample_id"],
            "group_id": str(arrays["groups"][index]),
            "image": row["assets"]["image"],
            "cloud_mask": row["assets"]["cloud_mask"],
        }
        for index, row in enumerate(rows)
    ]
    manifest = {
        "schema_version": 1,
        "status": "complete_candidate_specific_input_only_scores",
        "rows": len(rows),
        "samples": samples,
        "forbidden_assets_opened": False,
    }
    _atomic_json(manifest_path, manifest)
    receipt = {
        "schema_version": 1,
        "status": "complete_candidate_specific_input_only_scores",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Gaussian+DOFA candidate-specific MARS strict-spatial replay",
        "project_level_outcomes_previously_known": True,
        "candidate_inference_opened_outcome_arrays": False,
        "candidate_inference_opened_plume_assets": False,
        "rows": len(rows),
        "label_free_site_context_groups": int(np.unique(arrays["groups"]).size),
        "protocol": {"path": protocol_path.relative_to(root).as_posix(), "sha256": sha256(protocol_path)},
        "scorer": {"path": Path(__file__).resolve().relative_to(root).as_posix(), "sha256": sha256(Path(__file__).resolve())},
        "scores": {"path": score_path.relative_to(root).as_posix(), "sha256": sha256(score_path)},
        "manifest": {"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256(manifest_path)},
        "runtime": runtime,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input-preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = (ROOT / DEFAULT_PROTOCOL).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dependencies = protocol["scoring_dependencies"]
    assert_scoring_dependency_safety(dependencies)
    bindings = validate_bindings(ROOT, dependencies)
    catalog_path = ROOT / dependencies["strict_remote_catalog"]["path"]
    cache_path = ROOT / dependencies["label_free_paper_cache"]["path"]
    metadata_dir = ROOT / protocol["input_contract"]["metadata_root"]
    rows = load_label_free_asset_catalog(catalog_path)
    cache = load_label_free_base_cache(cache_path)
    rows, aligned = align_strict_inputs(rows, cache)
    expected_rows = int(protocol["cohort"]["rows"])
    expected_groups = int(protocol["cohort"]["label_free_site_context_groups"])
    if len(rows) != expected_rows or np.unique(aligned["groups"]).size != expected_groups:
        raise RuntimeError("Strict input cohort differs from frozen row/group counts")
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "rows": len(rows), "label_free_site_context_groups": expected_groups, "verified_bindings": len(bindings), "outcome_arrays_opened": False}, sort_keys=True))
        return 0
    preflight = input_preflight(
        metadata_dir,
        rows,
        ROOT / dependencies["strict_download_receipt"]["path"],
    )
    if args.input_preflight:
        print(json.dumps({"status": "input_preflight_passed", **preflight, "outcome_arrays_opened": False}, sort_keys=True))
        return 0
    if not bool(protocol["implementation_gate"].get("real_inference_authorized", False)) and not args.smoke:
        raise RuntimeError("Strict candidate inference is not authorized by the frozen protocol")
    selected_rows = rows[:2] if args.smoke else rows
    selected = {name: value[: len(selected_rows)] for name, value in aligned.items()}
    arrays, runtime = score_rows(
        metadata_dir=metadata_dir,
        rows=selected_rows,
        aligned=selected,
        protocol=protocol,
        batch_size=args.batch_size,
    )
    if args.smoke:
        print(json.dumps({"status": "input_only_model_smoke_passed", "rows": len(selected_rows), "runtime": runtime, "arrays": {key: list(np.asarray(value).shape) for key, value in arrays.items()}, "outcome_arrays_opened": False}, sort_keys=True))
        return 0
    outputs = protocol["outputs"]
    receipt = write_outputs(
        root=ROOT,
        arrays=arrays,
        rows=rows,
        protocol_path=protocol_path,
        score_path=ROOT / outputs["scores"],
        manifest_path=ROOT / outputs["score_manifest"],
        receipt_path=ROOT / outputs["receipt"],
        runtime=runtime,
    )
    print(json.dumps({"status": receipt["status"], "rows": receipt["rows"], "scores_sha256": receipt["scores"]["sha256"], "outcome_arrays_opened": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
