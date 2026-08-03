#!/usr/bin/env python3
"""Hash-bound, outcome-blind scoring for the frozen Stanford crop cohort.

This module deliberately separates deterministic score composition and receipt
writing from heavy model inference. The production path refuses to emit partial
or approximate scores and never opens Stanford labels, rates, or detector
outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for _path in (ROOT / "tools", MODEL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

FROZEN_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
PIXEL_THRESHOLD = 0.5
MINIMUM_CONNECTED_PIXELS = 100
FORBIDDEN_OUTCOME_TOKENS = (
    "label", "truth", "metered", "release_rate", "ch4_kgh", "outcome",
    "positive_stratum", "official_test",
)
DEFAULT_PROTOCOL = Path("configs/stanford_large_controlled_release_scoring_protocol.json")
DEFAULT_PAIR_MANIFEST = Path(
    ".research/stanford_controlled_release_2024_2025/l1c_stress/pair_manifest.json"
)
DEFAULT_CROP_MANIFEST = Path(
    ".research/stanford_controlled_release_2024_2025/l1c_stress/crop_manifest.json"
)
DEFAULT_SCORE_PATH = Path(
    ".research/stanford_controlled_release_2024_2025/l1c_stress/scores/label_free_scores.npz"
)
DEFAULT_MANIFEST_PATH = Path(
    ".research/stanford_controlled_release_2024_2025/l1c_stress/scores/label_free_score_manifest.json"
)
DEFAULT_RECEIPT_PATH = Path(
    "reports/acquisition/stanford_large_controlled_release_label_free_scores.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_logit(values: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values >= 0.0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values)))


def build_mars_input(
    target_dn: np.ndarray,
    reference_dn: np.ndarray,
    *,
    wind_uv: tuple[float, float] = (4.0, 4.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact 16-channel released-MARS input plus physical pair.

    Physical reflectance is DN/10000. The released model scales this by 1/0.5,
    hence DN/5000 clipped to [0,2]. The cloud channel is frozen to zero.
    """
    from mars_s2l_adapter import compute_mbmp

    target = np.asarray(target_dn)
    reference = np.asarray(reference_dn)
    if target.shape != reference.shape or target.ndim != 3 or target.shape[0] != 6:
        raise ValueError("Expected aligned six-band target/reference arrays")
    if not np.issubdtype(target.dtype, np.integer) or not np.issubdtype(reference.dtype, np.integer):
        raise ValueError("Frozen crop contract requires integer L1C DN arrays")
    physical_pair = np.clip(
        np.concatenate((target, reference), axis=0).astype(np.float32) / 10000.0,
        0.0,
        1.0,
    )
    mars_pair = np.clip(
        np.concatenate((target, reference), axis=0).astype(np.float32) / 5000.0,
        0.0,
        2.0,
    )
    observable = np.all(target > 0, axis=0) & np.all(reference > 0, axis=0)
    mbmp = compute_mbmp(mars_pair[:6], mars_pair[6:])
    height, width = mbmp.shape
    wind = np.broadcast_to(
        np.asarray(wind_uv, dtype=np.float32)[:, None, None] / 8.0,
        (2, height, width),
    ).copy()
    cloud = np.zeros((1, height, width), dtype=np.float32)
    model_input = np.concatenate((mbmp[None], mars_pair, wind, cloud), axis=0).astype(np.float32)
    if model_input.shape != (16, height, width) or not np.isfinite(model_input).all():
        raise RuntimeError("Constructed MARS input is invalid")
    return model_input, observable, physical_pair


def connected_scene_score(score: np.ndarray) -> float:
    values = np.asarray(score, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Connected score expects one finite probability map")
    low = float(np.min(values))
    high = float(np.max(values))
    threshold = (low + high) / 2.0
    while high - low > 1e-3:
        labels, count = ndimage.label(
            values > threshold, structure=np.ones((3, 3), dtype=np.uint8)
        )
        largest = 0 if count == 0 else int(np.bincount(labels.ravel())[1:].max())
        if largest >= MINIMUM_CONNECTED_PIXELS:
            low = threshold
        else:
            high = threshold
        threshold = (low + high) / 2.0
    return float(threshold)


def released_scene_decision(score: float) -> bool:
    return float(score) > 0.5


def oof_extratrees_current_score(
    artifact: dict[str, Any],
    *,
    augmented_features: np.ndarray,
    augmented_feature_names: Iterable[str],
    primary_scores: np.ndarray,
) -> np.ndarray:
    names = [str(value) for value in augmented_feature_names]
    expected = [str(value) for value in artifact.get("augmented_feature_names", [])]
    if names != expected:
        raise ValueError("Current-head feature schema differs from frozen artifact")
    values = np.asarray(augmented_features, dtype=np.float64)
    primary = np.asarray(primary_scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != primary.size:
        raise ValueError("Current-head feature rows do not align")
    probability = np.asarray(artifact["fitted"].predict_proba(values)[:, 1], dtype=np.float64)
    weight = float(artifact["blend_lambda"])
    result = _sigmoid((1.0 - weight) * _safe_logit(primary) + weight * _safe_logit(probability))
    if not np.isfinite(result).all():
        raise RuntimeError("Current scores are non-finite")
    return result


def _local_logit(probability: np.ndarray, gate: float) -> np.ndarray:
    local = (np.asarray(probability, dtype=np.float64) - gate) / (1.0 - gate)
    return _safe_logit(local)


def _protected_logit_blend(
    current: np.ndarray, candidate: np.ndarray, *, gate: float, weight: float
) -> np.ndarray:
    base = np.asarray(current, dtype=np.float64)
    other = np.asarray(candidate, dtype=np.float64)
    if base.shape != other.shape or base.ndim != 1:
        raise ValueError("Protected blend inputs must be aligned vectors")
    eligible = base >= gate
    result = base.copy()
    if eligible.any():
        combined = (1.0 - weight) * _local_logit(base[eligible], gate) + weight * _local_logit(other[eligible], gate)
        result[eligible] = gate + (1.0 - gate) * _sigmoid(combined)
    return result


def compose_gaussian_dofa_score(
    current: np.ndarray,
    gaussian_raw_logits: np.ndarray,
    dofa_raw_scores: np.ndarray,
    *,
    gaussian_strength: float = 0.1,
    final_gate: float = 0.25,
    dofa_gate: float = 0.5,
    dofa_weight: float = 0.05,
) -> np.ndarray:
    base = np.asarray(current, dtype=np.float64)
    raw = np.asarray(gaussian_raw_logits, dtype=np.float64)
    dofa_raw = np.asarray(dofa_raw_scores, dtype=np.float64)
    if not (base.shape == raw.shape == dofa_raw.shape and base.ndim == 1):
        raise ValueError("Gaussian+DOFA vectors do not align")
    gaussian = base.copy()
    eligible = base >= final_gate
    if eligible.any():
        combined = _local_logit(base[eligible], final_gate) + float(gaussian_strength) * 2.0 * np.tanh(raw[eligible] / 2.0)
        gaussian[eligible] = final_gate + (1.0 - final_gate) * _sigmoid(combined)
    dofa = _protected_logit_blend(base, dofa_raw, gate=dofa_gate, weight=dofa_weight)
    result = base.copy()
    if eligible.any():
        current_logit = _local_logit(base[eligible], final_gate)
        combined = (
            _local_logit(gaussian[eligible], final_gate)
            + _local_logit(dofa[eligible], final_gate)
            - current_logit
        )
        result[eligible] = final_gate + (1.0 - final_gate) * _sigmoid(combined)
    if not np.array_equal(result[~eligible], base[~eligible]):
        raise RuntimeError("Protected composition changed below-gate scores")
    return result


def calibrated_spatial_prithvi_score(
    spatial_scores: np.ndarray,
    prithvi_scores: np.ndarray,
    *,
    prithvi_weight: float,
    logit_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    spatial = np.asarray(spatial_scores, dtype=np.float64)
    prithvi = np.asarray(prithvi_scores, dtype=np.float64)
    if spatial.shape != prithvi.shape or spatial.ndim != 1:
        raise ValueError("Spatial and Prithvi scores do not align")
    raw = _sigmoid((1.0 - prithvi_weight) * _safe_logit(spatial) + prithvi_weight * _safe_logit(prithvi))
    calibrated = _sigmoid(_safe_logit(raw) + float(logit_offset))
    return raw, calibrated


def _negative_attestation_only(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_negative_attestation_only(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_negative_attestation_only(child) for child in value)
    return value is False or value is None


def assert_no_outcome_data(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            forbidden = any(token in lowered for token in FORBIDDEN_OUTCOME_TOKENS)
            if forbidden:
                if not _negative_attestation_only(child):
                    raise ValueError(f"forbidden outcome field at {path}.{key}")
            else:
                assert_no_outcome_data(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_outcome_data(child, f"{path}[{index}]")


def _binding_records(value: Any, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            records.append((prefix.rstrip("."), value))
        else:
            for key, child in value.items():
                records.extend(_binding_records(child, f"{prefix}{key}."))
    return records


def validate_frozen_bindings(root: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    root = root.resolve()
    result = []
    for name, binding in _binding_records(protocol):
        path = (root / str(binding["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Frozen binding must remain beneath repository root: {name}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Frozen binding is missing: {name}: {path}")
        observed = sha256(path)
        expected = str(binding["sha256"])
        if observed != expected:
            raise ValueError(f"Frozen binding hash mismatch: {name}")
        result.append({"binding": name, "path": path.relative_to(root).as_posix(), "sha256": observed, "verified": True})
    return result


def audit_deployability(protocol: dict[str, Any]) -> dict[str, Any]:
    blockers = []
    gate = protocol.get("implementation_gate", {})
    if not bool(gate.get("real_inference_authorized", False)):
        blockers.append({"code": "INFERENCE_NOT_AUTHORIZED"})
    if not gate.get("current_feature_extractor"):
        blockers.append({"code": "CURRENT_FEATURE_EXTRACTOR_NOT_FROZEN"})
    if not gate.get("spatial_prithvi_deployment_state"):
        blockers.append({"code": "SPATIAL_PRITHVI_DEPLOYMENT_STATE_NOT_FROZEN"})
    if not gate.get("label_free_scorer"):
        blockers.append({"code": "LABEL_FREE_SCORER_NOT_HASH_BOUND"})
    return {"deployable": not blockers, "blockers": blockers, "no_shortcut_used": True}


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, schema_version=np.asarray(1, dtype=np.int64), **arrays)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_label_free_outputs(
    *,
    root: Path,
    arrays: dict[str, np.ndarray],
    sample_manifest: list[dict[str, Any]],
    score_path: Path,
    manifest_path: Path,
    receipt_path: Path,
    protocol_sha256: str,
    script_sha256: str,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert_no_outcome_data(sample_manifest)
    rows = len(sample_manifest)
    if rows <= 0 or any(np.asarray(value).shape[0] != rows for value in arrays.values()):
        raise ValueError("All label-free score arrays must have one row per sample")
    event_ids = np.asarray(arrays.get("event_ids", []), dtype=str)
    if event_ids.size != rows or len(set(event_ids.tolist())) != rows:
        raise ValueError("Score event identities must be complete and unique")
    for path in (score_path, manifest_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite one-shot output: {path}")
    _atomic_npz(score_path, arrays)
    manifest = {
        "schema_version": 1,
        "status": "complete_outcome_blind_label_free_scores",
        "rows": rows,
        "samples": sample_manifest,
        "labels_or_outcomes_accessed": False,
    }
    _atomic_json(manifest_path, manifest)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels_or_outcomes_accessed": False,
        "outcome_blindness": {"labels_or_rates_accessed": False},
        "summary": {"complete": True, "rows": rows},
        "protocol": {"sha256": protocol_sha256},
        "script": {"sha256": script_sha256},
        "runtime": {} if runtime is None else runtime,
        "scores": {"path": score_path.relative_to(root).as_posix(), "sha256": sha256(score_path)},
        "manifest": {"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256(manifest_path)},
    }
    assert_no_outcome_data(receipt)
    _atomic_json(receipt_path, receipt)
    return receipt


def _load_crop_inputs(
    crop_manifest_path: Path,
    pair_manifest_path: Path,
    *,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    import rasterio

    crop_payload = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
    pair_payload = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    if crop_payload.get("errors") or crop_payload["contract"]["band_order"] != list(FROZEN_BANDS):
        raise ValueError("Frozen crop manifest is incomplete or has a different band contract")
    samples = list(crop_payload["samples"])
    pairs = list(pair_payload["pairs"])
    assert_no_outcome_data(samples)
    assert_no_outcome_data(pairs)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No frozen crop samples selected")
    pair_by_id = {str(row["event_id"]): row for row in pairs}
    if len(pair_by_id) != len(pairs):
        raise ValueError("Pair manifest event identities are not unique")
    inputs: list[np.ndarray] = []
    observables: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    sample_manifest: list[dict[str, Any]] = []
    for row in samples:
        event_id = str(row["event_id"])
        if event_id not in pair_by_id:
            raise ValueError(f"Crop event is absent from pair manifest: {event_id}")
        target_record = row["assets"]["target_l1c"]
        reference_record = row["assets"]["reference_l1c"]
        target_path = (ROOT / target_record["path"]).resolve()
        reference_path = (ROOT / reference_record["path"]).resolve()
        if sha256(target_path) != target_record["sha256"] or sha256(reference_path) != reference_record["sha256"]:
            raise ValueError(f"Frozen crop checksum mismatch: {event_id}")
        with rasterio.open(target_path) as source:
            target = source.read()
            descriptions = tuple(str(value) for value in source.descriptions)
        with rasterio.open(reference_path) as source:
            reference = source.read()
            reference_descriptions = tuple(str(value) for value in source.descriptions)
        expected_reference_descriptions = tuple(f"{band}_reference" for band in FROZEN_BANDS)
        if descriptions != FROZEN_BANDS or reference_descriptions != expected_reference_descriptions:
            raise ValueError(f"Frozen crop band descriptions differ: {event_id}")
        if target.shape != (6, 256, 256) or reference.shape != target.shape:
            raise ValueError(f"Frozen crop geometry differs: {event_id}")
        model_input, observable, _ = build_mars_input(target, reference)
        inputs.append(model_input)
        observables.append(observable[None])
        pair = pair_by_id[event_id]
        metadata.append(
            {
                "event_id": event_id,
                "target_datetime": str(pair["target"]["datetime"]),
                "reference_scene_id": str(pair["reference"]["scene_id"]),
                "latitude": float(pair["center"][1]),
                "longitude": float(pair["center"][0]),
            }
        )
        sample_manifest.append(
            {
                "event_id": event_id,
                "target_sha256": str(target_record["sha256"]),
                "reference_sha256": str(reference_record["sha256"]),
            }
        )
    event_ids = [row["event_id"] for row in metadata]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("Selected crop identities are not unique")
    return (
        np.stack(inputs).astype(np.float32),
        np.stack(observables).astype(bool),
        metadata,
        sample_manifest,
    )


def _batch_ranges(rows: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, rows, batch_size):
        yield start, min(start + batch_size, rows)


def _residual_current_and_spatial(
    inputs: np.ndarray,
    observables: np.ndarray,
    dependencies: dict[str, Any],
    *,
    batch_size: int,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    import gc
    import joblib
    import torch
    from evaluate_mars_residual_endpoint_blend import load_residual_model, trust_region_logits
    from extract_mars_scene_features import pooled_scene_features, tensor_feature_names
    from extract_mars_spatial_scene_inputs import spatial_scene_channels
    from train_mars_context_scene_ranker import augment_site_context

    residual_payload = torch.load(
        ROOT / dependencies["residual_artifact"]["path"], map_location="cpu", weights_only=False
    )
    with torch.no_grad():
        model = load_residual_model(
            ROOT / dependencies["released_checkpoint"]["path"], residual_payload, device
        )
    feature_rows: list[np.ndarray] = []
    spatial_rows: list[np.ndarray] = []
    released_scores: list[float] = []
    for start, end in _batch_ranges(inputs.shape[0], batch_size):
        local_input = torch.from_numpy(inputs[start:end]).to(device)
        local_observable = torch.from_numpy(observables[start:end]).to(device)
        sensors = torch.zeros(end - start, dtype=torch.long, device=device)
        clear = torch.ones_like(local_observable, dtype=torch.bool)
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(local_input, local_observable, sensors)
            primary_logits = trust_region_logits(
                output["baseline_logits"], output["segmentation_logits"], 0.5
            )
            pooled = pooled_scene_features(
                local_input, primary_logits, output["baseline_logits"], clear, local_observable
            ).cpu().numpy()
            spatial = spatial_scene_channels(
                local_input, output["baseline_logits"], local_observable
            ).cpu().numpy().astype(np.float16)
        primary_probability = torch.sigmoid(primary_logits).float().cpu().numpy()
        released_probability = torch.sigmoid(output["baseline_logits"]).float().cpu().numpy()
        for index in range(end - start):
            primary_score = connected_scene_score(primary_probability[index, 0])
            released_score = connected_scene_score(released_probability[index, 0])
            feature_rows.append(
                np.concatenate(
                    (np.asarray([primary_score, released_score], dtype=np.float32), pooled[index])
                ).astype(np.float32)
            )
            released_scores.append(released_score)
        spatial_rows.append(spatial)
        print(json.dumps({"stage": "residual_current", "rows": end}), flush=True)
    raw_features = np.stack(feature_rows)
    feature_names = ["primary_connected_score", "released_connected_score", *tensor_feature_names()]
    groups = np.asarray(["casa_grande"] * inputs.shape[0])
    augmented, augmented_names = augment_site_context(
        raw_features, np.asarray(feature_names), groups
    )
    current_artifact = joblib.load(ROOT / dependencies["current_artifact"]["path"])
    primary = raw_features[:, feature_names.index("primary_connected_score")]
    current = oof_extratrees_current_score(
        current_artifact,
        augmented_features=augmented,
        augmented_feature_names=augmented_names,
        primary_scores=primary,
    )
    del model, residual_payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        np.asarray(released_scores, dtype=np.float64),
        current,
        augmented.astype(np.float32),
        list(augmented_names),
        np.concatenate(spatial_rows).astype(np.float32),
    )


def _gaussian_raw_logits(
    inputs: np.ndarray,
    observables: np.ndarray,
    dependencies: dict[str, Any],
    *,
    batch_size: int,
    device: Any,
) -> np.ndarray:
    import gc
    import torch
    from train_mars_gaussian_contrast_crossfit import TransferGaussianContrastViTUNet

    protocol = json.loads(
        (ROOT / dependencies["gaussian_protocol"]["path"]).read_text(encoding="utf-8")
    )
    state_payload = torch.load(
        ROOT / dependencies["gaussian_state"]["path"], map_location="cpu", weights_only=False
    )
    states = state_payload.get("states_by_held_fold", {})
    if set(states) != {"3", "4"}:
        raise ValueError("Gaussian deployment requires exactly held-fold states 3 and 4")
    endpoint_logits: list[np.ndarray] = []
    for held_fold in (3, 4):
        model = TransferGaussianContrastViTUNet(**protocol["architecture"]["model"]).to(device)
        model.load_state_dict(states[str(held_fold)], strict=True)
        model.eval()
        parts: list[np.ndarray] = []
        for start, end in _batch_ranges(inputs.shape[0], batch_size):
            local = torch.from_numpy(inputs[start:end]).to(device)
            observable = torch.from_numpy(observables[start:end]).float().to(device)
            sensors = torch.zeros(end - start, dtype=torch.long, device=device)
            with torch.inference_mode(), torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                output = model(local, observable, sensors)
            parts.append(output["scene_logit"].float().reshape(-1).cpu().numpy())
        endpoint_logits.append(np.concatenate(parts).astype(np.float64))
        print(json.dumps({"stage": "gaussian", "held_fold": held_fold, "rows": inputs.shape[0]}), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = np.mean(np.stack(endpoint_logits), axis=0)
    if not np.isfinite(result).all():
        raise RuntimeError("Gaussian external logits are non-finite")
    return result


def _dofa_scores(
    inputs: np.ndarray,
    observables: np.ndarray,
    dependencies: dict[str, Any],
    *,
    batch_size: int,
    device: Any,
) -> np.ndarray:
    import gc
    import joblib
    import torch
    from dofa_v2_backbone import vit_base_patch14
    from extract_mars_dofa_v2_scene_features import (
        build_dofa_frames,
        sensor_wavelengths,
        temporal_scene_features,
    )
    from materialize_mars_dofa_v2_external_deployment_state import external_mean_logit_score
    from train_mars_dofa_v2_scene_probe import select_features
    from extract_mars_dofa_v2_scene_features import feature_names

    model = vit_base_patch14()
    model.load_state_dict(
        torch.load(
            ROOT / dependencies["dofa_checkpoint"]["path"],
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    model = model.to(device).eval()
    rows: list[np.ndarray] = []
    for start, end in _batch_ranges(inputs.shape[0], batch_size):
        batch = {
            "inputs": torch.from_numpy(inputs[start:end]).to(device),
            "observable": torch.from_numpy(observables[start:end]).to(device),
        }
        frames = build_dofa_frames(batch).flatten(0, 1)
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            outputs = model.forward_features(frames, sensor_wavelengths(0))
        features = temporal_scene_features(
            [output[0::2] for output in outputs], [output[1::2] for output in outputs]
        )
        rows.append(features.cpu().numpy().astype(np.float32))
        print(json.dumps({"stage": "dofa", "rows": end}), flush=True)
    full_features = np.concatenate(rows)
    selected, selected_names = select_features(
        full_features, np.asarray(feature_names()), "change_extreme"
    )
    deployment = joblib.load(ROOT / dependencies["dofa_deployment_state"]["path"])
    if list(deployment["feature_names"]) != selected_names.astype(str).tolist():
        raise ValueError("DOFA external feature schema differs from materialized state")
    result = external_mean_logit_score(deployment, selected)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(result, dtype=np.float64)


def _prithvi_cls(
    inputs: np.ndarray,
    observables: np.ndarray,
    metadata: list[dict[str, Any]],
    dependencies: dict[str, Any],
    *,
    batch_size: int,
    device: Any,
) -> np.ndarray:
    import gc
    import torch
    from extract_mars_prithvi_scene_features import (
        INPUT_SIZE,
        build_features,
        build_input,
        date_coordinate,
        reference_date_coordinate,
    )

    foundation_dir = (ROOT / dependencies["prithvi_foundation_dir"]).resolve()
    if str(foundation_dir) not in sys.path:
        sys.path.insert(0, str(foundation_dir))
    from prithvi_mae import PrithviMAE  # type: ignore

    config = json.loads((foundation_dir / "config.json").read_text(encoding="utf-8"))[
        "pretrained_cfg"
    ]
    means = torch.tensor(config["mean"], dtype=torch.float32)[None, :, None, None, None].to(device)
    stds = torch.tensor(config["std"], dtype=torch.float32)[None, :, None, None, None].to(device)
    model_config = dict(config)
    model_config.update(img_size=INPUT_SIZE, num_frames=2, in_chans=6)
    model = PrithviMAE(**model_config)
    state = torch.load(
        ROOT / dependencies["prithvi_checkpoint"]["path"], map_location="cpu", weights_only=True
    )
    state["encoder.pos_embed"] = model.encoder.pos_embed
    state["decoder.decoder_pos_embed"] = model.decoder.decoder_pos_embed
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    rows: list[np.ndarray] = []
    for start, end in _batch_ranges(inputs.shape[0], batch_size):
        local_metadata = metadata[start:end]
        temporal = torch.tensor(
            [
                [
                    reference_date_coordinate(row),
                    date_coordinate(str(row["target_datetime"])),
                ]
                for row in local_metadata
            ],
            dtype=torch.float32,
            device=device,
        )
        location = torch.tensor(
            [[float(row["latitude"]), float(row["longitude"])] for row in local_metadata],
            dtype=torch.float32,
            device=device,
        )
        batch = {
            "inputs": torch.from_numpy(inputs[start:end]).to(device),
            "observable": torch.from_numpy(observables[start:end]).to(device),
        }
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model.forward_features(build_input(batch, means, stds), temporal, location)
        rows.append(build_features(output)[:, :768].cpu().numpy().astype(np.float32))
        print(json.dumps({"stage": "prithvi", "rows": end}), flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(rows)


def _spatial_prithvi_scores(
    current: np.ndarray,
    augmented: np.ndarray,
    spatial_images: np.ndarray,
    prithvi_cls: np.ndarray,
    dependencies: dict[str, Any],
    *,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    import joblib
    import torch
    from sklearn.linear_model import LogisticRegression
    from train_mars_adaptive_prithvi_probe import domain_normalize, load_features
    from train_mars_crossfold_bagged_scene_head import load_development
    from train_mars_scene_ranker import blend_scores
    from train_mars_site_relative_spatial_classifier import build_site_templates, predict_model

    groups = np.asarray(["casa_grande"] * current.size)
    sensors = np.zeros(current.size, dtype=np.uint8)
    spatial_artifact = torch.load(
        ROOT / dependencies["spatial_artifact"]["path"], map_location="cpu", weights_only=False
    )
    means, counts, inverse = build_site_templates(spatial_images, groups)
    spatial_raw = predict_model(
        spatial_artifact["fitted"],
        spatial_images,
        np.arange(current.size),
        sensors,
        means,
        counts,
        inverse,
        device,
    )
    spatial = blend_scores(current, spatial_raw, float(spatial_artifact["blend_weight"]))
    development = load_development(
        {
            "inner": ROOT / dependencies["development_inner"]["path"],
            "fold0": ROOT / dependencies["development_fold0"]["path"],
            "fold1": ROOT / dependencies["development_fold1"]["path"],
        },
        ROOT / dependencies["development_scores"]["path"],
    )
    source = load_features(
        ROOT / dependencies["development_prithvi"]["path"], development, "cls_plus_base"
    ).astype(np.float64)
    target = np.concatenate((prithvi_cls, augmented.astype(np.float32)), axis=1).astype(np.float64)
    source_norm, target_norm = domain_normalize(source, target)
    positives = int(np.count_nonzero(development["labels"] == 1))
    negatives = int(np.count_nonzero(development["labels"] == 0))
    weights = np.where(development["labels"] == 1, np.sqrt(negatives / positives), 1.0)
    adaptive = joblib.load(ROOT / dependencies["adaptive_prithvi_artifact"]["path"])
    probe = LogisticRegression(
        C=float(adaptive["C"]), max_iter=500, solver="lbfgs", random_state=20261550
    ).fit(source_norm, development["labels"], sample_weight=weights)
    prithvi_raw = probe.predict_proba(target_norm)[:, 1]
    prithvi = blend_scores(current, prithvi_raw, float(adaptive["blend_weight"]))
    ensemble = joblib.load(ROOT / dependencies["spatial_prithvi_artifact"]["path"])
    calibration = joblib.load(ROOT / dependencies["calibrated_spatial_prithvi_artifact"]["path"])
    return calibrated_spatial_prithvi_score(
        spatial,
        prithvi,
        prithvi_weight=float(ensemble["prithvi_weight"]),
        logit_offset=float(calibration["logit_offset"]),
    )


def _score_input_arrays(
    *,
    inputs: np.ndarray,
    observables: np.ndarray,
    metadata: list[dict[str, Any]],
    protocol: dict[str, Any],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    dependencies = protocol["deployment_dependencies"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    released, current, augmented, _, spatial_images = _residual_current_and_spatial(
        inputs, observables, dependencies, batch_size=batch_size, device=device
    )
    gaussian_logits = _gaussian_raw_logits(
        inputs, observables, dependencies, batch_size=batch_size, device=device
    )
    dofa_raw = _dofa_scores(
        inputs, observables, dependencies, batch_size=batch_size, device=device
    )
    gaussian_dofa = compose_gaussian_dofa_score(
        current,
        gaussian_logits,
        dofa_raw,
        gaussian_strength=float(protocol["gaussian_dofa_candidate"]["gaussian_strength"]),
        final_gate=float(protocol["gaussian_dofa_candidate"]["final_protection_gate"]),
        dofa_gate=float(protocol["gaussian_dofa_candidate"]["dofa_protected_fusion"]["gate"]),
        dofa_weight=float(protocol["gaussian_dofa_candidate"]["dofa_protected_fusion"]["weight"]),
    )
    prithvi = _prithvi_cls(
        inputs, observables, metadata, dependencies, batch_size=batch_size, device=device
    )
    _, calibrated_spatial = _spatial_prithvi_scores(
        current, augmented, spatial_images, prithvi, dependencies, device=device
    )
    event_ids = np.asarray([row["event_id"] for row in metadata])
    arrays = {
        "event_ids": event_ids,
        "released_mars_v3_scores": released,
        "released_mars_v3_decisions": np.asarray(
            [released_scene_decision(value) for value in released], dtype=np.uint8
        ),
        "current_oof_extratrees_scores": current,
        "gaussian_dofa_scores": gaussian_dofa,
        "gaussian_dofa_decisions": (
            gaussian_dofa >= float(protocol["gaussian_dofa_candidate"]["operational_threshold"])
        ).astype(np.uint8),
        "calibrated_spatial_prithvi_scores": calibrated_spatial,
        "calibrated_spatial_prithvi_decisions": (
            calibrated_spatial
            >= float(protocol["spatial_prithvi_posttest_candidate"]["operational_threshold"])
        ).astype(np.uint8),
    }
    for key, values in arrays.items():
        if np.asarray(values).shape[0] != event_ids.size:
            raise RuntimeError(f"Incomplete score array: {key}")
        if key != "event_ids" and not np.isfinite(np.asarray(values)).all():
            raise RuntimeError(f"Non-finite score array: {key}")
    runtime = {
        "device": str(torch.cuda.get_device_name(device) if device.type == "cuda" else device),
        "torch": torch.__version__,
        "rows": int(event_ids.size),
    }
    return arrays, runtime


def score_frozen_cohort(
    *,
    protocol: dict[str, Any],
    pair_manifest_path: Path,
    crop_manifest_path: Path,
    limit: int | None,
    batch_size: int = 2,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    inputs, observables, metadata, sample_manifest = _load_crop_inputs(
        crop_manifest_path, pair_manifest_path, limit=limit
    )
    arrays, runtime = _score_input_arrays(
        inputs=inputs,
        observables=observables,
        metadata=metadata,
        protocol=protocol,
        batch_size=batch_size,
    )
    return arrays, sample_manifest, runtime


def synthetic_model_smoke(
    protocol: dict[str, Any], *, batch_size: int
) -> dict[str, Any]:
    generator = np.random.default_rng(20260802)
    inputs: list[np.ndarray] = []
    observables: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for index in range(2):
        target = generator.integers(500, 2500, size=(6, 256, 256), dtype=np.uint16)
        reference = generator.integers(500, 2500, size=(6, 256, 256), dtype=np.uint16)
        model_input, observable, _ = build_mars_input(target, reference)
        inputs.append(model_input)
        observables.append(observable[None])
        metadata.append(
            {
                "event_id": f"synthetic-{index}",
                "target_datetime": f"2025-01-0{index + 1}T18:00:00+00:00",
                "reference_scene_id": "S2A_12SVB_20240102_0_L1C",
                "latitude": 32.821749,
                "longitude": -111.785795,
            }
        )
    arrays, runtime = _score_input_arrays(
        inputs=np.stack(inputs).astype(np.float32),
        observables=np.stack(observables).astype(bool),
        metadata=metadata,
        protocol=protocol,
        batch_size=batch_size,
    )
    return {
        "status": "synthetic_model_smoke_passed",
        "runtime": runtime,
        "arrays": {key: list(np.asarray(value).shape) for key, value in arrays.items()},
        "stanford_imagery_accessed": False,
        "stanford_outcomes_accessed": False,
    }


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--pair-manifest", default=DEFAULT_PAIR_MANIFEST.as_posix())
    parser.add_argument("--crop-manifest", default=DEFAULT_CROP_MANIFEST.as_posix())
    parser.add_argument("--scores", default=DEFAULT_SCORE_PATH.as_posix())
    parser.add_argument("--score-manifest", default=DEFAULT_MANIFEST_PATH.as_posix())
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT_PATH.as_posix())
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input-preflight", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    binding_records = validate_frozen_bindings(ROOT, protocol.get("deployment_dependencies", {}))
    audit = audit_deployability(protocol)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "audit": audit,
                    "verified_deployment_bindings": len(binding_records),
                },
                sort_keys=True,
            )
        )
        return 0 if audit["deployable"] else 2
    if args.synthetic_smoke:
        result = synthetic_model_smoke(protocol, batch_size=args.batch_size)
        result["verified_deployment_bindings"] = len(binding_records)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.input_preflight:
        inputs, observables, metadata, samples = _load_crop_inputs(
            (ROOT / args.crop_manifest).resolve(),
            (ROOT / args.pair_manifest).resolve(),
            limit=None,
        )
        expected_rows = int(protocol["cohort_inputs"]["expected_rows"])
        if len(samples) != expected_rows:
            raise RuntimeError(f"Input preflight row count differs: {len(samples)} != {expected_rows}")
        print(
            json.dumps(
                {
                    "status": "input_preflight_passed",
                    "rows": len(samples),
                    "model_input_shape": list(inputs.shape),
                    "observable_shape": list(observables.shape),
                    "metadata_rows": len(metadata),
                    "stanford_detector_outputs_accessed": False,
                    "stanford_labels_or_rates_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if not audit["deployable"]:
        raise RuntimeError(f"Frozen deployment audit is blocked: {audit['blockers']}")
    if args.limit is not None:
        raise ValueError("Partial real-cohort scoring is forbidden; omit --limit")
    expected_rows = int(protocol["cohort_inputs"]["expected_rows"])
    arrays, sample_manifest, runtime = score_frozen_cohort(
        protocol=protocol,
        pair_manifest_path=(ROOT / args.pair_manifest).resolve(),
        crop_manifest_path=(ROOT / args.crop_manifest).resolve(),
        limit=None,
        batch_size=args.batch_size,
    )
    if len(sample_manifest) != expected_rows:
        raise RuntimeError(
            f"Frozen full-cohort row count differs: {len(sample_manifest)} != {expected_rows}"
        )
    receipt = write_label_free_outputs(
        root=ROOT,
        arrays=arrays,
        sample_manifest=sample_manifest,
        score_path=(ROOT / args.scores).resolve(),
        manifest_path=(ROOT / args.score_manifest).resolve(),
        receipt_path=(ROOT / args.receipt).resolve(),
        protocol_sha256=sha256(protocol_path),
        script_sha256=sha256(Path(__file__).resolve()),
        runtime=runtime,
    )
    print(
        json.dumps(
            {
                "status": "complete_outcome_blind_label_free_scores",
                "rows": len(sample_manifest),
                "scores_sha256": receipt["scores"]["sha256"],
                "manifest_sha256": receipt["manifest"]["sha256"],
                "labels_or_outcomes_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
