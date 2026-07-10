"""Shared ERSRR segmentation model, preprocessing, and artifact contract.

This module is the single source of truth used by research training and Flask
serving.  It intentionally exposes only the validated segmentation task.  The
legacy methane regression head remains in ``ERSRR_Model.py`` as a historical
baseline until target-independent physical-unit calibration is available.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import keras
import numpy as np
from keras import layers, ops

ARTIFACT_SCHEMA_VERSION = 1
MODEL_NAME = "ersrr_compact_resunet_v1"
BAND_ORDER = ("B2", "B3", "B4", "B11", "B12")
RAW_FEATURE_NAMES = tuple(f"log1p_{band}" for band in BAND_ORDER)
PHYSICS_FEATURE_NAMES = (
    "log_B12_over_B11",
    "normalized_difference_B12_B11",
    "log_B11_over_B4",
    "log_B12_over_B4",
    "log_SWIR_over_visible",
    "normalized_difference_B4_B2",
)
FEATURE_NAMES = RAW_FEATURE_NAMES + PHYSICS_FEATURE_NAMES
EPSILON = 1.0


class ArtifactError(RuntimeError):
    """Raised when a model artifact violates the serving contract."""


def transform_sentinel2(images: np.ndarray, *, physics: bool = True) -> np.ndarray:
    """Convert canonical B2/B3/B4/B11/B12 arrays to model feature channels."""
    values = np.asarray(images, dtype="float32")
    if values.shape[-1] != len(BAND_ORDER):
        raise ValueError(f"Expected {len(BAND_ORDER)} bands in {BAND_ORDER}; got shape {values.shape}")
    values = np.clip(values, 0.0, None)
    log_values = np.log1p(values)
    if not physics:
        return log_values.astype("float32")

    b2, b3, b4, b11, b12 = (values[..., index] for index in range(5))
    engineered = np.stack(
        [
            log_values[..., 4] - log_values[..., 3],
            (b12 - b11) / (b12 + b11 + EPSILON),
            log_values[..., 3] - log_values[..., 2],
            log_values[..., 4] - log_values[..., 2],
            np.log1p(b11 + b12) - np.log1p(b2 + b3 + b4),
            (b4 - b2) / (b4 + b2 + EPSILON),
        ],
        axis=-1,
    )
    return np.concatenate([log_values, engineered], axis=-1).astype("float32")


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype="float32")
    mean_values = np.asarray(mean, dtype="float32")
    std_values = np.asarray(std, dtype="float32")
    if values.shape[-1] != mean_values.size or mean_values.shape != std_values.shape:
        raise ValueError(
            f"Feature/stat shape mismatch: features={values.shape}, mean={mean_values.shape}, std={std_values.shape}"
        )
    safe_std = np.where(std_values < 1e-6, 1.0, std_values)
    return ((values - mean_values) / safe_std).astype("float32")


@keras.saving.register_keras_serializable(package="ersrr")
class MaskedBCEDice(keras.losses.Loss):
    """Validity-aware positive-weighted BCE plus soft Dice."""

    def __init__(self, positive_weight: float = 1.0, name: str = "masked_bce_dice") -> None:
        super().__init__(name=name)
        self.positive_weight = float(positive_weight)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        labels = y_true[..., :1]
        valid = y_true[..., 1:2]
        probabilities = ops.clip(y_pred, 1e-6, 1.0 - 1e-6)
        weights = 1.0 + labels * (self.positive_weight - 1.0)
        bce = -(labels * ops.log(probabilities) + (1.0 - labels) * ops.log(1.0 - probabilities))
        bce = ops.sum(bce * weights * valid, axis=(1, 2, 3)) / (
            ops.sum(weights * valid, axis=(1, 2, 3)) + 1e-6
        )
        intersection = ops.sum(labels * probabilities * valid, axis=(1, 2, 3))
        denominator = ops.sum((labels + probabilities) * valid, axis=(1, 2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        return 0.5 * bce + 0.5 * dice_loss

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "positive_weight": self.positive_weight}


@keras.saving.register_keras_serializable(package="ersrr")
def masked_dice(y_true: Any, y_pred: Any) -> Any:
    labels = y_true[..., :1]
    valid = y_true[..., 1:2]
    predicted = ops.cast(y_pred >= 0.5, "float32")
    intersection = ops.sum(labels * predicted * valid)
    denominator = ops.sum((labels + predicted) * valid)
    return (2.0 * intersection + 1.0) / (denominator + 1.0)


@keras.saving.register_keras_serializable(package="ersrr")
def masked_iou(y_true: Any, y_pred: Any) -> Any:
    labels = y_true[..., :1]
    valid = y_true[..., 1:2]
    predicted = ops.cast(y_pred >= 0.5, "float32")
    intersection = ops.sum(labels * predicted * valid)
    union = ops.sum(ops.maximum(labels, predicted) * valid)
    return (intersection + 1.0) / (union + 1.0)


def _normalization_groups(filters: int) -> int:
    return next(group for group in (8, 6, 4, 3, 2, 1) if filters % group == 0)


def _residual_block(x: Any, filters: int, dropout: float = 0.0) -> Any:
    shortcut = x
    groups = _normalization_groups(filters)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=groups)(x)
    x = layers.Activation("swish")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.GroupNormalization(groups=groups)(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("swish")(x)
    if dropout:
        x = layers.SpatialDropout2D(dropout)(x)
    return x


def build_compact_resunet(
    *,
    input_channels: int = len(FEATURE_NAMES),
    base_filters: int = 8,
    image_size: int | None = None,
) -> keras.Model:
    """Build the validated compact residual U-Net (input must be divisible by 8)."""
    spatial = image_size if image_size is not None else None
    inputs = keras.Input((spatial, spatial, input_channels), name="sentinel2_features")
    encoder_1 = _residual_block(inputs, base_filters)
    x = layers.Conv2D(base_filters * 2, 3, strides=2, padding="same")(encoder_1)
    encoder_2 = _residual_block(x, base_filters * 2)
    x = layers.Conv2D(base_filters * 4, 3, strides=2, padding="same")(encoder_2)
    encoder_3 = _residual_block(x, base_filters * 4, dropout=0.05)
    x = layers.Conv2D(base_filters * 6, 3, strides=2, padding="same")(encoder_3)
    x = _residual_block(x, base_filters * 6, dropout=0.1)
    for skip, filters in (
        (encoder_3, base_filters * 4),
        (encoder_2, base_filters * 2),
        (encoder_1, base_filters),
    ):
        x = layers.UpSampling2D(interpolation="bilinear")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = _residual_block(x, filters)
    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="plume_mask")(x)
    return keras.Model(inputs, outputs, name=MODEL_NAME)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError(f"Unsupported artifact schema: {config.get('schema_version')}")
    if tuple(config.get("band_order", [])) != BAND_ORDER:
        raise ArtifactError(f"Artifact band order must be {BAND_ORDER}")
    if config.get("task") != "plume_segmentation":
        raise ArtifactError(f"Unsupported task: {config.get('task')}")
    if not isinstance(config.get("physics_features"), bool):
        raise ArtifactError("physics_features must be a boolean")
    expected_names = FEATURE_NAMES if config["physics_features"] else RAW_FEATURE_NAMES
    if tuple(config.get("feature_names", [])) != expected_names:
        raise ArtifactError(f"Artifact feature_names must be {expected_names}")
    if not isinstance(config.get("model_file"), str) or not config["model_file"].strip():
        raise ArtifactError("Artifact must declare a non-empty model_file")
    model_hash = config.get("model_sha256")
    if not isinstance(model_hash, str) or re.fullmatch(r"[0-9a-fA-F]{64}", model_hash) is None:
        raise ArtifactError("Artifact must declare a 64-character hexadecimal model_sha256")
    for field in ("input_product", "product_level", "status"):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise ArtifactError(f"Artifact must declare a non-empty {field}")
    tile_size = config.get("inference_tile_size")
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0 or tile_size % 8:
        raise ArtifactError(f"inference_tile_size must be a positive multiple of 8; got {tile_size}")
    resolution = config.get("input_resolution_m")
    if not isinstance(resolution, (int, float)) or isinstance(resolution, bool) or not np.isfinite(resolution) or resolution <= 0:
        raise ArtifactError(f"input_resolution_m must be positive and finite; got {resolution}")
    for field in ("input_radiometric_scale", "input_radiometric_offset"):
        value = config.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            raise ArtifactError(f"{field} must be finite; got {value}")
    if config["input_radiometric_scale"] <= 0:
        raise ArtifactError("input_radiometric_scale must be positive")
    if config.get("input_resampling") not in {"nearest", "bilinear", "bicubic"}:
        raise ArtifactError("input_resampling must be nearest, bilinear, or bicubic")
    nodata = config.get("input_nodata_value")
    if not isinstance(nodata, (int, float)) or isinstance(nodata, bool) or not np.isfinite(nodata):
        raise ArtifactError(f"input_nodata_value must be finite; got {nodata}")
    normalization = config.get("normalization", {})
    if not isinstance(normalization, dict):
        raise ArtifactError("normalization must be an object")
    mean = normalization.get("mean", [])
    std = normalization.get("std", [])
    expected = len(expected_names)
    if len(mean) != expected or len(std) != expected:
        raise ArtifactError(f"Expected {expected} normalization channels; got {len(mean)}/{len(std)}")
    try:
        mean_values = np.asarray(mean, dtype="float64")
        std_values = np.asarray(std, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ArtifactError("Normalization values must be numeric") from exc
    if not np.all(np.isfinite(mean_values)) or not np.all(np.isfinite(std_values)):
        raise ArtifactError("Normalization values must be finite")
    if np.any(std_values <= 0.0):
        raise ArtifactError("Normalization standard deviations must be positive")
    threshold = config.get("decision_threshold")
    if (
        not isinstance(threshold, (int, float))
        or not np.isfinite(float(threshold))
        or not 0.0 < float(threshold) < 1.0
    ):
        raise ArtifactError(f"Invalid decision threshold: {threshold}")


def load_artifact(artifact_dir: Path) -> tuple[keras.Model, dict[str, Any]]:
    artifact_root = artifact_dir.resolve()
    config_path = artifact_root / "config.json"
    if not config_path.exists():
        raise ArtifactError(f"Missing artifact config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_artifact_config(config)
    model_path = (artifact_root / config["model_file"]).resolve()
    if artifact_root not in model_path.parents or not model_path.exists():
        raise ArtifactError(f"Missing or unsafe model path: {model_path}")
    expected_hash = config["model_sha256"].lower()
    if sha256(model_path) != expected_hash:
        raise ArtifactError("Model SHA-256 does not match config")
    model = keras.models.load_model(model_path, compile=False)
    expected_channels = len(FEATURE_NAMES) if config["physics_features"] else len(RAW_FEATURE_NAMES)
    if not isinstance(model.input_shape, tuple) or model.input_shape[-1] != expected_channels:
        raise ArtifactError(
            f"Model expects input shape {model.input_shape}; artifact contract requires {expected_channels} channels"
        )
    expected_spatial = (config["inference_tile_size"], config["inference_tile_size"])
    if tuple(model.input_shape[1:3]) != expected_spatial:
        raise ArtifactError(
            f"Model expects spatial shape {model.input_shape[1:3]}; artifact contract requires {expected_spatial}"
        )
    if not isinstance(model.output_shape, tuple) or model.output_shape[-1] != 1:
        raise ArtifactError(f"Model must produce one plume-probability channel; got {model.output_shape}")
    return model, config


def predict_tile(model: keras.Model, config: dict[str, Any], sentinel2_tile: np.ndarray) -> np.ndarray:
    tile = np.asarray(sentinel2_tile, dtype="float32")
    if tile.ndim != 3 or tile.shape[-1] != len(BAND_ORDER):
        raise ValueError(f"Expected H×W×5 tile; got {tile.shape}")
    if not np.all(np.isfinite(tile)):
        raise ValueError("Sentinel-2 tile contains non-finite values")
    expected_size = int(config["inference_tile_size"])
    if tile.shape[:2] != (expected_size, expected_size):
        raise ValueError(f"Artifact expects {expected_size}×{expected_size} tiles; got {tile.shape[:2]}")
    tile = tile * float(config["input_radiometric_scale"]) + float(config["input_radiometric_offset"])
    features = transform_sentinel2(tile, physics=bool(config["physics_features"]))
    normalization = config["normalization"]
    normalized = normalize_features(features, normalization["mean"], normalization["std"])
    prediction = model.predict(normalized[None], verbose=0)
    result = np.asarray(prediction[0, ..., 0], dtype="float32")
    if result.shape != tile.shape[:2] or not np.all(np.isfinite(result)):
        raise ArtifactError(f"Model returned invalid prediction shape/values: {result.shape}")
    return result
