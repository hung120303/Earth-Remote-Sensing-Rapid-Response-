"""Flask API for deterministic Sentinel-2 selection and ERSRR inference.

The trained model, feature transformation, normalization, band order, and
decision threshold are owned by the versioned artifact contract in
``EarthRemoteSensingRapidResponse/ersrr_core.py``.  This module deliberately
does not authenticate Earth Engine or load a model at import time: deployments
must provide non-interactive Earth Engine credentials and a validated artifact.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import mercantile
import numpy as np
import rasterio
import requests
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject

try:
    import ee
except ImportError as exc:  # Keep /health useful in minimal deployments.
    ee = None
    _EE_IMPORT_ERROR: str | None = type(exc).__name__
else:
    _EE_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PACKAGE_DIR = REPO_ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_PACKAGE_DIR))

from ersrr_core import (  # noqa: E402  (sibling package path is intentional)
    BAND_ORDER,
    ArtifactError,
    load_artifact,
    predict_tile,
    validate_artifact_config,
)


app = Flask(__name__)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ERSRR_CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
CORS(
    app,
    resources={r"/*": {"origins": CORS_ORIGINS}},
    methods=["GET", "POST", "OPTIONS"],
)

WEB_TILE_SIZE = 256
DEFAULT_ARTIFACT_DIR = MODEL_PACKAGE_DIR / "artifacts" / "compact_resunet_v1"
_artifact_setting = Path(os.environ.get("ERSRR_ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)).expanduser()
ARTIFACT_DIR = (_artifact_setting if _artifact_setting.is_absolute() else REPO_ROOT / _artifact_setting).resolve()
PREDICTION_DIR = Path(__file__).resolve().parent / "Predictions"
PRED_PATH = PREDICTION_DIR / "testprediction.tif"
SENTINEL_PATH = Path(__file__).resolve().parent / "temp_s2.tif"
DEFAULT_START_DATE = os.environ.get("ERSRR_DEFAULT_START", "2022-01-01")
DEFAULT_END_DATE = os.environ.get("ERSRR_DEFAULT_END", "2023-01-01")
DEFAULT_MAX_CLOUD = float(os.environ.get("ERSRR_MAX_CLOUD", "20"))
MAX_ROI_SPAN_DEGREES = float(os.environ.get("ERSRR_MAX_ROI_SPAN_DEGREES", "0.25"))
EE_PROJECT = os.environ.get("ERSRR_EE_PROJECT", "ersrr-475700")

prediction_data: np.ndarray | None = None
prediction_profile: dict[str, Any] | None = None
prediction_id: str | None = None

_state_lock = threading.RLock()
_inference_lock = threading.Lock()
_artifact_model: Any | None = None
_artifact_config: dict[str, Any] | None = None
_artifact_error: str | None = None
_ee_initialized = False
_ee_error: str | None = _EE_IMPORT_ERROR


class RequestValidationError(ValueError):
    """Raised for client-supplied query parameters that cannot be accepted."""


def _json_error(message: str, status: int, *, component: str | None = None):
    payload: dict[str, Any] = {"error": message}
    if component:
        payload["component"] = component
    return jsonify(payload), status


def _artifact_status() -> dict[str, Any]:
    """Describe artifact readiness without forcing a Keras model load."""
    with _state_lock:
        if _artifact_model is not None and _artifact_config is not None:
            return {
                "status": "ready",
                "model": _artifact_config.get("model_name"),
                "research_status": _artifact_config.get("status"),
            }
        if _artifact_error:
            return {
                "status": "unavailable",
                "error": "artifact load or validation failed",
                "error_type": _artifact_error,
            }

    config_path = ARTIFACT_DIR / "config.json"
    if not config_path.exists():
        return {
            "status": "unavailable",
            "error": "artifact config is missing",
        }
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_artifact_config(config)
        product = config.get("input_product")
        if not isinstance(product, str) or not product.strip():
            raise ArtifactError("Artifact config must declare a non-empty input_product")
        model_path = ARTIFACT_DIR / str(config.get("model_file", ""))
        if not model_path.is_file():
            raise ArtifactError(f"Missing artifact model: {model_path}")
    except (ArtifactError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "unavailable",
            "error": "artifact config is invalid",
            "error_type": type(exc).__name__,
        }
    return {
        "status": "configured_not_loaded",
        "model": config.get("model_name"),
        "research_status": config.get("status"),
        "input_product": product,
    }


def _ensure_artifact() -> tuple[Any, dict[str, Any]]:
    """Load and validate the configured artifact exactly once on first use."""
    global _artifact_model, _artifact_config, _artifact_error
    with _state_lock:
        if _artifact_model is not None and _artifact_config is not None:
            return _artifact_model, _artifact_config
        try:
            model, config = load_artifact(ARTIFACT_DIR)
            product = config.get("input_product")
            if not isinstance(product, str) or not product.strip():
                raise ArtifactError("Artifact config must declare a non-empty input_product")
        except Exception as exc:  # Convert TensorFlow/filesystem failures to a stable 503.
            _artifact_error = type(exc).__name__
            app.logger.exception("Artifact load failed")
            raise ArtifactError("artifact load or validation failed") from exc
        _artifact_model = model
        _artifact_config = config
        _artifact_error = None
        return model, config


def _earth_engine_status() -> dict[str, Any]:
    if ee is None:
        return {"status": "unavailable", "project": EE_PROJECT, "error_type": _ee_error}
    with _state_lock:
        if _ee_initialized:
            return {"status": "ready", "project": EE_PROJECT}
        if _ee_error:
            return {"status": "unavailable", "project": EE_PROJECT, "error_type": _ee_error}
    return {"status": "not_initialized", "project": EE_PROJECT}


def _ensure_earth_engine() -> None:
    """Initialize from ambient credentials; never launch an interactive login."""
    global _ee_initialized, _ee_error
    with _state_lock:
        if _ee_initialized:
            return
        if ee is None:
            raise RuntimeError(_ee_error or "earthengine-api is not installed")
        try:
            ee.Initialize(project=EE_PROJECT)
        except Exception as exc:
            _ee_error = type(exc).__name__
            app.logger.exception("Earth Engine initialization failed")
            raise RuntimeError("ambient Earth Engine credentials are unavailable") from exc
        _ee_initialized = True
        _ee_error = None


def _request_value(payload: dict[str, Any], name: str, default: Any = None) -> Any:
    return payload.get(name, default)


def _query_float(payload: dict[str, Any], name: str) -> float:
    raw = _request_value(payload, name)
    if raw is None:
        raise RequestValidationError(f"Missing required query parameter: {name}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"Query parameter {name} must be numeric") from exc
    if not np.isfinite(value):
        raise RequestValidationError(f"Query parameter {name} must be finite")
    return value


def _parse_bbox(payload: dict[str, Any]) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = (
        _query_float(payload, name) for name in ("minx", "miny", "maxx", "maxy")
    )
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise RequestValidationError("Expected WGS84 bounds with minx < maxx and miny < maxy")
    if maxx - minx > MAX_ROI_SPAN_DEGREES or maxy - miny > MAX_ROI_SPAN_DEGREES:
        raise RequestValidationError(
            f"Requested area is too large; longitude and latitude spans must each be <= "
            f"{MAX_ROI_SPAN_DEGREES:g} degrees"
        )
    return minx, miny, maxx, maxy


def _iso_date(raw: str, name: str) -> str:
    if not isinstance(raw, str):
        raise RequestValidationError(f"Query parameter {name} must use YYYY-MM-DD")
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError as exc:
        raise RequestValidationError(f"Query parameter {name} must use YYYY-MM-DD") from exc


def _date_window(payload: dict[str, Any]) -> tuple[str, str]:
    single_date = _request_value(payload, "date")
    if single_date:
        start_value = date.fromisoformat(_iso_date(single_date, "date"))
        return start_value.isoformat(), (start_value + timedelta(days=1)).isoformat()
    start = _iso_date(_request_value(payload, "start", DEFAULT_START_DATE), "start")
    end = _iso_date(_request_value(payload, "end", DEFAULT_END_DATE), "end")
    if start >= end:
        raise RequestValidationError("start must be earlier than end")
    return start, end


def _cloud_limit(payload: dict[str, Any]) -> float:
    raw = _request_value(payload, "max_cloud")
    if raw is None:
        value = DEFAULT_MAX_CLOUD
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise RequestValidationError("max_cloud must be numeric") from exc
    if not np.isfinite(value) or not 0 <= value <= 100:
        raise RequestValidationError("max_cloud must be between 0 and 100")
    return value


def _deterministic_scene(product: str, roi: Any, start: str, end: str, max_cloud: float):
    """Prefer least-cloudy, then earliest scene using an explicit sort key."""
    collection = (
        ee.ImageCollection(product)
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )

    def add_selection_key(image):
        cloud = ee.Number(image.get("CLOUDY_PIXEL_PERCENTAGE"))
        timestamp = ee.Number(image.get("system:time_start"))
        # A padded string avoids floating-point collisions in a numeric
        # composite key and makes the tie-break order explicit.
        selection_key = (
            cloud.format("%010.6f")
            .cat("_")
            .cat(timestamp.format("%013d"))
            .cat("_")
            .cat(ee.String(image.get("system:index")))
        )
        return image.set("_ersrr_selection_key", selection_key)

    ranked = collection.map(add_selection_key).sort("_ersrr_selection_key")
    count = int(ranked.size().getInfo())
    if count == 0:
        raise LookupError(
            f"No {product} scene intersects the ROI in {start}..{end} with cloud <= {max_cloud:g}%"
        )
    return ee.Image(ranked.first()).select(list(BAND_ORDER))


def _download_scene(image: Any, roi: Any, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    resampling = config["input_resampling"]
    download_image = image if resampling == "nearest" else image.resample(resampling)
    download_url = download_image.getDownloadURL(
        {
            "region": roi,
            "scale": float(config["input_resolution_m"]),
            "format": "GEO_TIFF",
            "bands": list(BAND_ORDER),
        }
    )
    response = requests.get(download_url, timeout=(15, 180))
    response.raise_for_status()
    SENTINEL_PATH.write_bytes(response.content)
    with rasterio.open(SENTINEL_PATH) as source:
        if source.count != len(BAND_ORDER):
            raise RuntimeError(f"Expected {len(BAND_ORDER)} downloaded bands; got {source.count}")
        tile = source.read().transpose(1, 2, 0).astype("float32")
        profile = source.profile.copy()
    return np.nan_to_num(tile, nan=0.0, posinf=0.0, neginf=0.0), profile


def _pad_image(image: np.ndarray, tile_size: int) -> tuple[np.ndarray, int, int]:
    height, width, _ = image.shape
    pad_height = (-height) % tile_size
    pad_width = (-width) % tile_size
    padded = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant")
    return padded.astype("float32"), height, width


def _predict_image(model: Any, config: dict[str, Any], image: np.ndarray) -> np.ndarray:
    inference_tile_size = int(config["inference_tile_size"])
    nodata_value = float(config["input_nodata_value"])
    valid = np.all(np.isfinite(image), axis=-1) & np.all(image != nodata_value, axis=-1)
    padded, original_height, original_width = _pad_image(image, inference_tile_size)
    padded_valid = np.pad(
        valid,
        ((0, padded.shape[0] - valid.shape[0]), (0, padded.shape[1] - valid.shape[1])),
        mode="constant",
        constant_values=False,
    )
    output = np.zeros(padded.shape[:2], dtype="float32")
    for row in range(0, padded.shape[0], inference_tile_size):
        for column in range(0, padded.shape[1], inference_tile_size):
            tile = padded[row : row + inference_tile_size, column : column + inference_tile_size]
            output[row : row + inference_tile_size, column : column + inference_tile_size] = predict_tile(
                model, config, tile
            )
    output[~padded_valid] = 0.0
    return output[:original_height, :original_width]


def _reproject_to_web_mercator(data: np.ndarray, profile: dict[str, Any]):
    source_crs = profile.get("crs")
    if source_crs is None:
        raise RuntimeError("Downloaded Sentinel-2 raster has no CRS")
    bounds = rasterio.transform.array_bounds(
        profile["height"], profile["width"], profile["transform"]
    )
    transform, width, height = calculate_default_transform(
        source_crs,
        "EPSG:3857",
        profile["width"],
        profile["height"],
        *bounds,
    )
    destination = np.zeros((data.shape[0], height, width), dtype=data.dtype)
    for band_index in range(data.shape[0]):
        reproject(
            source=data[band_index],
            destination=destination[band_index],
            src_transform=profile["transform"],
            src_crs=source_crs,
            dst_transform=transform,
            dst_crs="EPSG:3857",
            resampling=Resampling.nearest,
        )
    output_profile = profile.copy()
    output_profile.update(
        crs="EPSG:3857", transform=transform, width=width, height=height, count=4, dtype="uint8"
    )
    return destination, output_profile


def _publish_prediction(probability: np.ndarray, profile: dict[str, Any], threshold: float) -> str:
    clipped = np.clip(probability, 0.0, 1.0)
    active = clipped >= threshold
    red = np.where(active, 255, 0).astype("uint8")
    green = np.zeros_like(red)
    blue = np.zeros_like(red)
    confidence = np.clip((clipped - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    alpha = np.where(active, 80 + confidence * 175, 0).astype("uint8")
    rgba = np.stack([red, green, blue, alpha], axis=0)
    web_data, web_profile = _reproject_to_web_mercator(rgba, profile)

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    published_id = uuid.uuid4().hex[:16]
    temporary_path = PRED_PATH.with_name(f".{PRED_PATH.stem}.{published_id}.tmp.tif")
    try:
        with rasterio.open(temporary_path, "w", **web_profile) as destination:
            destination.write(web_data)
        with _state_lock:
            global prediction_data, prediction_profile, prediction_id
            os.replace(temporary_path, PRED_PATH)
            prediction_data = web_data
            prediction_profile = web_profile
            prediction_id = published_id
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return published_id


@app.get("/health")
def health():
    artifact = _artifact_status()
    earth_engine = _earth_engine_status()
    ready = artifact["status"] == "ready" and earth_engine["status"] == "ready"
    return jsonify(
        {
            "status": "ready" if ready else "degraded",
            "artifact": artifact,
            "earth_engine": earth_engine,
            "prediction": {"available": prediction_data is not None, "id": prediction_id},
            "band_order": list(BAND_ORDER),
            "max_roi_span_degrees": MAX_ROI_SPAN_DEGREES,
        }
    )


@app.get("/prediction/<requested_id>.tif")
def get_prediction(requested_id: str):
    with _state_lock:
        if prediction_id is None or requested_id != prediction_id or not PRED_PATH.is_file():
            return _json_error("Prediction is unavailable or superseded", 404)
        try:
            payload = PRED_PATH.read_bytes()
        except OSError:
            return _json_error("Prediction is unavailable", 404)
    return send_file(
        BytesIO(payload),
        mimetype="image/tiff",
        download_name=f"ersrr-{requested_id}.tif",
    )


@app.get("/tiles/<requested_id>/<int:z>/<int:x>/<int:y>.png")
def get_tile(requested_id: str, z: int, x: int, y: int):
    with _state_lock:
        if prediction_id is None or requested_id != prediction_id:
            return _json_error("Prediction is unavailable or superseded", 404)
        if prediction_data is None or prediction_profile is None:
            return _json_error("Prediction is unavailable", 404)
        data = prediction_data
        profile = prediction_profile

    bounds = mercantile.xy_bounds(x, y, z)
    transform = profile["transform"]

    def world_to_pixel(world_x: float, world_y: float) -> tuple[int, int]:
        column = int((world_x - transform.c) / transform.a)
        row = int((world_y - transform.f) / transform.e)
        return column, row

    column_min, row_max = world_to_pixel(bounds.left, bounds.bottom)
    column_max, row_min = world_to_pixel(bounds.right, bounds.top)
    column_min = max(column_min, 0)
    row_min = max(row_min, 0)
    column_max = min(column_max, data.shape[2])
    row_max = min(row_max, data.shape[1])
    tile = data[:, row_min:row_max, column_min:column_max]
    if tile.size == 0:
        return "", 204

    rgba = np.transpose(tile, (1, 2, 0))
    tile_image = Image.fromarray(rgba, mode="RGBA").resize(
        (WEB_TILE_SIZE, WEB_TILE_SIZE), Image.Resampling.BILINEAR
    )
    buffer = BytesIO()
    tile_image.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


@app.post("/sentinel")
def sentinel():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("Expected an application/json request body", 415)
    try:
        bounds = _parse_bbox(payload)
        start, end = _date_window(payload)
        max_cloud = _cloud_limit(payload)
    except RequestValidationError as exc:
        return _json_error(str(exc), 400)

    try:
        model, config = _ensure_artifact()
    except ArtifactError as exc:
        return _json_error(f"Model artifact unavailable: {exc}", 503, component="artifact")
    try:
        _ensure_earth_engine()
    except RuntimeError as exc:
        return _json_error(
            f"Earth Engine unavailable; configure non-interactive credentials: {exc}",
            503,
            component="earth_engine",
        )

    product = config["input_product"]
    roi = ee.Geometry.Rectangle(list(bounds), proj="EPSG:4326", geodesic=False)
    # The service publishes one current prediction and uses fixed local paths.
    # Serialize inference so concurrent requests cannot mix rasters or metadata.
    with _inference_lock:
        try:
            image = _deterministic_scene(product, roi, start, end, max_cloud)
            metadata = image.toDictionary(
                ["system:id", "system:index", "system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]
            ).getInfo()
            sentinel_image, profile = _download_scene(image, roi, config)
            probability = _predict_image(model, config, sentinel_image)
            decision_threshold = float(config["decision_threshold"])
            published_id = _publish_prediction(probability, profile, decision_threshold)
            map_id = image.getMapId({"min": 0, "max": 3000, "bands": ["B4", "B3", "B2"]})
        except LookupError as exc:
            return _json_error(str(exc), 404, component="earth_engine")
        except (requests.RequestException, rasterio.errors.RasterioError, RuntimeError, ValueError) as exc:
            app.logger.exception("Sentinel-2 inference failed")
            return _json_error(f"Sentinel-2 inference failed: {type(exc).__name__}", 502)
        except Exception as exc:
            app.logger.exception("Unexpected Sentinel-2 inference failure")
            return _json_error(f"Sentinel-2 inference failed: {type(exc).__name__}", 502)

    acquisition_ms = metadata.get("system:time_start")
    acquisition_time = None
    if isinstance(acquisition_ms, (int, float)):
        acquisition_time = datetime.fromtimestamp(acquisition_ms / 1000, tz=timezone.utc).isoformat()
    return jsonify(
        {
            "tile_url": map_id["tile_fetcher"].url_format,
            "prediction_tile_url": f"/tiles/{published_id}/{{z}}/{{x}}/{{y}}.png",
            "prediction_download_url": f"/prediction/{published_id}.tif",
            "scene_id": metadata.get("system:id") or metadata.get("system:index"),
            "acquisition_time": acquisition_time,
            "cloud_percentage": metadata.get("CLOUDY_PIXEL_PERCENTAGE"),
            "product": product,
            "band_order": list(BAND_ORDER),
            "decision_threshold": decision_threshold,
            "artifact_status": config.get("status"),
        }
    )


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
