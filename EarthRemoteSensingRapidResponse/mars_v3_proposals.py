"""Connected-proposal extraction and descriptors for ERSRR MARS v3 stage two."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from mars_v3_model import CLOUD_INDEX, INPUT_CHANNELS, WIND_U_INDEX, WIND_V_INDEX

PROPOSAL_THRESHOLDS = (0.30, 0.50, 0.70)
MINIMUM_PROPOSAL_PIXELS = 10
POSITIVE_MINIMUM_IOU = 0.05
POSITIVE_MINIMUM_TRUTH_RECALL = 0.10
BAND_NAMES = ("B02", "B03", "B04", "B08", "B11", "B12")


@dataclass(frozen=True)
class Proposal:
    mask: np.ndarray
    source_threshold: float

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))


def _intersection_over_union(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return 0.0 if union == 0 else intersection / union


def extract_proposals(
    probability: np.ndarray,
    observable: np.ndarray,
    *,
    thresholds: tuple[float, ...] = PROPOSAL_THRESHOLDS,
    minimum_pixels: int = MINIMUM_PROPOSAL_PIXELS,
    duplicate_iou: float = 0.85,
) -> list[Proposal]:
    """Extract high-recall 8-connected components across fixed score levels."""
    score = np.asarray(probability, dtype=np.float32)
    valid = np.asarray(observable, dtype=bool)
    if score.shape != valid.shape or score.ndim != 2:
        raise ValueError("Probability and observable arrays must be matching 2D grids")
    if minimum_pixels <= 0:
        raise ValueError("minimum_pixels must be positive")
    proposals: list[Proposal] = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for threshold in sorted(set(float(value) for value in thresholds), reverse=True):
        labels, count = ndimage.label((score >= threshold) & valid, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(labels.ravel())
        components = [
            labels == label
            for label in range(1, count + 1)
            if int(sizes[label]) >= minimum_pixels
        ]
        components.sort(key=lambda mask: float(np.max(score[mask])), reverse=True)
        for mask in components:
            if any(_intersection_over_union(mask, prior.mask) >= duplicate_iou for prior in proposals):
                continue
            proposals.append(Proposal(mask=mask, source_threshold=threshold))
    proposals.sort(
        key=lambda item: (float(np.max(score[item.mask])), item.area, item.source_threshold),
        reverse=True,
    )
    return proposals


def label_proposal(proposal: Proposal, truth: np.ndarray) -> dict[str, Any]:
    """Assign positive/negative/ignore without treating partial overlaps as negatives."""
    target = np.asarray(truth, dtype=bool)
    if target.shape != proposal.mask.shape:
        raise ValueError("Proposal and truth grids must match")
    intersection = int(np.count_nonzero(proposal.mask & target))
    union = int(np.count_nonzero(proposal.mask | target))
    truth_area = int(np.count_nonzero(target))
    iou = 0.0 if union == 0 else intersection / union
    proposal_precision = 0.0 if proposal.area == 0 else intersection / proposal.area
    truth_recall = 0.0 if truth_area == 0 else intersection / truth_area
    if iou >= POSITIVE_MINIMUM_IOU and truth_recall >= POSITIVE_MINIMUM_TRUTH_RECALL:
        state = "positive"
        label: int | None = 1
    elif intersection == 0:
        state = "negative"
        label = 0
    else:
        state = "ignore_ambiguous_overlap"
        label = None
    return {
        "state": state,
        "label": label,
        "intersection_pixels": intersection,
        "iou": iou,
        "proposal_precision": proposal_precision,
        "truth_recall": truth_recall,
    }


def _summary(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.quantile(values, 0.90)),
        float(np.max(values)),
    )


def proposal_feature_names(decoder_channels: int = 128) -> list[str]:
    names = [
        "source_threshold",
        "area_pixels",
        "area_fraction_observable",
        "score_mean",
        "score_std",
        "score_p90",
        "score_max",
        "bbox_height",
        "bbox_width",
        "bbox_fill_fraction",
        "perimeter_pixels",
        "perimeter_sqrt_area_ratio",
        "centroid_x",
        "centroid_y_north",
        "variance_x",
        "variance_y",
        "covariance_xy",
        "major_variance",
        "minor_variance",
        "eccentricity",
        "major_axis_wind_alignment",
        "wind_speed",
        "mbmp_mean",
        "mbmp_std",
        "mbmp_p90",
        "mbmp_max",
        "mbmp_ring_contrast",
        "absolute_change_mean",
        "cloud_fraction_component",
        "cloud_fraction_ring",
    ]
    for band in BAND_NAMES:
        names.extend([f"change_{band}_mean", f"change_{band}_ring_contrast"])
    names.extend(f"decoder_mean_{index:03d}" for index in range(decoder_channels))
    names.extend(f"decoder_max_{index:03d}" for index in range(decoder_channels))
    return names


def proposal_features(
    proposal: Proposal,
    probability: np.ndarray,
    inputs: np.ndarray,
    observable: np.ndarray,
    decoder_features: np.ndarray,
) -> np.ndarray:
    """Build an interpretable component descriptor plus learned decoder context."""
    score = np.asarray(probability, dtype=np.float32)
    values = np.asarray(inputs, dtype=np.float32)
    valid = np.asarray(observable, dtype=bool)
    dense = np.asarray(decoder_features, dtype=np.float32)
    if values.shape[0] != len(INPUT_CHANNELS) or values.shape[1:] != score.shape:
        raise ValueError("Input tensor does not match the frozen 16-channel proposal contract")
    if dense.ndim != 3 or dense.shape[1:] != score.shape:
        raise ValueError("Decoder feature grid must be CxHxW and match the probability grid")
    mask = proposal.mask & valid
    if not np.any(mask):
        raise ValueError("Proposal contains no observable pixels")
    ring = ndimage.binary_dilation(mask, iterations=5) & ~mask & valid
    rows, columns = np.nonzero(mask)
    height, width = score.shape
    x = (columns.astype(np.float64) / max(width - 1, 1)) * 2.0 - 1.0
    y = 1.0 - (rows.astype(np.float64) / max(height - 1, 1)) * 2.0
    centroid_x = float(np.mean(x))
    centroid_y = float(np.mean(y))
    centered = np.stack([x - centroid_x, y - centroid_y], axis=0)
    covariance = centered @ centered.T / max(centered.shape[1], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major = float(max(eigenvalues[order[0]], 0.0))
    minor = float(max(eigenvalues[order[1]], 0.0))
    major_axis = eigenvectors[:, order[0]]
    wind = np.asarray(
        [
            float(np.mean(values[WIND_U_INDEX])) * 8.0,
            float(np.mean(values[WIND_V_INDEX])) * 8.0,
        ]
    )
    wind_speed = float(np.linalg.norm(wind))
    alignment = 0.0 if wind_speed < 1e-6 else float(abs(np.dot(major_axis, wind / wind_speed)))
    eccentricity = float(np.sqrt(max(0.0, 1.0 - minor / max(major, 1e-8))))
    row_min, row_max = int(rows.min()), int(rows.max())
    column_min, column_max = int(columns.min()), int(columns.max())
    bbox_height = row_max - row_min + 1
    bbox_width = column_max - column_min + 1
    boundary = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
    perimeter = int(np.count_nonzero(boundary))
    score_mean, score_std, score_p90, score_max = _summary(score[mask])
    mbmp_mean, mbmp_std, mbmp_p90, mbmp_max = _summary(values[0][mask])
    mbmp_ring = float(np.mean(values[0][ring])) if np.any(ring) else mbmp_mean
    change = values[1:7] - values[7:13]
    change_features: list[float] = []
    for band_change in change:
        component_mean = float(np.mean(band_change[mask]))
        ring_mean = float(np.mean(band_change[ring])) if np.any(ring) else component_mean
        change_features.extend([component_mean, component_mean - ring_mean])
    cloud = values[CLOUD_INDEX]
    decoder_mean = dense[:, mask].mean(axis=1)
    decoder_max = dense[:, mask].max(axis=1)
    base = [
        proposal.source_threshold,
        float(proposal.area),
        float(proposal.area / max(np.count_nonzero(valid), 1)),
        score_mean,
        score_std,
        score_p90,
        score_max,
        float(bbox_height),
        float(bbox_width),
        float(proposal.area / (bbox_height * bbox_width)),
        float(perimeter),
        float(perimeter / np.sqrt(max(proposal.area, 1))),
        centroid_x,
        centroid_y,
        float(covariance[0, 0]),
        float(covariance[1, 1]),
        float(covariance[0, 1]),
        major,
        minor,
        eccentricity,
        alignment,
        wind_speed,
        mbmp_mean,
        mbmp_std,
        mbmp_p90,
        mbmp_max,
        mbmp_mean - mbmp_ring,
        float(np.mean(np.abs(change[:, mask]))),
        float(np.mean(cloud[mask])),
        float(np.mean(cloud[ring])) if np.any(ring) else float(np.mean(cloud[mask])),
        *change_features,
    ]
    result = np.concatenate(
        [
            np.asarray(base, dtype=np.float32),
            decoder_mean.astype(np.float32),
            decoder_max.astype(np.float32),
        ]
    )
    expected = len(proposal_feature_names(dense.shape[0]))
    if result.size != expected or not np.all(np.isfinite(result)):
        raise ValueError(
            f"Proposal descriptor contract failed: expected {expected}, got {result.size}"
        )
    return result
