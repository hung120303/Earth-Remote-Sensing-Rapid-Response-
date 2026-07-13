"""Interpretable scene scores derived from ERSRR v4 segmentation logits."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import ndimage
from scipy.special import expit

TOP_FRACTIONS = (0.001, 0.0025, 0.005, 0.01, 0.02)
LOCAL_WINDOWS = (3, 7, 15, 31)
COMPONENT_THRESHOLDS = (0.5, 0.7, 0.8, 0.9)
CURRENT_SCORE = "current_top_0_5pct_blend"


def _suffix(value: float) -> str:
    return f"{100.0 * value:g}".replace(".", "_")


CANDIDATE_FORMULAS: Mapping[str, str] = {
    CURRENT_SCORE: "0.8 * mean(top 0.5% observable logits) + 0.2 * max(observable logit)",
    **{
        f"top_logits_{_suffix(fraction)}pct_mean": (
            f"mean(top {100.0 * fraction:g}% observable segmentation logits)"
        )
        for fraction in TOP_FRACTIONS
    },
    **{
        f"local_probability_mean_k{window}": (
            f"maximum observability-normalized {window}x{window} mean segmentation probability"
        )
        for window in LOCAL_WINDOWS
    },
    **{
        f"component_excess_p{int(threshold * 100)}": (
            "maximum eight-connected "
            f"sum(probability - {threshold:g}) for probability >= {threshold:g}"
        )
        for threshold in COMPONENT_THRESHOLDS
    },
}


def _masked_top(logits: np.ndarray, observable: np.ndarray, fraction: float) -> np.ndarray:
    flat = np.asarray(logits, dtype=np.float64).ravel()
    valid = np.asarray(observable, dtype=bool).ravel()
    if flat.shape != valid.shape or not np.any(valid):
        raise ValueError("Scene scoring requires aligned logits with observable support")
    masked = np.where(valid, flat, -1e4)
    count = max(1, int(masked.size * fraction))
    return np.partition(masked, masked.size - count)[-count:]


def _local_probability_score(
    probability: np.ndarray, observable: np.ndarray, window: int
) -> float:
    valid = np.asarray(observable, dtype=np.float64)
    numerator = ndimage.uniform_filter(
        np.asarray(probability, dtype=np.float64) * valid,
        size=window,
        mode="constant",
        cval=0.0,
    )
    support = ndimage.uniform_filter(valid, size=window, mode="constant", cval=0.0)
    eligible = support >= 0.75
    if not np.any(eligible):
        eligible = support > 0
    local_mean = np.divide(
        numerator,
        support,
        out=np.zeros_like(numerator),
        where=support > 0,
    )
    return float(np.max(local_mean[eligible]))


def _component_excess_score(
    probability: np.ndarray, observable: np.ndarray, threshold: float
) -> float:
    values = np.asarray(probability, dtype=np.float64)
    candidate = (values >= threshold) & np.asarray(observable, dtype=bool)
    labels, count = ndimage.label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0.0
    excess = np.maximum(values - threshold, 0.0)
    component_mass = ndimage.sum(excess, labels=labels, index=np.arange(1, count + 1))
    return float(np.max(component_mass))


def scene_score_candidates(
    segmentation_logits: np.ndarray, observable: np.ndarray
) -> dict[str, float]:
    """Return the frozen, label-blind v4.2 scene-score candidate family."""
    logits = np.asarray(segmentation_logits, dtype=np.float64).squeeze()
    valid = np.asarray(observable, dtype=bool).squeeze()
    if logits.ndim != 2 or logits.shape != valid.shape or not np.all(np.isfinite(logits)):
        raise ValueError("Scene score inputs must be finite aligned 2D arrays")
    top_values = {
        fraction: _masked_top(logits, valid, fraction) for fraction in TOP_FRACTIONS
    }
    probability = expit(logits)
    current_top = top_values[0.005]
    scores = {
        CURRENT_SCORE: float(0.8 * np.mean(current_top) + 0.2 * np.max(current_top)),
        **{
            f"top_logits_{_suffix(fraction)}pct_mean": float(np.mean(values))
            for fraction, values in top_values.items()
        },
        **{
            f"local_probability_mean_k{window}": _local_probability_score(
                probability, valid, window
            )
            for window in LOCAL_WINDOWS
        },
        **{
            f"component_excess_p{int(threshold * 100)}": _component_excess_score(
                probability, valid, threshold
            )
            for threshold in COMPONENT_THRESHOLDS
        },
    }
    if tuple(scores) != tuple(CANDIDATE_FORMULAS) or not np.all(
        np.isfinite(list(scores.values()))
    ):
        raise ValueError("V4.2 scene-score candidate schema is invalid")
    return scores
