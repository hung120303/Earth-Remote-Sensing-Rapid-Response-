from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from train_mars_robust_site_template_ranker import (  # noqa: E402
    CHANNELS,
    build_robust_site_templates,
    compose_robust_site_batch,
)


def test_robust_templates_are_label_free_site_quantiles() -> None:
    images = np.zeros((4, CHANNELS, 64, 64), dtype=np.float32)
    images[0] = 0.0
    images[1] = 2.0
    images[2] = 10.0
    images[3] = 14.0
    templates, counts, inverse, groups = build_robust_site_templates(
        images,
        np.arange(4),
        np.asarray(["a", "a", "b", "b"]),
    )
    assert groups.tolist() == ["a", "b"]
    assert counts.tolist() == [2, 2]
    assert inverse.tolist() == [0, 0, 1, 1]
    assert np.allclose(templates[0, :, 0, 0, 0], [0.5, 1.0, 1.5])
    assert np.allclose(templates[1, :, 0, 0, 0], [11.0, 12.0, 13.0])


def test_robust_feature_contracts_have_expected_shapes_and_finite_values() -> None:
    images = np.zeros((3, CHANNELS, 64, 64), dtype=np.float32)
    images[0] = 1.0
    images[1] = 1.0
    images[2] = 2.0
    templates, _, inverse, _ = build_robust_site_templates(
        images,
        np.arange(3),
        np.asarray(["site", "site", "site"]),
    )
    robust = compose_robust_site_batch(
        images,
        np.arange(3),
        np.arange(3),
        templates,
        inverse,
        "original_robust_residual",
    )
    explicit = compose_robust_site_batch(
        images,
        np.arange(3),
        np.arange(3),
        templates,
        inverse,
        "original_median_residual",
    )
    assert robust.shape == (3, CHANNELS * 2, 64, 64)
    assert explicit.shape == (3, CHANNELS * 3, 64, 64)
    assert np.isfinite(robust).all()
    assert np.isfinite(explicit).all()
    assert float(np.abs(robust[:, CHANNELS:]).max()) <= 1.0
    assert np.allclose(explicit[0, CHANNELS : CHANNELS * 2], 1.0)
    assert np.allclose(explicit[0, CHANNELS * 2 :], 0.0)
