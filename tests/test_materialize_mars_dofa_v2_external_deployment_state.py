from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from materialize_mars_dofa_v2_external_deployment_state import (  # noqa: E402
    apply_normalization,
    endpoint_probability,
    external_mean_logit_score,
    fit_normalization,
    validate_protocol,
)


def endpoint(intercept: float) -> dict[str, object]:
    return {
        "raw_center": np.zeros(2, dtype=np.float32),
        "raw_scale": np.ones(2, dtype=np.float32),
        "projection_components": scipy.sparse.eye(2, dtype=np.float32, format="csr"),
        "projected_center": np.zeros(2, dtype=np.float32),
        "projected_scale": np.ones(2, dtype=np.float32),
        "classifier_coef": np.asarray([1.0, -1.0], dtype=np.float64),
        "classifier_intercept": intercept,
    }


def frozen_protocol() -> dict[str, object]:
    return {
        "deployment_rule": {
            "feature_set": "change_extreme",
            "C": 0.01,
            "projection_dimension": 2048,
            "projection_seeds": [20260780, 20260781, 20260782, 20260783, 20260784],
            "normalization_mode": "global_train_fitted",
            "folds": [3, 4],
            "endpoint_aggregation": "equal_mean_logit_all_10_endpoints",
        },
        "forbidden_access": {
            "folds": [0, 1, 2],
            "official_test": True,
            "external_cohort": True,
        },
    }


def test_source_fitted_normalization_replays_without_target_statistics() -> None:
    source = np.asarray([[1.0, 4.0], [3.0, 8.0]], dtype=np.float32)
    target = np.asarray([[100.0, -10.0]], dtype=np.float32)
    center, scale = fit_normalization(source)
    observed = apply_normalization(target, center, scale)
    assert np.allclose(center, [2.0, 6.0])
    assert np.allclose(scale, [1.0, 2.0])
    assert np.allclose(observed, [[98.0, -8.0]])


def test_external_rule_requires_and_equal_weights_all_ten_endpoints() -> None:
    features = np.asarray([[2.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    payload = {"endpoints": [endpoint(-0.5)] * 5 + [endpoint(0.5)] * 5}
    score = external_mean_logit_score(payload, features)
    # The endpoint intercepts cancel exactly in equal logit aggregation.
    assert np.allclose(score, [1.0 / (1.0 + np.exp(-1.0)), 0.5])
    payload["endpoints"] = payload["endpoints"][:-1]
    with pytest.raises(ValueError, match="exactly ten"):
        external_mean_logit_score(payload, features)


def test_endpoint_probability_rejects_non_sparse_projection() -> None:
    state = endpoint(0.0)
    state["projection_components"] = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match="remain sparse"):
        endpoint_probability(state, np.ones((1, 2), dtype=np.float32))


def test_protocol_freezes_rule_and_forbidden_access() -> None:
    protocol = frozen_protocol()
    validate_protocol(protocol)
    protocol["deployment_rule"]["endpoint_aggregation"] = "mean_probability"  # type: ignore[index]
    with pytest.raises(ValueError, match="differs"):
        validate_protocol(protocol)
