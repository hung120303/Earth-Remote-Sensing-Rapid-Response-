from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_mars_dofa_gaussian_protected_ensemble import (  # noqa: E402
    gaussian_local_candidate,
    load_gaussian_scene_cache,
    validate_fixed_dofa_result,
    validate_gaussian_replicate_result,
)


def test_gaussian_local_candidate_preserves_protection_regions() -> None:
    current = np.asarray([0.10, 0.25, 0.50, 0.90], dtype=np.float64)
    raw = np.asarray([100.0, -100.0, 2.0, -2.0], dtype=np.float64)
    candidate = gaussian_local_candidate(current, raw, strength=1.0, gate=0.25)
    assert candidate[0] == current[0]
    assert np.all(candidate[1:] >= 0.25)
    assert np.isfinite(candidate).all()


def test_gaussian_scene_cache_requires_protocol_binding(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    np.savez_compressed(
        path,
        protocol_sha256=np.asarray("a" * 64),
        sample_ids=np.asarray(["one"]),
        labels=np.asarray([1], dtype=np.uint8),
        sensors=np.asarray([0], dtype=np.uint8),
        groups=np.asarray(["site"]),
        folds=np.asarray([3], dtype=np.uint8),
        base_scores=np.asarray([0.5]),
        raw_scene_logits=np.asarray([1.0], dtype=np.float32),
    )
    values = {
        "sample_ids": np.asarray(["one"]),
        "labels": np.asarray([1], dtype=np.uint8),
        "sensors": np.asarray([0], dtype=np.uint8),
        "groups": np.asarray(["site"]),
        "folds": np.asarray([3], dtype=np.uint8),
        "current": np.asarray([0.5]),
    }
    with pytest.raises(ValueError, match="protocol binding"):
        load_gaussian_scene_cache(path, values, "b" * 64)
    observed = load_gaussian_scene_cache(path, values, "a" * 64)
    assert np.array_equal(observed, np.asarray([1.0]))


def test_fixed_dofa_result_binding(tmp_path: Path) -> None:
    path = tmp_path / "dofa.json"
    path.write_text(
        """{
  "all_promotion_gates_pass": true,
  "fixed_candidate": {
    "feature_set": "change_extreme", "C": 0.01,
    "projection_seeds": [1, 2]
  },
  "selected": {
    "normalization_mode": "global_train_fitted",
    "evaluation": {"spec": {"gate": 0.5, "weight": 0.05}}
  }
}\n""",
        encoding="utf-8",
    )
    fixed = {
        "feature_set": "change_extreme",
        "C": 0.01,
        "projection_seeds": [1, 2],
        "normalization_mode": "global_train_fitted",
        "gate": 0.5,
        "weight": 0.05,
    }
    validate_fixed_dofa_result(path, fixed)
    with pytest.raises(ValueError, match="Fixed DOFA"):
        validate_fixed_dofa_result(path, {**fixed, "weight": 0.1})


def test_gaussian_replicate_result_binds_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache.bin"
    cache.write_bytes(b"cache")
    cache_hash = __import__("hashlib").sha256(b"cache").hexdigest()
    result = tmp_path / "result.json"
    result.write_text(
        """{
  "protocol_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "scene_cache": {"sha256": "%s"},
  "eligible_strengths": [0.05],
  "eligible_for_preregistered_ensemble": true
}\n""" % cache_hash,
        encoding="utf-8",
    )
    _, eligible = validate_gaussian_replicate_result(result, cache, "a" * 64)
    assert eligible == {0.05}
    with pytest.raises(ValueError, match="protocol binding"):
        validate_gaussian_replicate_result(result, cache, "b" * 64)
