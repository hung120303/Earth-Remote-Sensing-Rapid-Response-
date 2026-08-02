#!/usr/bin/env python3
"""Materialize the already-selected folds-3/4 Gaussian+DOFA champion scores."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from evaluate_mars_dofa_anchored_protected_ensemble import (  # noqa: E402
    fixed_dofa_scores,
    protected_residual_ensemble,
)
from evaluate_mars_dofa_gaussian_protected_ensemble import (  # noqa: E402
    gaussian_local_candidate,
    load_gaussian_scene_cache,
    restricted_selection_values,
    validate_fixed_dofa_result,
    validate_gaussian_replicate_result,
)
from train_mars_dofa_v2_scene_probe import align_features  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_dofa_gaussian_protected_ensemble_protocol.json")
DEFAULT_RESULT = Path("reports/experiments/mars_dofa_gaussian_protected_ensemble.json")
DEFAULT_OUTPUT = Path("outputs/mars_dofa_gaussian_champion_folds34_scores.npz")
DEFAULT_RECEIPT = Path("reports/experiments/mars_dofa_gaussian_champion_cache.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--result", default=DEFAULT_RESULT.as_posix())
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    result_path = (ROOT / args.result).resolve()
    output = (ROOT / args.output).resolve()
    receipt_path = (ROOT / args.receipt).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("all_promotion_gates_pass") is not True:
        raise ValueError("Gaussian+DOFA result was not promoted")
    selected_strength = float(result["selected"]["gaussian_strength"])
    paths = {
        name: (ROOT / contract["path"]).resolve()
        for name, contract in protocol["inputs"].items()
    }
    for name, contract in protocol["inputs"].items():
        if sha256(paths[name]) != contract["sha256"]:
            raise ValueError(f"Champion-cache frozen input mismatch: {name}")
    values = restricted_selection_values(paths["inner"], paths["current_scores"])
    validate_fixed_dofa_result(paths["dofa_result"], protocol["fixed_dofa"])
    encoded, names = align_features(paths["dofa_features"], values)
    dofa = fixed_dofa_scores(protocol, values, encoded, names)
    del encoded
    gc.collect()
    raw_gaussian = load_gaussian_scene_cache(
        paths["gaussian_scene_cache"],
        values,
        protocol["gaussian_cache_protocol_sha256"],
    )
    _, eligible = validate_gaussian_replicate_result(
        paths["gaussian_cache_result"],
        paths["gaussian_scene_cache"],
        protocol["gaussian_cache_protocol_sha256"],
    )
    if selected_strength not in eligible:
        raise ValueError("Selected Gaussian strength is not reproducibility-eligible")
    gate = float(protocol["architecture"]["final_protection_gate"])
    gaussian = gaussian_local_candidate(
        values["current"], raw_gaussian, strength=selected_strength, gate=gate
    )
    champion = protected_residual_ensemble(
        values["current"], gaussian, dofa, gate=gate, anchored_multiplier=1.0
    )
    arrays = {
        "schema_version": np.asarray(1, dtype=np.uint8),
        "sample_ids": values["sample_ids"].astype(str),
        "labels": values["labels"].astype(np.uint8),
        "sensors": values["sensors"].astype(np.uint8),
        "groups": values["groups"].astype(str),
        "folds": values["folds"].astype(np.uint8),
        "released_primary_scores": values["primary"].astype(np.float64),
        "spatial_prithvi_scores": values["current"].astype(np.float64),
        "champion_scores": champion.astype(np.float64),
        "gaussian_strength": np.asarray(selected_strength, dtype=np.float64),
        "protection_gate": np.asarray(gate, dtype=np.float64),
        "source_result_sha256": np.asarray(sha256(result_path)),
        "source_protocol_sha256": np.asarray(sha256(protocol_path)),
    }
    if len(set(arrays["sample_ids"].tolist())) != arrays["sample_ids"].size:
        raise ValueError("Champion score identities are not unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)
    receipt = {
        "schema_version": 1,
        "scope": "deterministic cache of the already-selected Gaussian+DOFA folds-3/4 champion",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(champion.size),
        "folds": sorted(set(map(int, values["folds"]))),
        "gaussian_strength": selected_strength,
        "protection_gate": gate,
        "cache": {
            "path": output.relative_to(ROOT).as_posix(),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "tracked": False,
        },
        "inputs": {
            "result": {
                "path": result_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(result_path),
            },
            "protocol": {
                "path": protocol_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(protocol_path),
            },
        },
        "held_outcomes_accessed": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

