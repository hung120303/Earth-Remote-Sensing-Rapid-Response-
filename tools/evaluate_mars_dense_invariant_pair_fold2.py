#!/usr/bin/env python3
"""Run the one-shot fold-2 confirmation for the fixed invariant residual pair."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402
from explore_mars_dense_invariant_nonlinear_residuals import (  # noqa: E402
    transform_residual,
)
from explore_mars_dense_invariant_univariate import (  # noqa: E402
    ap_views,
    rank_normalize_by_domain,
)
from explore_mars_dense_sparse_invariant_ensemble import (  # noqa: E402
    evaluate_views,
    point_gates,
)
from train_mars_dense_ap_residual_ranker import (  # noqa: E402
    load_cache,
    logit,
    sigmoid,
)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_scene_ranker import metric_summary  # noqa: E402
from train_mars_unseen_low_prevalence_router import (  # noqa: E402
    low_prevalence_mask,
)


DEFAULT_PROTOCOL = Path(
    "configs/mars_dense_invariant_pair_fold2_protocol.json"
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def fixed_scores(
    matrix: np.ndarray,
    metadata: dict[str, np.ndarray],
    protocol: dict[str, Any],
) -> np.ndarray:
    folds = metadata["folds"].astype(np.uint8)
    sensors = metadata["sensors"].astype(np.uint8)
    base_scores = metadata["exact_base_scores"].astype(np.float64)
    names = metadata["feature_names"].astype(str)
    residuals: dict[str, np.ndarray] = {}
    for name in ("anchor", "booster"):
        spec = protocol["candidate"]["components"][name]
        feature_index = int(spec["feature_index"])
        if str(names[feature_index]) != str(spec["feature_name"]):
            raise ValueError(f"Fixed fold-2 {name} feature name mismatch")
        direction = int(spec["direction"])
        if direction not in (-1, 1):
            raise ValueError("Fold-2 candidate direction must be exactly -1 or +1")
        normalized = rank_normalize_by_domain(
            matrix[:, feature_index],
            folds,
            sensors,
        )
        residuals[name] = transform_residual(
            str(spec["transform"]),
            direction * normalized,
            base_scores,
            folds,
            sensors,
            float(protocol["candidate"]["winsor_limit"]),
        )
    return sigmoid(
        logit(base_scores)
        + float(protocol["candidate"]["anchor_strength"])
        * residuals["anchor"]
        + float(protocol["candidate"]["booster_strength"])
        * residuals["booster"]
    )


def verify_receipt(
    protocol: dict[str, Any],
    static_paths: dict[str, Path],
) -> tuple[Path, Path, dict[str, Any]]:
    contract = protocol["fold2_cache"]
    receipt_path = (ROOT / contract["receipt"]).resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError("Frozen fold-2 extraction receipt is unavailable")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if str(receipt["protocol_sha256"]) != sha256(
        static_paths["extractor_protocol"]
    ):
        raise ValueError("Fold-2 receipt does not match the frozen extractor protocol")
    if str(receipt["extractor_sha256"]) != str(
        protocol["extractor"]["sha256"]
    ):
        raise ValueError("Fold-2 receipt extractor identity mismatch")
    if list(map(int, receipt["folds"])) != [2]:
        raise ValueError("Fold-2 receipt contains an unexpected fold")
    if int(receipt["feature_count"]) != 671:
        raise ValueError("Fold-2 receipt feature count mismatch")
    if len(receipt["jobs"]) != 1:
        raise ValueError("Fold-2 receipt has an unexpected job count")
    job = receipt["jobs"][0]
    if (
        int(job["held_fold"]) != 2
        or list(map(int, job["fit_folds"])) != [3, 4]
        or str(job["artifact_sha256"])
        != str(protocol["fold2_cache"]["adapter_sha256"])
    ):
        raise ValueError("Fold-2 receipt adapter contract mismatch")
    features = (ROOT / str(receipt["features"]["path"])).resolve()
    metadata = (ROOT / str(receipt["metadata"]["path"])).resolve()
    if features != (ROOT / contract["features"]).resolve():
        raise ValueError("Fold-2 receipt feature path mismatch")
    if metadata != (ROOT / contract["metadata"]).resolve():
        raise ValueError("Fold-2 receipt metadata path mismatch")
    if sha256(features) != str(receipt["features"]["sha256"]):
        raise ValueError("Fold-2 feature cache hash mismatch")
    if sha256(metadata) != str(receipt["metadata"]["sha256"]):
        raise ValueError("Fold-2 metadata cache hash mismatch")
    return features, metadata, receipt


def view_baseline(
    labels: np.ndarray,
    scores: np.ndarray,
    folds: np.ndarray,
    sensors: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    return {
        "average_precision": ap_views(
            labels[rows],
            scores[rows],
            folds[rows],
            sensors[rows],
        ),
        "matched_fpr_metrics": metric_summary(
            labels[rows],
            scores[rows],
            sensors[rows],
        ),
    }


def reproduce_development(
    protocol: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    matrix, metadata, _columns = load_cache(
        paths["development_features"],
        paths["development_metadata"],
    )
    labels = metadata["labels"].astype(np.uint8)
    sensors = metadata["sensors"].astype(np.uint8)
    groups = metadata["groups"].astype(str)
    folds = metadata["folds"].astype(np.uint8)
    base_scores = metadata["exact_base_scores"].astype(np.float64)
    scores = fixed_scores(matrix, metadata, protocol)
    low_rows = low_prevalence_mask(
        labels,
        groups,
        float(protocol["target_domain"]["maximum_site_positive_rate"]),
    )
    all_rows = np.ones(labels.size, dtype=bool)
    whole = evaluate_views(
        labels,
        base_scores,
        scores,
        folds,
        sensors,
        all_rows,
    )
    low = evaluate_views(
        labels,
        base_scores,
        scores,
        folds,
        sensors,
        low_rows,
    )
    reference = json.loads(
        paths["development_report"].read_text(encoding="utf-8")
    )["summary"]["selected"]
    checks = {
        "candidate_key": str(reference["key"])
        == str(protocol["candidate"]["development_key"]),
        "anchor_strength": float(reference["anchor_strength"])
        == float(protocol["candidate"]["anchor_strength"]),
        "booster_strength": float(reference["booster_strength"])
        == float(protocol["candidate"]["booster_strength"]),
        "whole_ap": bool(
            np.isclose(
                whole["average_precision"]["pooled"],
                reference["whole"]["average_precision"]["pooled"],
                rtol=0.0,
                atol=float(protocol["smoke"]["absolute_tolerance"]),
            )
        ),
        "whole_recall_delta": bool(
            np.isclose(
                whole["matched_fpr_recall_delta"],
                reference["whole"]["matched_fpr_recall_delta"],
                rtol=0.0,
                atol=float(protocol["smoke"]["absolute_tolerance"]),
            )
        ),
        "low_ap": bool(
            np.isclose(
                low["average_precision"]["pooled"],
                reference["low_prevalence"]["average_precision"]["pooled"],
                rtol=0.0,
                atol=float(protocol["smoke"]["absolute_tolerance"]),
            )
        ),
        "low_recall_delta": bool(
            np.isclose(
                low["matched_fpr_recall_delta"],
                reference["low_prevalence"]["matched_fpr_recall_delta"],
                rtol=0.0,
                atol=float(protocol["smoke"]["absolute_tolerance"]),
            )
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Fixed candidate failed development reproduction: {checks}")
    return {
        "ok": True,
        "checks": checks,
        "whole_ap": whole["average_precision"]["pooled"],
        "low_prevalence_ap": low["average_precision"]["pooled"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["evaluator"]["sha256"]:
        raise ValueError("Frozen fold-2 invariant-pair evaluator hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen fold-2 evaluator input mismatch: {name}")
        paths[name] = path
    if args.smoke:
        print(json.dumps(reproduce_development(protocol, paths), indent=2))
        return 0

    output = (ROOT / protocol["output"]).resolve()
    if output.exists():
        raise FileExistsError(
            "Refusing to repeat the one-shot fold-2 confirmation report"
        )
    features_path, metadata_path, receipt = verify_receipt(protocol, paths)
    matrix, metadata, columns = load_cache(features_path, metadata_path)
    labels = metadata["labels"].astype(np.uint8)
    sensors = metadata["sensors"].astype(np.uint8)
    groups = metadata["groups"].astype(str)
    folds = metadata["folds"].astype(np.uint8)
    base_scores = metadata["exact_base_scores"].astype(np.float64)
    if list(map(int, np.unique(folds))) != [2]:
        raise ValueError("One-shot cache contains a non-fold-2 row")
    if len(np.unique(metadata["sample_ids"].astype(str))) != labels.size:
        raise ValueError("One-shot fold-2 sample identities are not unique")
    authorized = set(map(int, columns))
    for name in ("anchor", "booster"):
        if int(protocol["candidate"]["components"][name]["feature_index"]) not in authorized:
            raise ValueError(f"Fixed {name} feature is not authorized")

    scores = fixed_scores(matrix, metadata, protocol)
    maximum_rate = float(
        protocol["target_domain"]["maximum_site_positive_rate"]
    )
    low_rows = low_prevalence_mask(labels, groups, maximum_rate)
    all_rows = np.ones(labels.size, dtype=bool)
    whole = evaluate_views(
        labels,
        base_scores,
        scores,
        folds,
        sensors,
        all_rows,
    )
    low = evaluate_views(
        labels,
        base_scores,
        scores,
        folds,
        sensors,
        low_rows,
    )
    whole_baseline = view_baseline(
        labels,
        base_scores,
        folds,
        sensors,
        all_rows,
    )
    low_baseline = view_baseline(
        labels,
        base_scores,
        folds,
        sensors,
        low_rows,
    )
    checks = point_gates(whole, low, protocol)
    checks.update(
        {
            "whole_fpr_no_worse": bool(
                whole["matched_fpr_metrics"]["false_positive_rate"]
                <= whole_baseline["matched_fpr_metrics"]["false_positive_rate"]
                + float(protocol["gates"]["fpr_absolute_tolerance"])
            ),
            "low_prevalence_fpr_no_worse": bool(
                low["matched_fpr_metrics"]["false_positive_rate"]
                <= low_baseline["matched_fpr_metrics"]["false_positive_rate"]
                + float(protocol["gates"]["fpr_absolute_tolerance"])
            ),
        }
    )
    whole_interval = ap_group_bootstrap(
        labels,
        base_scores,
        scores,
        groups,
        replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["whole_seed"]),
    )
    low_interval = ap_group_bootstrap(
        labels[low_rows],
        base_scores[low_rows],
        scores[low_rows],
        groups[low_rows],
        replicates=int(protocol["bootstrap"]["replicates"]),
        seed=int(protocol["bootstrap"]["low_prevalence_seed"]),
    )
    checks.update(
        {
            "whole_paired_site_ap_lower_strictly_positive": bool(
                whole_interval["lower"] > 0.0
            ),
            "low_prevalence_paired_site_ap_lower_strictly_positive": bool(
                low_interval["lower"] > 0.0
            ),
        }
    )
    passed = all(checks.values())
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "scope": "one-shot fixed-candidate fold-2 confirmation",
        "status": (
            "passed_fold2_confirmation"
            if passed
            else "rejected_on_fold2_confirmation"
        ),
        "decision": (
            "Authorize the fixed pair for separately frozen five-fold and fresh safety work."
            if passed
            else (
                "Retire the fixed pair before fresh or exact-paper evaluation; "
                "resume development research without reusing fold 2."
            )
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "extractor_protocol_sha256": sha256(paths["extractor_protocol"]),
            "extraction_receipt_sha256": sha256(
                (ROOT / protocol["fold2_cache"]["receipt"]).resolve()
            ),
            "features_sha256": sha256(features_path),
            "metadata_sha256": sha256(metadata_path),
            "development_report_sha256": sha256(
                paths["development_report"]
            ),
            "numpy": np.__version__,
        },
        "candidate": protocol["candidate"],
        "rows": {
            "whole": int(labels.size),
            "whole_positive": int(labels.sum()),
            "whole_sites": int(len(np.unique(groups))),
            "low_prevalence": int(np.count_nonzero(low_rows)),
            "low_prevalence_positive": int(labels[low_rows].sum()),
            "low_prevalence_sites": int(len(np.unique(groups[low_rows]))),
        },
        "baseline": {
            "whole": whole_baseline,
            "low_prevalence": low_baseline,
        },
        "candidate_metrics": {
            "whole": whole,
            "low_prevalence": low,
        },
        "paired_site_ap_delta": {
            "whole": whole_interval,
            "low_prevalence": low_interval,
        },
        "promotion_checks": checks,
        "all_promotion_gates_pass": passed,
        "extraction_receipt": receipt,
        "fold2_accessed": True,
        "fresh_inputs_accessed": False,
        "exact_paper_inputs_accessed": False,
    }
    write_json(output, report)
    print(
        json.dumps(
            {
                "ok": passed,
                "whole_ap_delta": whole["average_precision_delta"]["pooled"],
                "low_prevalence_ap_delta": low["average_precision_delta"][
                    "pooled"
                ],
                "whole_interval": whole_interval,
                "low_prevalence_interval": low_interval,
                "checks": checks,
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
