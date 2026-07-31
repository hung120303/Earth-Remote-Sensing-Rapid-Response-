#!/usr/bin/env python3
"""Run the one-shot folds-0/1 confirmation for fixed global shrinkage."""

from __future__ import annotations

import argparse
import json
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
from evaluate_mars_dense_invariant_pair_fold2 import (  # noqa: E402
    fixed_scores,
    view_baseline,
    write_json,
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
from train_mars_unseen_low_prevalence_router import (  # noqa: E402
    low_prevalence_mask,
)


DEFAULT_PROTOCOL = Path(
    "configs/mars_dense_invariant_shrinkage_folds01_protocol.json"
)


def shrunken_scores(
    matrix: np.ndarray,
    metadata: dict[str, np.ndarray],
    protocol: dict[str, Any],
) -> np.ndarray:
    base_scores = metadata["exact_base_scores"].astype(np.float64)
    complete_pair = fixed_scores(matrix, metadata, protocol)
    shrinkage = float(protocol["candidate"]["shrinkage"])
    if not 0.0 < shrinkage < 1.0:
        raise ValueError("Frozen global shrinkage must be inside (0,1)")
    return sigmoid(
        logit(base_scores)
        + shrinkage * (logit(complete_pair) - logit(base_scores))
    )


def verify_receipt(
    protocol: dict[str, Any],
    static_paths: dict[str, Path],
) -> tuple[Path, Path, dict[str, Any]]:
    contract = protocol["folds01_cache"]
    receipt_path = (ROOT / contract["receipt"]).resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(
            "Frozen folds-0/1 extraction receipt is unavailable"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if str(receipt["protocol_sha256"]) != sha256(
        static_paths["extractor_protocol"]
    ):
        raise ValueError(
            "Folds-0/1 receipt does not match the frozen extractor protocol"
        )
    if str(receipt["extractor_sha256"]) != str(
        protocol["extractor"]["sha256"]
    ):
        raise ValueError("Folds-0/1 receipt extractor identity mismatch")
    if list(map(int, receipt["folds"])) != [0, 1]:
        raise ValueError("Folds-0/1 receipt contains unexpected folds")
    if int(receipt["feature_count"]) != 671:
        raise ValueError("Folds-0/1 receipt feature count mismatch")
    if len(receipt["jobs"]) != 2:
        raise ValueError("Folds-0/1 receipt has an unexpected job count")
    seen: set[int] = set()
    for job in receipt["jobs"]:
        held_fold = int(job["held_fold"])
        if (
            held_fold not in {0, 1}
            or held_fold in seen
            or list(map(int, job["fit_folds"])) != [3, 4]
            or str(job["artifact_sha256"])
            != str(contract["adapter_sha256"])
        ):
            raise ValueError("Folds-0/1 receipt adapter contract mismatch")
        seen.add(held_fold)
    if seen != {0, 1}:
        raise ValueError("Folds-0/1 receipt is incomplete")
    features = (ROOT / str(receipt["features"]["path"])).resolve()
    metadata = (ROOT / str(receipt["metadata"]["path"])).resolve()
    if features != (ROOT / contract["features"]).resolve():
        raise ValueError("Folds-0/1 receipt feature path mismatch")
    if metadata != (ROOT / contract["metadata"]).resolve():
        raise ValueError("Folds-0/1 receipt metadata path mismatch")
    if sha256(features) != str(receipt["features"]["sha256"]):
        raise ValueError("Folds-0/1 feature cache hash mismatch")
    if sha256(metadata) != str(receipt["metadata"]["sha256"]):
        raise ValueError("Folds-0/1 metadata cache hash mismatch")
    return features, metadata, receipt


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
    scores = shrunken_scores(matrix, metadata, protocol)
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
        "shrinkage": float(reference["shrinkage"])
        == float(protocol["candidate"]["shrinkage"]),
        "effective_anchor_strength": float(
            reference["effective_anchor_strength"]
        )
        == float(protocol["candidate"]["anchor_strength"])
        * float(protocol["candidate"]["shrinkage"]),
        "effective_booster_strength": float(
            reference["effective_booster_strength"]
        )
        == float(protocol["candidate"]["booster_strength"])
        * float(protocol["candidate"]["shrinkage"]),
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
        raise ValueError(
            f"Fixed shrinkage failed development reproduction: {checks}"
        )
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
        raise ValueError("Frozen folds-0/1 shrinkage evaluator hash mismatch")
    for dependency in protocol["code_dependencies"]:
        path = (ROOT / dependency["path"]).resolve()
        if sha256(path) != dependency["sha256"]:
            raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if sha256(path) != contract["sha256"]:
            raise ValueError(
                f"Frozen folds-0/1 evaluator input mismatch: {name}"
            )
        paths[name] = path
    if args.smoke:
        print(json.dumps(reproduce_development(protocol, paths), indent=2))
        return 0

    output = (ROOT / protocol["output"]).resolve()
    if output.exists():
        raise FileExistsError(
            "Refusing to repeat the one-shot folds-0/1 confirmation"
        )
    features_path, metadata_path, receipt = verify_receipt(protocol, paths)
    matrix, metadata, columns = load_cache(features_path, metadata_path)
    labels = metadata["labels"].astype(np.uint8)
    sensors = metadata["sensors"].astype(np.uint8)
    groups = metadata["groups"].astype(str)
    folds = metadata["folds"].astype(np.uint8)
    base_scores = metadata["exact_base_scores"].astype(np.float64)
    if list(map(int, np.unique(folds))) != [0, 1]:
        raise ValueError("One-shot cache contains a non-folds-0/1 row")
    if len(np.unique(metadata["sample_ids"].astype(str))) != labels.size:
        raise ValueError("Folds-0/1 sample identities are not unique")
    authorized = set(map(int, columns))
    for name in ("anchor", "booster"):
        feature_index = int(
            protocol["candidate"]["components"][name]["feature_index"]
        )
        if feature_index not in authorized:
            raise ValueError(f"Fixed {name} feature is not authorized")

    scores = shrunken_scores(matrix, metadata, protocol)
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
    tolerance = float(protocol["gates"]["fpr_absolute_tolerance"])
    checks.update(
        {
            "whole_fpr_no_worse": bool(
                whole["matched_fpr_metrics"]["false_positive_rate"]
                <= whole_baseline["matched_fpr_metrics"]["false_positive_rate"]
                + tolerance
            ),
            "low_prevalence_fpr_no_worse": bool(
                low["matched_fpr_metrics"]["false_positive_rate"]
                <= low_baseline["matched_fpr_metrics"]["false_positive_rate"]
                + tolerance
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
        "scope": "one-shot fixed-shrinkage folds-0/1 confirmation",
        "status": (
            "passed_folds01_confirmation"
            if passed
            else "rejected_on_folds01_confirmation"
        ),
        "decision": (
            "Authorize a separately frozen five-fold/fresh safety phase."
            if passed
            else (
                "Retire this shrinkage before fresh or exact-paper access and "
                "resume research without reusing folds 0/1/2."
            )
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": commit,
            "protocol_sha256": sha256(protocol_path),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "extractor_protocol_sha256": sha256(
                paths["extractor_protocol"]
            ),
            "extraction_receipt_sha256": sha256(
                (ROOT / protocol["folds01_cache"]["receipt"]).resolve()
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
        "folds01_accessed": True,
        "fold2_reaccessed": False,
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
