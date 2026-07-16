#!/usr/bin/env python3
"""Build deterministic site-relative OOF scores as a feature cache, not a promoted model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_mars_site_relative_development_scores as base  # noqa: E402
from acquire_mars_metadata import sha256  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_site_relative_development_scores_feature_cache_protocol.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    args = parser.parse_args()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if sha256(Path(__file__).resolve()) != protocol["wrapper"]["sha256"]:
        raise ValueError("Feature-cache wrapper hash mismatch")
    if sha256(Path(base.__file__).resolve()) != protocol["builder"]["sha256"]:
        raise ValueError("Underlying score-builder hash mismatch")

    original_assert = base.assert_delta

    def cache_assert(
        actual: dict[str, Any], expected: dict[str, Any], tolerance: float, name: str
    ) -> None:
        if name != "inner":
            original_assert(actual, expected, tolerance, name)
            return
        for metric, limit in (
            ("average_precision", protocol["drift_checks"]["inner_ap_tolerance"]),
            ("recall_at_fpr_0_0713", protocol["drift_checks"]["inner_recall_tolerance"]),
        ):
            if abs(float(actual[metric]) - float(expected[metric])) > float(limit):
                raise RuntimeError(f"Deterministic inner {metric} drift exceeds feature-cache tolerance")

    base.assert_delta = cache_assert
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    old_argv = sys.argv
    try:
        sys.argv = [str(Path(base.__file__).resolve()), "--protocol", str(protocol_path)]
        result = base.main()
    finally:
        sys.argv = old_argv
        base.assert_delta = original_assert
    if result != 0:
        return result

    report_path = (ROOT / protocol["outputs"]["report"]).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["all_frozen_metrics_reproduced"] = False
    report["feature_cache_checks_pass"] = True
    report["interpretation"] = (
        "New deterministic OOF feature cache. Predictive metrics are diagnostics only; "
        "the cache is not a promoted model and does not claim exact old stochastic OOF reproduction."
    )
    report["provenance"]["feature_cache_wrapper_sha256"] = sha256(Path(__file__).resolve())
    report["provenance"]["cuda_deterministic_algorithms"] = True
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "feature_cache": report["output"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
