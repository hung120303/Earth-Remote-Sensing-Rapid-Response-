#!/usr/bin/env python3
"""Build the self-contained ERSRR publication research dossier."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = Path("tools/templates/ersrr_research_report.html")
DEFAULT_OUTPUT = Path("reports/ERSRR_RESEARCH_REPORT.html")


def read_json(root: Path, value: str | Path) -> dict[str, Any]:
    return json.loads((root / value).read_text(encoding="utf-8"))


def model_row(
    name: str,
    display: str,
    primary: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    model = primary["models"][name]
    return {
        "name": name,
        "display": display,
        "metrics": model["metrics"],
        "rule": model["rule"],
        "uncertainty": bootstrap["models"][name],
    }


def build_data(root: Path) -> dict[str, Any]:
    primary = read_json(root, "reports/experiments/methanes2cm_v5_1_location_test.json")
    ensemble = read_json(
        root, "reports/experiments/methanes2cm_v5_1_ensemble_validation.json"
    )
    posthoc = read_json(
        root, "reports/experiments/methanes2cm_v5_1_location_test_posthoc.json"
    )
    protocol = read_json(root, "reports/experiments/methanes2cm_v5_protocol.json")
    campaign = read_json(
        root, "reports/experiments/methanes2cm_v5_1_campaign_protocol.json"
    )
    signal = read_json(root, "reports/experiments/methanes2cm_v5_signal_audit.json")
    train_acquisition = read_json(
        root, "reports/experiments/methanes2cm_v5_train_acquisition.json"
    )
    test_acquisition = read_json(
        root, "reports/experiments/methanes2cm_v5_location_test_acquisition.json"
    )
    strict = read_json(root, "reports/experiments/mars_v4_3_strict_comparison.json")
    v3 = read_json(root, "reports/experiments/mars_v3_strict_campaign.json")
    bootstrap = primary["group_bootstrap"]
    models = [
        model_row("ersrr_v5_1", "ERSRR v5.1", primary, bootstrap),
        model_row("ersrr_v4_3", "ERSRR v4.3 zero-shot", primary, bootstrap),
        model_row(
            "released_mars_s2l", "Released MARS-S2L zero-shot", primary, bootstrap
        ),
    ]
    best_physics_name, best_physics = max(
        signal["physics_baselines"].items(),
        key=lambda item: item[1]["scene_average_precision"],
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "status": {
            "decision": primary["decision"],
            "test_frozen": True,
            "retuning_permitted": False,
            "all_mars_point_checks": primary["comparison"][
                "v5_1_vs_released_mars_s2l"
            ]["all_point_checks_pass"],
            "all_mars_bootstrap_checks": primary["comparison"][
                "v5_1_vs_released_mars_s2l"
            ]["all_bootstrap_checks_pass"],
        },
        "cohort": primary["cohort"],
        "models": models,
        "paired_delta_vs_mars": bootstrap["deltas"][
            "ersrr_v5_1_minus_released_mars_s2l"
        ],
        "development": {
            "group_held": ensemble["group_held_calibration_audit"],
            "final_rule": ensemble["final_all_development_rule"],
            "seed_mean": ensemble["seed_mean"],
            "seeds": ensemble["seeds"],
            "v5_mask_derived_reference": ensemble[
                "controlled_v5_seed1101_reference"
            ],
            "bootstrap": ensemble["group_bootstrap"],
        },
        "calibration_transfer": posthoc["frozen_operating_points"],
        "architecture": {
            **campaign["architecture"],
            "checkpoint_bytes_per_seed": ensemble["seeds"][0]["checkpoint"]["bytes"],
            "learned_parameters_per_seed": 9_358_256,
            "input_contract": protocol["input_contract"],
            "model_bands": protocol["source"]["model_bands"],
            "patch_shape": protocol["source"]["patch_shape"],
            "pixel_size_m": protocol["source"]["pixel_size_m"],
            "best_physics_baseline": {
                "name": best_physics_name,
                **best_physics,
            },
            "mask_audit": signal["mask_audit"],
        },
        "mars_context": {
            "same_strict_cohort": {
                "ersrr_v4_3": strict["strict_spatial_test"],
                "released_mars_s2l": strict["same_cohort_comparison"][
                    "released_mars_s2l"
                ],
                "comparison": strict["same_cohort_comparison"],
            },
            "official_different_cohort": strict[
                "official_mars_s2l_paper_targets_not_same_cohort"
            ],
            "v3_retired": {
                "ersrr_mean": v3["same_cohort_comparison"]["ersrr_seed_mean"],
                "released_mars_s2l": v3["same_cohort_comparison"][
                    "released_mars_s2l"
                ],
                "decision": v3["decision"],
            },
        },
        "data": {
            "dataset": protocol["source"]["dataset"],
            "revision": protocol["source"]["revision"],
            "license": protocol["source"]["license"],
            "train": protocol["cohort"]["train"],
            "development": protocol["development_protocol"],
            "test": protocol["cohort"]["test"],
            "compressed_archive_bytes": sum(
                int(item["bytes"]) for item in train_acquisition["archives"]
            ),
            "packed_train_bytes": train_acquisition["extraction"]["packed_bytes"],
            "packed_test_bytes": test_acquisition["extraction"]["packed_bytes"],
            "test_assets": test_acquisition["extraction"]["files_decoded"],
            "coordinate_overlap": protocol["cohort"]["coordinate_overlap"],
        },
        "paper": {
            "supported": [
                "V5.1 materially improves scene ranking and dense localization over both frozen zero-shot comparators on the same MethaneS2CM location test.",
                "The paired 25 km bootstrap supports positive AP, AUROC, recall, Dice, and IoU deltas versus released MARS-S2L.",
                "A small context head improved scene discrimination over mask-only v5 while preserving the dense decoder.",
                "Spatial grouping, sealed-test discipline, immutable hashes, and bulk-data exclusion are implemented end to end.",
            ],
            "not_supported": [
                "Across-the-board MARS-S2L superiority is not established: v5.1 has higher frozen-rule FPR on MethaneS2CM.",
                "MethaneS2CM precision is not operational PPV because the crop benchmark is approximately balanced.",
                "The L2A in-domain versus L1C zero-shot comparison is not an architecture-only causal experiment.",
                "No concentration, flux, operational deployment, or test-driven recalibration claim is supported.",
            ],
            "next_study": [
                "Fit spatially group-held calibration or conformal risk control on new calibration groups, never on this test.",
                "Train product-aware L1C/L2A domain harmonization with missing-frame handling across MARS-S2L and MethaneS2CM.",
                "Improve dense boundaries with mask-quality weighting and plume-scale sampling while retaining hard no-plume negatives.",
                "Acquire and preregister a new geographically isolated, prevalence-aware confirmation cohort before v5.2 is frozen.",
            ],
        },
        "provenance": {
            "primary_report": "reports/experiments/methanes2cm_v5_1_location_test.json",
            "primary_prediction_cache_sha256": primary["prediction_cache"]["sha256"],
            "test_pack_sha256": primary["seal"]["packed_test"]["sha256"],
            "ensemble_report": "reports/experiments/methanes2cm_v5_1_ensemble_validation.json",
            "campaign_protocol": "reports/experiments/methanes2cm_v5_1_campaign_protocol.json",
            "posthoc_report": "reports/experiments/methanes2cm_v5_1_location_test_posthoc.json",
            "bootstrap_seed": bootstrap["seed"],
            "bootstrap_replicates": bootstrap["replicates"],
            "research_ledger": "docs/RESEARCH_LEDGER.md",
            "paper_outline": "docs/PAPER_OUTLINE.md",
        },
    }


def render(template: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).replace(
        "</", "<\/"
    )
    if template.count("__ERSRR_REPORT_DATA__") != 1:
        raise ValueError("Report template must contain exactly one data placeholder")
    return template.replace("__ERSRR_REPORT_DATA__", payload)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE.as_posix())
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    args = parser.parse_args()
    root = ROOT
    template = (root / args.template).read_text(encoding="utf-8")
    output = (root / args.output).resolve()
    if root not in output.parents:
        raise ValueError("Output must resolve beneath the repository root")
    write(output, render(template, build_data(root)))
    print(output.relative_to(root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
