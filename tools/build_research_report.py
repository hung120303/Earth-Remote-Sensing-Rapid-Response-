#!/usr/bin/env python3
"""Build the self-contained ERSRR publication-facing HTML research report."""

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
SEEDS = (101, 202, 303, 404, 505)


def read_json(root: Path, value: str | Path) -> dict[str, Any]:
    return json.loads((root / value).read_text(encoding="utf-8"))


def validation_path(seed: int) -> Path:
    if seed == 303:
        return Path("reports/experiments/mars_v3_proposal_validation.json")
    return Path(f"reports/experiments/mars_v3_seed{seed}_proposal_validation.json")


def build_data(root: Path) -> dict[str, Any]:
    campaign = read_json(root, "reports/experiments/mars_v3_strict_campaign.json")
    diagnostic = read_json(
        root, "reports/experiments/mars_v3_strict_posthoc_diagnostic.json"
    )
    strict_download = read_json(
        root, "reports/acquisition/mars_s2l_v3_strict_download.json"
    )
    emit_seal = read_json(
        root, "reports/acquisition/emit_v002_external_cohort_seal.json"
    )
    wind = read_json(root, "reports/acquisition/emit_v002_era5_wind_requests.json")

    validation: list[dict[str, Any]] = []
    validation_by_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        report = read_json(root, validation_path(seed))
        scene = report["validation"]["scene"]
        item = {
            "seed": seed,
            "recall": scene["recall"],
            "fpr": scene["false_positive_rate"],
            "ap": scene["average_precision"],
            "auroc": scene["auroc"],
            "proposal_weight": report["operating_rule"]["neural_presence_weight"],
        }
        validation.append(item)
        validation_by_seed[seed] = item

    strict = campaign["same_cohort_comparison"]["ersrr_per_seed"]
    strict_by_seed = {int(item["seed"]): item for item in strict}
    seeds = [
        {
            **validation_by_seed[seed],
            "strict_recall": strict_by_seed[seed]["recall"],
            "strict_fpr": strict_by_seed[seed]["false_positive_rate"],
            "strict_ap": strict_by_seed[seed]["average_precision"],
            "strict_auroc": strict_by_seed[seed]["auroc"],
        }
        for seed in SEEDS
    ]

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "decision": campaign["decision"],
        "gate": campaign["promotion_gate"],
        "cohort": campaign["cohort"],
        "seeds": seeds,
        "baseline": campaign["same_cohort_comparison"]["released_mars_s2l"],
        "ersrr_mean": campaign["same_cohort_comparison"]["ersrr_seed_mean"],
        "ersrr_sd": campaign["same_cohort_comparison"][
            "ersrr_seed_standard_deviation"
        ],
        "delta": campaign["same_cohort_comparison"]["delta"],
        "bootstrap": campaign["paired_seed_group_bootstrap"],
        "segmentation": campaign["segmentation"],
        "official_mars": campaign[
            "official_mars_s2l_paper_targets_not_same_cohort"
        ],
        "diagnostic": {
            "agreement": diagnostic["agreement"],
            "positive_strata": diagnostic["positive_strata"],
            "negative_strata": diagnostic["negative_strata"],
            "atlas": diagnostic["error_atlas"],
            "decision": diagnostic["decision"],
        },
        "architecture": {
            "name": "ersrr_mars_full_unet_proposal_v3",
            "parameters": 14_268_915,
            "channels": 16,
            "heads": ["segmentation", "proposal presence", "quality / abstention"],
            "inputs": [
                "release-compatible MBMP",
                "six target Sentinel-2 L1C bands",
                "six reference Sentinel-2 L1C bands",
                "ERA5-Land u/v wind",
                "CloudSEN12 observability mask",
            ],
        },
        "data": {
            "training_assets": 61_928,
            "training_bytes": 30_366_803_325,
            "strict_assets": strict_download["result"]["selected_asset_count"],
            "strict_bytes": strict_download["result"]["selected_total_bytes"],
            "strict_verified": strict_download["result"]["complete_size_match_count"],
            "emit_sealed": emit_seal["summary"]["final_gate_pass"],
            "emit_preliminary": emit_seal["summary"]["preliminary_gate_pass"],
            "wind_requests": wind["summary"]["requests"],
            "wind_validation": wind["summary"][
                "official_costing_api_validation"
            ],
            "wind_authentication": wind["summary"]["authentication_state"],
        },
        "provenance": {
            "manifest_sha256": campaign["cohort"]["strict_manifest_sha256"],
            "bootstrap_seed": campaign["paired_seed_group_bootstrap"][
                "random_seed"
            ],
            "bootstrap_replicates": campaign["paired_seed_group_bootstrap"][
                "replicates"
            ],
            "campaign_report": "reports/experiments/mars_v3_strict_campaign.json",
            "diagnostic_report": "reports/experiments/mars_v3_strict_posthoc_diagnostic.json",
            "research_ledger": "docs/RESEARCH_LEDGER.md",
            "paper_outline": "docs/PAPER_OUTLINE.md",
        },
    }


def render(template: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).replace(
        "</", "<\\/"
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
    data = build_data(root)
    write(output, render(template, data))
    print(output.relative_to(root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
