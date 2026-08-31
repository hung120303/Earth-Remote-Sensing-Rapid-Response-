#!/usr/bin/env python3
"""Build the self-contained ERSRR MARS-S2L successor research dossier."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESCRIPTIVE = Path("reports/experiments/mars_sensor_ordinal_folds34_descriptive.json")
DEFAULT_TEMPLATE = Path("tools/templates/ersrr_mars_successor_report.html")
DEFAULT_OUTPUT = Path("reports/ERSRR_MARS_SUCCESSOR_REPORT.html")


def read_json(root: Path, value: str | Path) -> dict[str, Any]:
    return json.loads((root / value).read_text(encoding="utf-8"))


def _prior_view(view: Mapping[str, Any]) -> dict[str, Any]:
    metrics = view["metrics"]
    return {
        "passed": bool(view["passed"]),
        "rows": int(metrics["rows"]),
        "positive": int(metrics["positive"]),
        "candidate": metrics["candidate"],
        "baseline": metrics["baseline"],
        "delta": metrics["delta"],
        "bootstrap": view["bootstrap"],
        "checks": view["checks"],
    }


def build_data(
    root: Path = ROOT,
    *,
    descriptive: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    descriptive = dict(descriptive) if descriptive is not None else read_json(root, DEFAULT_DESCRIPTIVE)
    paper = read_json(root, "reports/acquisition/mars_s2l_paper_v3_benchmark.json")
    prior = read_json(root, "reports/experiments/mars_spatial_prithvi_ensemble_paper_posttest.json")
    execution = read_json(root, "configs/mars_sensor_ordinal_protocol.json")
    reporting = read_json(root, "configs/mars_sensor_ordinal_reporting_protocol.json")
    ghgsat = read_json(root, "reports/acquisition/ghgsat_landfill_windowed_observability.json")
    ornl = read_json(root, "reports/acquisition/jpl_operational_ghg_ornl_header_bridge.json")
    development_passed = descriptive["decision"] == "PASS_DEVELOPMENT"
    official_established = False
    headline = (
        "A new architecture cleared development. The paper benchmark is next."
        if development_passed
        else "The new architecture was tested honestly—and stopped at development."
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "status": {
            "development_passed": development_passed,
            "official_superiority_established": official_established,
            "headline": headline,
            "decision": descriptive["decision"],
            "claim_language": descriptive["claim_language"],
            "forbidden_claim": descriptive["forbidden_claim"],
            "next_step": (
                "Freeze replication and protected-evaluation protocol before opening any new outcome."
                if development_passed
                else "Reject this branch; choose the next architecture using only fresh development evidence."
            ),
        },
        "ordinal": descriptive,
        "architecture": {
            "name": "Sensor-aware ordinal U-Net",
            "input_channels": execution["architecture"]["input_channels"],
            "sensor_stems": execution["architecture"]["sensor_stems"],
            "unet_widths": execution["architecture"]["unet_widths"],
            "ordinal_logits": execution["architecture"]["ordinal_cumulative_logits"],
            "scene_descriptor": execution["architecture"]["scene_descriptor"],
            "scene_gradient_isolated": not execution["architecture"]["scene_gradient_into_pixel_network"],
            "pretrained": execution["architecture"]["pretrained_or_auxiliary_candidate_components"],
            "training": execution["training"],
            "evaluation": execution["evaluation"],
            "scope": execution["scope"],
        },
        "paper": {
            "url": paper["paper"]["url"],
            "revision": paper["paper"]["revision"],
            "revision_date": paper["paper"]["revision_date"],
            "published": paper["reconstruction"]["published"],
            "reconstructed": {
                "full": paper["reconstruction"]["full"],
                "test_only_sites": paper["reconstruction"]["test_only_sites"],
            },
            "superiority_gate": paper["superiority_gate"],
            "assignment_sha256": paper["artifacts"]["assignment_sha256"],
            "upstream_code_revision": paper["source"]["upstream_code_revision"],
        },
        "prior_official": {
            "name": "Frozen spatial-Prithvi ensemble",
            "decision": prior["decision"],
            "all_exact_paper_gates_pass": prior["all_exact_paper_gates_pass"],
            "full": _prior_view(prior["views"]["full"]),
            "test_only_sites": _prior_view(prior["views"]["test_only_sites"]),
        },
        "research_path": [
            {
                "date": "2026-07",
                "name": "Exact MARS-S2L replay",
                "state": "frozen",
                "result": "43,529 official rows reconstructed; stronger archive values retained as the comparator.",
            },
            {
                "date": "2026-07",
                "name": "Spatial-Prithvi ensemble",
                "state": "inconclusive",
                "result": "Full-view AP and IoU improved, but test-only-site AP and matched-recall confidence intervals crossed zero.",
            },
            {
                "date": "2026-08",
                "name": "ORNL/JPL negative bridge",
                "state": "rejected",
                "result": (
                    f"{ornl['stage_b_covid_permian_metadata']['counts']['eligible_rows']:,} eligible rows collapsed to "
                    f"{ornl['stage_b_covid_permian_metadata']['counts']['eligible_25km_components']} independent 25 km components; 20 were required."
                ),
            },
            {
                "date": "2026-08",
                "name": "GHGSat null transfer",
                "state": "rejected",
                "result": (
                    f"{ghgsat['frozen_pairs_attempted']} exact target/reference pairs were attempted; "
                    f"{ghgsat['counts']['source_sensor_pairs']} locally observable pairs survived."
                ),
            },
            {
                "date": "2026-08",
                "name": "Sensor-aware ordinal U-Net",
                "state": "passed" if development_passed else "rejected",
                "result": descriptive["claim_language"],
            },
        ],
        "provenance": {
            "reporting_contract": "configs/mars_sensor_ordinal_reporting_protocol.json",
            "reporting_contract_sha256": descriptive["provenance"]["reporting_protocol_sha256"],
            "execution_protocol": execution["protocol_sha256_self_excluding_field"],
            "science_digest": reporting["frozen_execution"]["science_digest"],
            "compact_result_sha256": descriptive["provenance"]["compact_result_sha256"],
            "candidate_predictions": descriptive["provenance"]["candidate_predictions"],
            "endpoint_states": descriptive["provenance"]["endpoint_states"],
            "paper_benchmark_receipt": reporting["future_inputs"]["paper_v3_benchmark_receipt"],
            "research_ledger": "docs/RESEARCH_LEDGER.md",
            "paper_outline": "docs/MARS_PAPER_SUCCESSOR_OUTLINE.md",
        },
    }


def render(template: str, data: Mapping[str, Any]) -> str:
    placeholder = "__ERSRR_SUCCESSOR_DATA__"
    if template.count(placeholder) != 1:
        raise ValueError("Successor template must contain exactly one data placeholder")
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(placeholder, payload)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptive", default=DEFAULT_DESCRIPTIVE.as_posix())
    parser.add_argument("--template", default=DEFAULT_TEMPLATE.as_posix())
    parser.add_argument("--output", default=DEFAULT_OUTPUT.as_posix())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    descriptive = read_json(ROOT, args.descriptive)
    template = (ROOT / args.template).read_text(encoding="utf-8")
    output = (ROOT / args.output).resolve()
    if ROOT not in output.parents:
        raise ValueError("Output must resolve beneath the repository root")
    write(output, render(template, build_data(ROOT, descriptive=descriptive)))
    print(output.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
