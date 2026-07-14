#!/usr/bin/env python3
"""Prove that ERSRR's paper successor initializes to the released MARS-S2L logits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from mars_paper_model import (  # noqa: E402
    MarsPaperResidualModel,
    RELEASED_CHECKPOINT_SHA256,
)

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from evaluate_released_marss2l import (  # noqa: E402
    RELEASE_SPECS,
    load_released_model,
)

DEFAULT_CHECKPOINT = RELEASE_SPECS["mars-s2l"]["directory"] / "best_epoch"
DEFAULT_JSON = Path(
    "reports/experiments/mars_paper_model_initialization_audit.json"
)
DEFAULT_MARKDOWN = Path(
    "reports/experiments/MARS_PAPER_MODEL_INITIALIZATION_AUDIT.md"
)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--output-json", default=DEFAULT_JSON.as_posix())
    parser.add_argument("--output-markdown", default=DEFAULT_MARKDOWN.as_posix())
    args = parser.parse_args()
    root = repo_root()
    checkpoint = (root / args.checkpoint).resolve()
    observed_checkpoint_hash = sha256(checkpoint)
    if observed_checkpoint_hash != RELEASED_CHECKPOINT_SHA256:
        raise ValueError("Released checkpoint identity differs from the frozen model contract")

    device = torch.device("cpu")
    independent = load_released_model(checkpoint, device, 16)
    successor = MarsPaperResidualModel().to(device).eval()
    successor.load_released_checkpoint(checkpoint)

    generator = torch.Generator().manual_seed(20260713)
    inputs = torch.rand(2, 16, 64, 64, generator=generator)
    inputs[:, 0] = 1.0 + 0.1 * (inputs[:, 0] - 0.5)
    observable = torch.ones(2, 1, 64, 64)
    observable[:, :, :3, :] = 0
    sensors = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        independent_logits = independent(inputs)[:, None]
        output = successor(inputs, observable, sensors)
    candidate_logits = output["segmentation_logits"]
    baseline_logits = output["baseline_logits"]
    correction = output["correction_logits"]
    maximum_independent_delta = float(
        torch.max(torch.abs(candidate_logits - independent_logits)).item()
    )
    maximum_internal_delta = float(
        torch.max(torch.abs(candidate_logits - baseline_logits)).item()
    )
    maximum_correction = float(torch.max(torch.abs(correction)).item())
    exact = (
        maximum_independent_delta == 0.0
        and maximum_internal_delta == 0.0
        and maximum_correction == 0.0
    )
    if not exact:
        raise ValueError("Successor initialization is not exactly baseline preserving")

    report: dict[str, object] = {
        "schema_version": 1,
        "scope": "independent released-checkpoint equivalence before successor training",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": {
            "exact": exact,
            "batch_sensors": ["Sentinel-2", "Landsat"],
            "input_shape": list(inputs.shape),
            "maximum_absolute_delta_vs_independent_reimplementation": maximum_independent_delta,
            "maximum_absolute_delta_vs_internal_backbone": maximum_internal_delta,
            "maximum_absolute_zero_initialized_correction": maximum_correction,
        },
        "model": successor.artifact_metadata(),
        "source": {
            "checkpoint": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": observed_checkpoint_hash,
            "independent_implementation": "tools/evaluate_released_marss2l.py::ReleasedUNet",
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": "tools/audit_mars_paper_model_initialization.py",
            "script_sha256": sha256(Path(__file__)),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    output_json = (root / args.output_json).resolve()
    output_markdown = (root / args.output_markdown).resolve()
    write_json(output_json, report)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(
        "\n".join(
            [
                "# MARS paper successor initialization audit",
                "",
                "The untrained successor exactly reproduces the independently implemented released MARS-S2L logits for both Sentinel-2 and Landsat sensor identities.",
                "",
                f"- Checkpoint SHA-256: `{observed_checkpoint_hash}`",
                f"- Maximum absolute logit delta: `{maximum_independent_delta:.1f}`",
                f"- Maximum absolute correction: `{maximum_correction:.1f}`",
                f"- Total parameters: {successor.artifact_metadata()['parameter_count']:,}",
                f"- Initially trainable correction parameters: {successor.artifact_metadata()['trainable_parameter_count_correction_only']:,}",
                "",
                "Any subsequent gain or regression is therefore attributable to the successor training and frozen decision rule, not a weaker random initialization.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["result"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
