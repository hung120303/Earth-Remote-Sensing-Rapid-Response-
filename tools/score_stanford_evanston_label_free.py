#!/usr/bin/env python3
"""Run the frozen Stanford Evanston label-free scorer with protocol-only paths.

This launcher deliberately exposes no pair, crop, or output path overrides. It
validates those cohort-specific paths against the frozen Evanston protocol, then
delegates all model and preprocessing work to the unchanged Casa Grande scorer.
No release summary, label, rate, or detector outcome is opened here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import score_stanford_large_controlled_release_label_free as base  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/stanford_evanston_label_free_scoring_protocol.json")
PAIR_BINDING = "pair_manifest"
CROP_BINDING = "crop_manifest"


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--batch-size", type=positive_int, default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--input-preflight", action="store_true")
    modes.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args(argv)


def _repository_path(value: str, *, name: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"{name} must remain beneath repository root") from exc
    return path


def load_and_validate_protocol(
    protocol_path: Path, *, require_inference_authorized: bool = True
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    try:
        protocol_path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("Protocol must remain beneath repository root") from exc
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dependencies = protocol["deployment_dependencies"]
    bindings = base.validate_frozen_bindings(ROOT, dependencies)
    if not bindings:
        raise ValueError("Evanston protocol has no frozen deployment bindings")

    cohort = protocol["cohort_inputs"]
    if int(cohort["expected_rows"]) != 9:
        raise ValueError("Evanston protocol must bind exactly nine outcome-blind rows")
    for name, hash_key in (
        (PAIR_BINDING, "pair_manifest_sha256"),
        (CROP_BINDING, "crop_manifest_sha256"),
    ):
        binding = dependencies[name]
        if str(binding["sha256"]) != str(cohort[hash_key]):
            raise ValueError(f"Evanston {name} cohort and deployment hashes differ")
        _repository_path(str(binding["path"]), name=name)

    outputs = protocol["score_outputs"]
    for name in ("scores", "score_manifest", "receipt"):
        _repository_path(str(outputs[name]), name=f"score_outputs.{name}")
    if bool(protocol["implementation_gate"].get("labels_allowed", True)):
        raise ValueError("Evanston label-free protocol must forbid labels")
    if require_inference_authorized and not bool(
        protocol["implementation_gate"].get("real_inference_authorized", False)
    ):
        raise ValueError("Evanston real inference is not authorized by the frozen protocol")
    return protocol


def delegated_argv(
    protocol_path: Path,
    protocol: dict[str, Any],
    *,
    batch_size: int,
    mode: str | None,
) -> list[str]:
    dependencies = protocol["deployment_dependencies"]
    outputs = protocol["score_outputs"]
    argv = [
        "--protocol",
        protocol_path.relative_to(ROOT).as_posix(),
        "--pair-manifest",
        str(dependencies[PAIR_BINDING]["path"]),
        "--crop-manifest",
        str(dependencies[CROP_BINDING]["path"]),
        "--scores",
        str(outputs["scores"]),
        "--score-manifest",
        str(outputs["score_manifest"]),
        "--receipt",
        str(outputs["receipt"]),
        "--batch-size",
        str(batch_size),
    ]
    if mode is not None:
        argv.append(f"--{mode}")
    return argv


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = _repository_path(args.protocol, name="protocol")
    mode = (
        "dry-run"
        if args.dry_run
        else "input-preflight"
        if args.input_preflight
        else "synthetic-smoke"
        if args.synthetic_smoke
        else None
    )
    protocol = load_and_validate_protocol(
        protocol_path, require_inference_authorized=mode is None
    )
    return base.main(
        delegated_argv(
            protocol_path,
            protocol,
            batch_size=args.batch_size,
            mode=mode,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
