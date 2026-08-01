#!/usr/bin/env python3
"""Run the patch-local NDMI pilot through the frozen cross-fit harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_mars_ndmi_bitemporal_fusion_pilot as harness  # noqa: E402
from acquire_mars_metadata import sha256  # noqa: E402
from mars_ndmi_patch_mil_fusion import NdmiPatchMilFusionAdapter  # noqa: E402
from train_mars_paper_residual import verify_acquisition_receipt  # noqa: E402


DEFAULT_PROTOCOL = Path("configs/mars_ndmi_patch_mil_pilot_protocol.json")


def verify_protocol(protocol: dict[str, Any], *, smoke: bool) -> dict[str, Path]:
    """Bind the inherited data/evaluation harness to this entry point."""

    frozen = str(protocol["status"]).startswith("frozen")
    if not frozen and not smoke:
        raise ValueError("Outcome evaluation requires a frozen patch-MIL protocol")
    if frozen and sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen NDMI patch-MIL trainer hash mismatch")
    if frozen:
        for dependency in protocol["code_dependencies"]:
            path = (ROOT / dependency["path"]).resolve()
            if sha256(path) != dependency["sha256"]:
                raise ValueError(f"Frozen dependency mismatch: {dependency['path']}")
    paths: dict[str, Path] = {}
    for name, contract in protocol["inputs"].items():
        path = (ROOT / contract["path"]).resolve()
        if frozen and path.is_file() and sha256(path) != contract["sha256"]:
            raise ValueError(f"Frozen input mismatch: {name}")
        paths[name] = path
    if not paths["metadata_root"].is_dir():
        raise ValueError("MARS metadata root is unavailable")
    verify_acquisition_receipt(paths["acquisition_receipt"], sha256(paths["manifest"]))
    return paths


def main() -> int:
    # The previously committed harness owns data loading, losses, cross-fit,
    # bootstraps, report serialization, and artifact rules. This entry point
    # injects only the predeclared patch-local model and its own hash verifier.
    harness.DEFAULT_PROTOCOL = DEFAULT_PROTOCOL
    harness.NdmiBitemporalFusionAdapter = NdmiPatchMilFusionAdapter
    harness.verify_protocol = verify_protocol
    return harness.main()


if __name__ == "__main__":
    raise SystemExit(main())
