#!/usr/bin/env python3
"""Acquire and verify the pinned official Prithvi-EO-2.0-100M-TL model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402

REPOSITORY = "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL"
REVISION = "2c84e383194986040f883cc43d7869002c425e1b"
FILES = (
    "config.json",
    "inference.py",
    "prithvi_mae.py",
    "README.md",
    "Prithvi_EO_V2_100M_TL.pt",
)
CHECKPOINT_BYTES = 454_660_610
CHECKPOINT_SHA256 = "d45406d5fc51af1d9657d48f2e2c3ff077408a2e1113f9a242889a4fe4469b17"
DEFAULT_DESTINATION = Path(
    "EarthRemoteSensingRapidResponse/artifacts/foundation/prithvi_eo_2_100m_tl"
)
DEFAULT_RECEIPT = Path("reports/acquisition/prithvi_eo_2_100m_tl.json")


def download(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        target = destination / filename
        downloaded = Path(
            hf_hub_download(
                repo_id=REPOSITORY,
                filename=filename,
                revision=REVISION,
                local_dir=destination,
            )
        )
        if downloaded.resolve() != target.resolve():
            os.replace(downloaded, target)
    checkpoint = destination / FILES[-1]
    if checkpoint.stat().st_size != CHECKPOINT_BYTES or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Prithvi-EO-2.0-100M-TL checkpoint identity mismatch")


def strict_audit(destination: Path) -> dict[str, object]:
    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))[
        "pretrained_cfg"
    ]
    if str(destination) not in sys.path:
        sys.path.insert(0, str(destination))
    from prithvi_mae import PrithviMAE  # type: ignore  # noqa: E402

    model = PrithviMAE(**config)
    state = torch.load(
        destination / "Prithvi_EO_V2_100M_TL.pt",
        map_location="cpu",
        weights_only=True,
    )
    incompatible = model.load_state_dict(state, strict=True)
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "encoder_parameters": sum(
            parameter.numel() for parameter in model.encoder.parameters()
        ),
        "state_tensors": len(state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "strict_load": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "pretrained_config": config,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default=DEFAULT_DESTINATION.as_posix())
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT.as_posix())
    args = parser.parse_args()
    destination = (ROOT / args.destination).resolve()
    receipt = (ROOT / args.receipt).resolve()
    download(destination)
    audit = strict_audit(destination)
    if not audit["strict_load"]:
        raise RuntimeError("Prithvi-EO-2.0-100M-TL failed strict loading")
    files = [
        {
            "path": (destination / filename).relative_to(ROOT).as_posix(),
            "bytes": (destination / filename).stat().st_size,
            "sha256": sha256(destination / filename),
            "tracked": False,
        }
        for filename in FILES
    ]
    report = {
        "schema_version": 1,
        "scope": "external foundation-model acquisition for MARS representation research",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "provider": "IBM-NASA Geospatial",
            "repository": REPOSITORY,
            "repository_url": f"https://huggingface.co/{REPOSITORY}",
            "revision": REVISION,
            "license": "Apache-2.0",
            "paper": "https://arxiv.org/abs/2412.02732",
            "pretraining": "4.2M global HLS V2 time-series samples; six bands; 30 m granularity",
        },
        "selection_rationale": (
            "The 100M temporal/location encoder is the smallest substantially larger official "
            "Prithvi-EO-2.0 checkpoint. It tests whether the tiny encoder, rather than residual "
            "head capacity, limits geographically transferable methane-scene representation."
        ),
        "files": files,
        "strict_model_audit": audit,
        "transport": "huggingface_hub downloads pinned to the immutable model revision",
        "bulk_policy": "all model files remain beneath the ignored artifacts/foundation directory",
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix(receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt)
    print(json.dumps({"ok": True, "receipt": args.receipt, **audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
