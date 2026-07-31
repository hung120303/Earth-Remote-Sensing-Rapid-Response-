#!/usr/bin/env python3
"""Acquire and verify the pinned DINOv3 ViT-S/16 checkpoint used by timm."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import timm
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402

MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
MODEL_REPOSITORY = "timm/vit_small_patch16_dinov3.lvd1689m"
MODEL_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
CHECKPOINT_NAME = "model.safetensors"
CHECKPOINT_BYTES = 86_362_376
CHECKPOINT_SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
DEFAULT_MODEL_DIR = Path(".research/foundation_models/dinov3/checkpoints")
DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / CHECKPOINT_NAME
DEFAULT_RECEIPT = Path("reports/acquisition/dinov3_vits16_lvd1689m.json")
SUPPORT_FILES = ("LICENSE.md", "README.md", "config.json")


def ensure_files(model_dir: Path) -> dict[str, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in (CHECKPOINT_NAME, *SUPPORT_FILES):
        downloaded = Path(
            hf_hub_download(
                repo_id=MODEL_REPOSITORY,
                filename=name,
                revision=MODEL_REVISION,
                local_dir=model_dir,
            )
        )
        paths[name] = downloaded.resolve()
    checkpoint = paths[CHECKPOINT_NAME]
    if (
        checkpoint.stat().st_size != CHECKPOINT_BYTES
        or sha256(checkpoint) != CHECKPOINT_SHA256
    ):
        raise ValueError("DINOv3 ViT-S/16 checkpoint identity mismatch")
    return paths


def strict_model_audit(checkpoint: Path) -> dict[str, object]:
    model = timm.create_model(MODEL_NAME, pretrained=False).eval()
    state = load_file(checkpoint.as_posix(), device="cpu")
    incompatible = model.load_state_dict(state, strict=True)
    with torch.inference_mode():
        outputs = model.forward_intermediates(
            torch.zeros((1, 3, 256, 256), dtype=torch.float32),
            indices=[2, 5, 8, 11],
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )
    shapes = [list(value.shape) for value in outputs]
    finite = all(bool(torch.isfinite(value).all()) for value in outputs)
    return {
        "state_tensors": len(state),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "strict_load": not incompatible.missing_keys and not incompatible.unexpected_keys,
        "intermediate_blocks": [2, 5, 8, 11],
        "intermediate_shapes": shapes,
        "finite_zero_input_forward": finite,
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR.as_posix())
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT.as_posix())
    args = parser.parse_args()
    model_dir = (ROOT / args.model_dir).resolve()
    receipt = (ROOT / args.receipt).resolve()
    paths = ensure_files(model_dir)
    audit = strict_model_audit(paths[CHECKPOINT_NAME])
    if not audit["strict_load"] or not audit["finite_zero_input_forward"]:
        raise RuntimeError("DINOv3 checkpoint failed the frozen architecture audit")
    report: dict[str, object] = {
        "schema_version": 1,
        "scope": "external DINOv3 ViT-S/16 foundation-model acquisition",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "provider": "Meta AI checkpoint redistributed by timm",
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "model_name": MODEL_NAME,
            "model_license": "DINOv3 License",
            "timm_code_license": "Apache-2.0",
            "paper": "https://arxiv.org/abs/2508.10104",
            "official_repository": "https://github.com/facebookresearch/dinov3",
        },
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "tracked": False,
            }
            for path in paths.values()
        ],
        "model_audit": audit,
        "runtime": {"torch": torch.__version__, "timm": timm.__version__},
        "transport": "huggingface_hub download at a pinned immutable model revision",
        "bulk_policy": "checkpoint and support files remain beneath ignored .research",
    }
    write_json(receipt, report)
    print(json.dumps({"ok": True, "receipt": args.receipt, **audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
