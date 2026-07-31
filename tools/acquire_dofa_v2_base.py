#!/usr/bin/env python3
"""Acquire and verify the pinned official DOFA-v2 base checkpoint and source."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
from dofa_v2_backbone import vit_base_patch14  # noqa: E402

CODE_REPOSITORY = "https://github.com/zhu-xlab/DOFA.git"
CODE_REVISION = "0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9"
ARCHITECTURE_REVISION = "c850a1623413d4c1fc1134202641e3059fe4ab50"
MODEL_REPOSITORY = "earthflow/DOFA"
MODEL_REVISION = "7a5219e48d2f8848511b0fabea7920a8836bc480"
CHECKPOINT_NAME = "dofav2_vit_base_e150.pth"
CHECKPOINT_BYTES = 421_811_730
CHECKPOINT_SHA256 = "e1be9d50fb3e4e3640e337d098b92d67797eaf2a579de3b7a1e363095885314d"
DEFAULT_SOURCE_DIR = Path(".research/foundation_models/dofa")
DEFAULT_CHECKPOINT = DEFAULT_SOURCE_DIR / "checkpoints" / CHECKPOINT_NAME
DEFAULT_RECEIPT = Path("reports/acquisition/dofa_v2_base.json")


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def ensure_source(path: Path) -> None:
    if not (path / ".git").is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", CODE_REPOSITORY, str(path)],
            check=True,
        )
    origin = git("remote", "get-url", "origin", cwd=path)
    if origin.rstrip("/") != CODE_REPOSITORY.removesuffix(".git") and origin != CODE_REPOSITORY:
        raise ValueError(f"Unexpected DOFA source origin: {origin}")
    try:
        git("cat-file", "-e", f"{CODE_REVISION}^{{commit}}", cwd=path)
        git("cat-file", "-e", f"{ARCHITECTURE_REVISION}^{{commit}}", cwd=path)
    except subprocess.CalledProcessError:
        subprocess.run(["git", "fetch", "origin", CODE_REVISION], cwd=path, check=True)
        subprocess.run(
            ["git", "fetch", "origin", ARCHITECTURE_REVISION], cwd=path, check=True
        )
    subprocess.run(
        ["git", "checkout", "--detach", CODE_REVISION],
        cwd=path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if git("rev-parse", "HEAD", cwd=path) != CODE_REVISION:
        raise ValueError("DOFA source revision did not pin exactly")
    # Windows Git may report every checked-out file as modified solely because
    # of CRLF conversion. All provenance below is read from immutable git blobs
    # at the pinned commits, never from the checkout's working-tree files.


def ensure_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != CHECKPOINT_BYTES or sha256(path) != CHECKPOINT_SHA256:
        downloaded = Path(
            hf_hub_download(
                repo_id=MODEL_REPOSITORY,
                filename=CHECKPOINT_NAME,
                revision=MODEL_REVISION,
                local_dir=path.parent,
            )
        )
        if downloaded.resolve() != path.resolve():
            os.replace(downloaded, path)
    if path.stat().st_size != CHECKPOINT_BYTES or sha256(path) != CHECKPOINT_SHA256:
        raise ValueError("DOFA-v2 checkpoint identity mismatch")


def strict_model_audit(checkpoint: Path) -> dict[str, object]:
    model = vit_base_patch14()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    return {
        "state_tensors": len(state),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "strict_load": not incompatible.missing_keys and not incompatible.unexpected_keys,
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR.as_posix())
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT.as_posix())
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT.as_posix())
    args = parser.parse_args()

    source_dir = (ROOT / args.source_dir).resolve()
    checkpoint = (ROOT / args.checkpoint).resolve()
    receipt = (ROOT / args.receipt).resolve()
    ensure_source(source_dir)
    ensure_checkpoint(checkpoint)
    audit = strict_model_audit(checkpoint)
    if not audit["strict_load"]:
        raise RuntimeError("DOFA-v2 checkpoint failed strict architecture loading")
    architecture_source = git(
        "show", f"{ARCHITECTURE_REVISION}:dofa_v2.py", cwd=source_dir
    ).encode("utf-8")
    dynamic_source = git(
        "show", f"{CODE_REVISION}:wave_dynamic_layer.py", cwd=source_dir
    ).encode("utf-8")

    import hashlib

    report: dict[str, object] = {
        "schema_version": 1,
        "scope": "external DOFA-v2 base foundation-model acquisition",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "provider": "DOFA authors / EarthFlow",
            "code_repository": CODE_REPOSITORY,
            "code_revision": CODE_REVISION,
            "architecture_revision": ARCHITECTURE_REVISION,
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "code_license": "MIT",
            "checkpoint_model_card_license": "CC-BY-4.0",
            "paper": "https://arxiv.org/abs/2403.15356",
        },
        "files": [
            {
                "path": checkpoint.relative_to(ROOT).as_posix(),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
                "tracked": False,
            }
        ],
        "source_audit": {
            "official_dofa_v2_py_sha256": hashlib.sha256(architecture_source).hexdigest(),
            "official_wave_dynamic_layer_sha256": hashlib.sha256(dynamic_source).hexdigest(),
            "local_backbone_path": "EarthRemoteSensingRapidResponse/dofa_v2_backbone.py",
            "local_backbone_sha256": sha256(MODEL_ROOT / "dofa_v2_backbone.py"),
            **audit,
        },
        "transport": "git clone at pinned revision plus huggingface_hub download at pinned model revision",
        "bulk_policy": "source clone and checkpoint remain beneath ignored .research",
    }
    write_json(receipt, report)
    print(json.dumps({"ok": True, "receipt": args.receipt, **audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
