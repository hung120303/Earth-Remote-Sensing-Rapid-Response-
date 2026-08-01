#!/usr/bin/env python3
"""Run the protected bi-sensor variant of anchored MARS full fine-tuning."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (MODEL_ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_mars_anchored_full_finetune_pilot as parent  # noqa: E402
from acquire_mars_metadata import sha256  # noqa: E402
from mars_protected_bisensor_finetune import ProtectedBisensorFinetune  # noqa: E402
from train_mars_paper_residual import verify_acquisition_receipt  # noqa: E402

DEFAULT_PROTOCOL = Path("configs/mars_protected_bisensor_finetune_pilot_protocol.json")


def verify_protocol(protocol: dict[str, Any], *, smoke: bool) -> dict[str, Path]:
    frozen = str(protocol["status"]).startswith("frozen")
    if not frozen and not smoke:
        raise ValueError("Outcome evaluation requires a frozen protocol")
    if frozen and sha256(Path(__file__).resolve()) != protocol["trainer"]["sha256"]:
        raise ValueError("Frozen protected-bi-sensor trainer hash mismatch")
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


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected"]
    scene = selected["versus_current"]["delta"]
    ap_ci = selected["paired_site_ap_delta"]
    iou_ci = selected["paired_site_pixel_iou_delta"]
    lines = [
        "# Recall-protected bi-sensor anchored fine-tune pilot",
        "",
        "Current scores below 0.25 are exact identities; affected scores are mapped back above 0.25. Teacher-student evidence reranks both sensor families.",
        "",
        f"- Promotion gates pass: **{report['all_promotion_gates_pass']}**",
        f"- Selected strength: **{selected['strength']}**",
        f"- AP delta versus current: **{scene['average_precision']:+.6f}**",
        f"- Matched-FPR recall delta: **{scene['recall_at_fpr_0_0713']:+.6f}**",
        f"- Paired-site AP interval: **[{ap_ci['lower']:+.6f}, {ap_ci['upper']:+.6f}]**",
        f"- Dense-mask IoU delta: **{selected['pixel_iou_delta']:+.6f}**",
        f"- Paired-site IoU interval: **[{iou_ci['lower']:+.6f}, {iou_ci['upper']:+.6f}]**",
        "",
        report["decision"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parent.AnchoredMarsFullFinetune = ProtectedBisensorFinetune
    parent.DEFAULT_PROTOCOL = DEFAULT_PROTOCOL
    parent.verify_protocol = verify_protocol
    parent.write_markdown = write_markdown
    return parent.main()


if __name__ == "__main__":
    raise SystemExit(main())
