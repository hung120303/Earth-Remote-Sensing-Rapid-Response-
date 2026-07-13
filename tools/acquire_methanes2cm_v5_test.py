#!/usr/bin/env python3
"""Unlock and pack the once-authorized MethaneS2CM location test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "EarthRemoteSensingRapidResponse"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from acquire_methanes2cm_v5_train import (  # noqa: E402
    ARCHIVES,
    DATA_ROOT,
    REVISION,
    SPLIT,
    download_one,
    expected_members,
    ignored,
    pack_selected,
    verify_archive,
)
from train_mars_v3 import safe_output, tracked_dirty, write_json  # noqa: E402

DEFAULT_PACKED_TEST = DATA_ROOT / SPLIT / "v5_location_test_packed.h5"
DEFAULT_REPORT = Path("reports/experiments/methanes2cm_v5_location_test_acquisition.json")
ENSEMBLE_REPORT = Path("reports/experiments/methanes2cm_v5_1_ensemble_validation.json")
CAMPAIGN_PROTOCOL = Path("reports/experiments/methanes2cm_v5_1_campaign_protocol.json")
DEVELOPMENT_PROTOCOL = Path("reports/experiments/methanes2cm_v5_protocol.json")
EVALUATOR = Path("tools/evaluate_methanes2cm_v5_1_test.py")

EXPECTED_ENSEMBLE_REPORT_SHA256 = (
    "03691437f3ce2c384aece9f00c7dc4462eebe5e0b580ca340a7db043dc0cdeca"
)
EXPECTED_CAMPAIGN_PROTOCOL_SHA256 = (
    "295f04e00e2d6762920dcd7162daa7fe6b34566fdf82fc09c6b6cfa952553a1f"
)
EXPECTED_DEVELOPMENT_PROTOCOL_SHA256 = (
    "aa7b41070acc26e80ff7a640a98bcaa31d5285af290289013949783e4891fffa"
)
EXPECTED_TEST_CSV_SHA256 = (
    "117a6de1bb8e0d0cb4a746afc2e5f1727e49870a71b9167e578472e89d41d2c4"
)


def verified_freeze(root: Path) -> dict[str, object]:
    paths = {
        ENSEMBLE_REPORT: EXPECTED_ENSEMBLE_REPORT_SHA256,
        CAMPAIGN_PROTOCOL: EXPECTED_CAMPAIGN_PROTOCOL_SHA256,
        DEVELOPMENT_PROTOCOL: EXPECTED_DEVELOPMENT_PROTOCOL_SHA256,
    }
    for relative, expected in paths.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Frozen pre-test identity mismatch: {relative}")
    ensemble = json.loads((root / ENSEMBLE_REPORT).read_text(encoding="utf-8"))
    if not ensemble["freeze"]["location_test_still_sealed"]:
        raise ValueError("The frozen ensemble report does not authorize the sealed boundary")
    if ensemble["cohort"]["location_test_images_opened"]:
        raise ValueError("The ensemble report says location-test images were already opened")
    return ensemble


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DATA_ROOT.as_posix())
    parser.add_argument("--packed-test", default=DEFAULT_PACKED_TEST.as_posix())
    parser.add_argument("--report", default=DEFAULT_REPORT.as_posix())
    args = parser.parse_args()

    root = repo_root()
    if tracked_dirty(root):
        raise RuntimeError("Refusing the one-shot test unlock from a dirty tracked worktree")
    ensemble = verified_freeze(root)
    evaluator_path = root / EVALUATOR
    if not evaluator_path.is_file():
        raise ValueError("The precommitted one-shot evaluator is missing")

    output = (root / args.output).resolve()
    packed_test = (root / args.packed_test).resolve()
    report_path = safe_output(root, args.report)
    split_dir = output / SPLIT
    test_csv = split_dir / "test.csv"
    if sha256(test_csv) != EXPECTED_TEST_CSV_SHA256:
        raise ValueError("Pinned MethaneS2CM test metadata identity mismatch")
    expected, test_rows = expected_members(split_dir, test_csv)
    if len(test_rows) != 20_789 or sum(int(row["label"]) for row in test_rows) != 10_453:
        raise ValueError("Pinned MethaneS2CM test cohort count mismatch")
    if not ignored(root, packed_test):
        raise ValueError("MethaneS2CM packed test output must be ignored by Git")

    archive_paths: list[Path] = []
    for filename, identity in sorted(ARCHIVES.items()):
        path = output / filename
        if not path.is_file():
            path = download_one(output, filename)
        verify_archive(path, identity)
        archive_paths.append(path)
        print(f"Verified {filename}", flush=True)

    extraction = pack_selected(
        archive_paths,
        expected,
        test_rows,
        packed_test,
        source_partition="test",
    )
    report = {
        "schema_version": 1,
        "scope": "methanes2cm_v5_once_authorized_location_test_acquisition",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo_id": "H1deaki/MethaneS2CM",
            "revision": REVISION,
            "split": f"{SPLIT}/test.csv",
            "test_csv_sha256": EXPECTED_TEST_CSV_SHA256,
            "license": "CC-BY-NC-4.0",
        },
        "unlock_authorization": {
            "ensemble_report": {
                "path": ENSEMBLE_REPORT.as_posix(),
                "sha256": EXPECTED_ENSEMBLE_REPORT_SHA256,
                "calibration_cache_sha256": ensemble["calibration_cache"]["sha256"],
            },
            "campaign_protocol": {
                "path": CAMPAIGN_PROTOCOL.as_posix(),
                "sha256": EXPECTED_CAMPAIGN_PROTOCOL_SHA256,
            },
            "development_protocol": {
                "path": DEVELOPMENT_PROTOCOL.as_posix(),
                "sha256": EXPECTED_DEVELOPMENT_PROTOCOL_SHA256,
            },
            "evaluator": {
                "path": EVALUATOR.as_posix(),
                "sha256": sha256(evaluator_path),
            },
        },
        "archives": [
            {
                "name": filename,
                "bytes": identity[0],
                "sha256": identity[1],
                "tracked": False,
            }
            for filename, identity in sorted(ARCHIVES.items())
        ],
        "extraction": {
            **extraction,
            "packed_path": packed_test.relative_to(root).as_posix(),
            "tracked": False,
        },
        "seal_transition": {
            "images_opened_before_this_committed_unlock": False,
            "images_opened_by_this_acquisition": True,
            "retuning_from_test_outcomes_permitted": False,
        },
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "script": Path(__file__).resolve().relative_to(root).as_posix(),
            "script_sha256": sha256(Path(__file__).resolve()),
            "tracked_worktree_dirty_at_start": False,
        },
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
