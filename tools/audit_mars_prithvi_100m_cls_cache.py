#!/usr/bin/env python3
"""Bind and audit the label-independent Prithvi-100M folds-3/4 CLS cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acquire_mars_metadata import sha256  # noqa: E402

DEFAULT_CACHE = Path("outputs/mars_prithvi_eo_2_100m_tl_cls_folds34.npz")
DEFAULT_CHAMPION = Path("outputs/mars_dofa_gaussian_champion_folds34_scores.npz")
DEFAULT_PROTOCOL = Path("configs/mars_prithvi_100m_cls_feature_protocol.json")
DEFAULT_RECEIPT = Path("reports/acquisition/mars_prithvi_100m_cls_folds34.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE.as_posix())
    parser.add_argument("--champion", default=DEFAULT_CHAMPION.as_posix())
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT.as_posix())
    args = parser.parse_args()
    cache_path = (ROOT / args.cache).resolve()
    champion_path = (ROOT / args.champion).resolve()
    protocol_path = (ROOT / args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = int(protocol["output"]["expected_rows"])
    with np.load(cache_path, allow_pickle=False) as cache, np.load(
        champion_path, allow_pickle=False
    ) as champion:
        ids = cache["sample_ids"].astype(str)
        champion_ids = champion["sample_ids"].astype(str)
        lookup = {value: index for index, value in enumerate(ids)}
        if ids.size != expected or len(lookup) != expected or set(lookup) != set(champion_ids):
            raise ValueError("Prithvi-100M cache identity contract failed")
        order = np.asarray([lookup[value] for value in champion_ids], dtype=np.int64)
        checks = {
            "finite_features": bool(np.isfinite(cache["features"]).all()),
            "feature_shape": list(cache["features"].shape) == [expected, 3072],
            "labels_equal": bool(np.array_equal(cache["labels"][order], champion["labels"])),
            "sensors_equal": bool(np.array_equal(cache["sensors"][order], champion["sensors"])),
            "folds_equal": bool(np.array_equal(cache["folds"][order], champion["folds"])),
            "groups_equal": bool(
                np.array_equal(cache["groups"][order].astype(str), champion["groups"].astype(str))
            ),
            "checkpoint_equal": str(cache["checkpoint_sha256"]) == protocol["acquisition"]["checkpoint_sha256"],
        }
        if not all(checks.values()):
            raise ValueError(f"Prithvi-100M cache audit failed: {checks}")
        report = {
            "schema_version": 1,
            "scope": "completed label-independent Prithvi-100M CLS cache for MARS folds 3/4",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "cache": {
                "path": args.cache,
                "bytes": cache_path.stat().st_size,
                "sha256": sha256(cache_path),
                "tracked": False,
                "rows": int(ids.size),
                "features": int(cache["features"].shape[1]),
                "dtype": str(cache["features"].dtype),
                "folds": sorted(map(int, np.unique(cache["folds"]))),
            },
            "source": {
                "feature_protocol": args.protocol,
                "feature_protocol_sha256": sha256(protocol_path),
                "champion_cache": args.champion,
                "champion_cache_sha256": sha256(champion_path),
                "checkpoint_sha256": str(cache["checkpoint_sha256"]),
                "foundation_revision": str(cache["foundation_revision"]),
            },
            "alignment_checks": checks,
            "outcomes_used_for_feature_selection": False,
        }
    receipt = (ROOT / args.receipt).resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix(receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt)
    print(json.dumps({"ok": True, **report["cache"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
