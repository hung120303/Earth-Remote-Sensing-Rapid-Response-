from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from aggregate_mars_v4_2_validation import build_campaign  # noqa: E402


def fake_report(seed: int, *, ap: float = 0.82, contract_marker: str = "same") -> dict:
    validation = {
        "average_precision": ap,
        "auroc": 0.95,
        "operating_points": {
            "0.05": {
                "recall": 0.84,
                "false_positive_rate": 0.05,
                "precision": 0.61,
                "threshold": 0.9,
            },
            "0.08": {"recall": 0.88},
        },
        "positive_pixel_dice": 0.70,
    }
    return {
        "scope": "v4_internal_validation_selection",
        "architecture_revision": "v4.2",
        "smoke_test": False,
        "cohort": {
            "strict_spatial_test_loaded": False,
            "group_overlap": 0,
            "training": 100,
            "validation": 20,
            "contract_marker": contract_marker,
        },
        "artifact": {"tracked": False, "sha256": f"checkpoint-{seed}"},
        "model": {"scene_topk_fraction": 0.02, "scene_max_weight": 0.0},
        "source": {"manifest_sha256": "manifest", "revision": "revision"},
        "simulation": {"lut_sha256": "lut", "validation_simulation": False},
        "runtime": {"torch": "test"},
        "provenance": {
            "git_commit": f"commit-{seed}",
            "git_tracked_worktree_dirty_at_start": False,
            "model_source": "model.py",
            "model_source_sha256": "model-hash",
            "script": "train.py",
            "script_sha256": "trainer-hash",
            "simulation_source": "simulation.py",
            "simulation_source_sha256": "simulation-hash",
        },
        "training": {
            "seed": seed,
            "best_epoch": 2,
            "batch_size": 12,
            "epochs_requested": 20,
            "learning_rate": 0.0002,
            "loss": "loss",
            "objective": "segmentation_bce_dice",
            "samples_per_epoch": 4096,
            "scene_max_weight": 0.0,
            "scene_topk_fraction": 0.02,
            "validation_every": 2,
            "history": [
                {"epoch": 1, "validation": None},
                {"epoch": 2, "validation": validation},
            ],
        },
        "validation": validation,
        "v3_internal_reference": {
            "seeds": 5,
            "mean": {
                "average_precision": 0.81,
                "auroc": 0.93,
                "recall_at_fpr5": 0.83,
                "positive_pixel_dice": 0.58,
            },
            "inputs": [],
        },
    }


class MarsV42CampaignTests(unittest.TestCase):
    def write_reports(self, root: Path, reports: list[dict]) -> tuple[Path, ...]:
        paths = []
        for report in reports:
            path = root / f"seed{report['training']['seed']}.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            paths.append(path)
        return tuple(paths)

    def test_three_seed_mean_gate_promotes_only_when_every_gate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_reports(root, [fake_report(seed) for seed in (606, 707, 808)])
            campaign = build_campaign(root, paths, capture_provenance=False)
        self.assertTrue(campaign["strict_evaluation_authorized"])
        self.assertAlmostEqual(campaign["aggregate"]["average_precision"]["mean"], 0.82)
        self.assertEqual([row["seed"] for row in campaign["per_seed"]], [606, 707, 808])

    def test_contract_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_reports(
                root,
                [
                    fake_report(606),
                    fake_report(707, contract_marker="changed"),
                    fake_report(808),
                ],
            )
            with self.assertRaisesRegex(ValueError, "frozen campaign contract"):
                build_campaign(root, paths, capture_provenance=False)

    def test_mislabeled_seed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = [fake_report(seed) for seed in (606, 707, 808)]
            reports[1]["training"]["seed"] = 999
            paths = (
                root / "seed606.json",
                root / "seed707.json",
                root / "seed808.json",
            )
            for path, report in zip(paths, reports):
                path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Expected seed 707"):
                build_campaign(root, paths, capture_provenance=False)


if __name__ == "__main__":
    unittest.main()
