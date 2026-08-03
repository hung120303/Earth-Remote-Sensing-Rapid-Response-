from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
MODEL_ROOT = ROOT / "EarthRemoteSensingRapidResponse"
for path in (TOOLS, MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MODULE_PATH = TOOLS / "score_mars_gaussian_dofa_strict_spatial_label_free.py"
SPEC = importlib.util.spec_from_file_location("strict_gaussian_dofa_scorer", MODULE_PATH)
assert SPEC and SPEC.loader
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)

PROTOCOL_PATH = ROOT / "configs/mars_gaussian_dofa_strict_spatial_scoring_protocol.json"
METADATA_ROOT = ROOT / "EarthRemoteSensingRapidResponse/Data Collection/s2_emit_pairs/publication-v1/external/MARS-S2L"
CATALOG_PATH = METADATA_ROOT / "publication_v3_strict_remote_catalog.jsonl"
CACHE_PATH = ROOT / "outputs/mars_paper_scene_features_label_free.npz"
LABELED_MANIFEST = METADATA_ROOT / "publication_v3_strict_samples.jsonl"


@pytest.fixture(scope="module")
def strict_inputs():
    rows = scorer.load_label_free_asset_catalog(CATALOG_PATH)
    cache = scorer.load_label_free_base_cache(CACHE_PATH)
    return scorer.align_strict_inputs(rows, cache)


def test_protocol_is_authorized_after_preflight_and_candidate_specific():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["status"] == "authorized_and_frozen_before_candidate_strict_spatial_inference"
    assert protocol["implementation_gate"]["real_inference_authorized"] is True
    assert protocol["cohort"] == {
        "selection": "all image/cloud pairs in the exact strict remote catalog",
        "rows": 4401,
        "strict_spatial_components": 150,
        "label_free_site_context_groups": 373,
        "countries": 32,
        "sensor": "Sentinel-2 MSI",
        "group_unit": "frozen 25 km connected spatial component",
        "fallback_rows": 0,
        "partial_scoring_forbidden": True,
    }
    assert protocol["candidate"]["operational_threshold"] == 0.16728139929966007
    assert protocol["candidate"]["retuning_forbidden"] is True
    assert protocol["scientific_boundary"]["project_level_holdout_status"].startswith("not pristine")
    assert scorer.sha256(MODULE_PATH) == protocol["scoring_dependencies"]["scorer"]["sha256"]


def test_cli_exposes_no_input_output_or_subset_shortcut():
    with pytest.raises(SystemExit):
        scorer.parse_args(["--catalog", "replacement.jsonl"])
    with pytest.raises(SystemExit):
        scorer.parse_args(["--limit", "2"])
    with pytest.raises(SystemExit):
        scorer.parse_args(["--output", "replacement.npz"])


def test_dependency_safety_blocks_outcome_and_labeled_paths():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    scorer.assert_scoring_dependency_safety(protocol["scoring_dependencies"])
    for path in (
        "publication_v3_strict_samples.jsonl",
        "validated_images_all.csv",
        "mars_paper_test_v3_diagnostic_cache.npz",
        "some_plume_mask.tif",
    ):
        with pytest.raises(ValueError, match="Forbidden candidate-scoring dependency"):
            scorer.assert_scoring_dependency_safety(
                {"bad": {"path": path, "sha256": "0" * 64}}
            )


def test_catalog_is_complete_input_only_and_aligns_all_rows(strict_inputs):
    rows, aligned = strict_inputs
    assert len(rows) == 4401
    assert len({row["sample_id"] for row in rows}) == 4401
    assert all(set(row["assets"]) == {"image", "cloud_mask"} for row in rows)
    assert aligned["sample_ids"].shape == (4401,)
    assert np.unique(aligned["groups"]).size == 373
    assert np.isfinite(aligned["released_scores"]).all()
    assert np.isfinite(aligned["current_scores"]).all()
    serialized = json.dumps(rows)
    assert "plume_mask" not in serialized
    assert "methane_enhancement" not in serialized
    assert "label_state" not in serialized


def test_cached_wind_reconstruction_matches_original_metadata(strict_inputs):
    from evaluate_released_marss2l import wind_lookup

    _, aligned = strict_inputs
    expected_lookup = wind_lookup(
        METADATA_ROOT / "validated_images_all.csv", set(aligned["sample_ids"].tolist())
    )
    expected = np.asarray([expected_lookup[sample_id] for sample_id in aligned["sample_ids"]])
    observed = np.stack((aligned["wind_u"], aligned["wind_v"]), axis=1)
    assert np.max(np.abs(expected - observed)) <= 3.0e-6


def test_direct_input_loader_matches_verified_adapter_on_real_negative(strict_inputs):
    from evaluate_released_marss2l import released_input
    from mars_s2l_adapter import load_sample

    rows, aligned = strict_inputs
    records = [json.loads(line) for line in LABELED_MANIFEST.read_text(encoding="utf-8").splitlines()]
    record = next(item for item in records if {asset["role"] for asset in item["assets"]} == {"image", "cloud_mask"})
    sample_id = str(record["sample_id"])
    index = int(np.flatnonzero(aligned["sample_ids"] == sample_id)[0])
    row = rows[index]
    direct_input, direct_observable = scorer.load_input_batch(
        METADATA_ROOT,
        [row],
        aligned["wind_u"][index : index + 1],
        aligned["wind_v"][index : index + 1],
    )
    sample = load_sample(METADATA_ROOT, record, require_enhancement=False)
    expected_input = released_input(
        sample,
        (float(aligned["wind_u"][index]), float(aligned["wind_v"][index])),
        "mars",
    )
    np.testing.assert_allclose(direct_input[0], expected_input, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(direct_observable[0, 0], sample.observable_mask)


def test_output_writer_is_input_only_and_refuses_overwrite(tmp_path):
    sandbox = ROOT / ".research" / f"pytest_strict_score_{tmp_path.name}"
    shutil.rmtree(sandbox, ignore_errors=True)
    sandbox.mkdir(parents=True)
    rows = [
        {
            "sample_id": "00000000-0000-0000-0000-000000000001",
            "assets": {
                "image": {"path": "data/image.tif", "size": 1, "remote_oid": "a", "remote_oid_type": "sha256_lfs"},
                "cloud_mask": {"path": "data/cloud.tif", "size": 1, "remote_oid": "b", "remote_oid_type": "sha256_lfs"},
            },
        }
    ]
    arrays = {
        "sample_ids": np.asarray([rows[0]["sample_id"]]),
        "groups": np.asarray(["group"]),
        "released_mars_v3_scores": np.asarray([0.1]),
        "current_v3_scores": np.asarray([0.2]),
        "gaussian_raw_logits": np.asarray([0.3]),
        "dofa_raw_scores": np.asarray([0.4]),
        "gaussian_dofa_scores": np.asarray([0.25]),
        "released_mars_v3_decisions": np.asarray([0], dtype=np.uint8),
        "gaussian_dofa_decisions": np.asarray([1], dtype=np.uint8),
    }
    protocol = sandbox / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    score = sandbox / "scores.npz"
    manifest = sandbox / "manifest.json"
    receipt = sandbox / "receipt.json"
    try:
        scorer.write_outputs(
            root=ROOT,
            arrays=arrays,
            rows=rows,
            protocol_path=protocol,
            score_path=score,
            manifest_path=manifest,
            receipt_path=receipt,
            runtime={"device": "test"},
        )
        with np.load(score, allow_pickle=False) as payload:
            assert not any("label" in name or "truth" in name for name in payload.files)
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert manifest_payload["forbidden_assets_opened"] is False
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            scorer.write_outputs(
                root=ROOT,
                arrays=arrays,
                rows=rows,
                protocol_path=protocol,
                score_path=score,
                manifest_path=manifest,
                receipt_path=receipt,
                runtime={"device": "test"},
            )
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
