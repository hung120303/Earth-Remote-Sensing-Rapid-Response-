from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from score_stanford_evanston_label_free import (  # noqa: E402
    delegated_argv,
    load_and_validate_protocol,
    parse_args,
)
from score_stanford_large_controlled_release_label_free import (  # noqa: E402
    audit_deployability,
)
from train_mars_context_scene_ranker import (  # noqa: E402
    CONTEXT_BASE_FEATURES,
    augment_site_context,
)

EVANSTON_PROTOCOL = ROOT / "configs/stanford_evanston_label_free_scoring_protocol.json"
CASA_PROTOCOL = ROOT / "configs/stanford_large_controlled_release_scoring_protocol.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_evanston_protocol_binds_nine_rows_and_all_dependencies_after_authorization() -> None:
    protocol = load_and_validate_protocol(EVANSTON_PROTOCOL)

    assert protocol["cohort_inputs"]["expected_rows"] == 9
    assert protocol["acquisition"]["pairs"] == 9
    assert protocol["acquisition"]["assets"] == 18
    assert protocol["implementation_gate"]["real_inference_authorized"] is True
    assert protocol["implementation_gate"]["labels_allowed"] is False
    assert protocol["deployment_dependencies"]["pair_manifest"]["sha256"] == (
        protocol["cohort_inputs"]["pair_manifest_sha256"]
    )
    assert protocol["deployment_dependencies"]["crop_manifest"]["sha256"] == (
        protocol["cohort_inputs"]["crop_manifest_sha256"]
    )


def test_final_frozen_protocol_is_deployable_without_shortcuts() -> None:
    protocol = load_and_validate_protocol(EVANSTON_PROTOCOL)

    audit = audit_deployability(protocol)

    assert audit["deployable"] is True
    assert audit["blockers"] == []
    assert audit["no_shortcut_used"] is True


def test_launcher_exposes_no_pair_crop_or_output_override() -> None:
    for forbidden in (
        "--pair-manifest",
        "--crop-manifest",
        "--scores",
        "--score-manifest",
        "--receipt",
        "--limit",
    ):
        with pytest.raises(SystemExit):
            parse_args([forbidden, "arbitrary"])


def test_delegated_argv_uses_only_protocol_bound_paths() -> None:
    protocol = load(EVANSTON_PROTOCOL)
    argv = delegated_argv(
        EVANSTON_PROTOCOL,
        protocol,
        batch_size=3,
        mode="input-preflight",
    )

    dependencies = protocol["deployment_dependencies"]
    outputs = protocol["score_outputs"]
    assert argv[argv.index("--pair-manifest") + 1] == dependencies["pair_manifest"]["path"]
    assert argv[argv.index("--crop-manifest") + 1] == dependencies["crop_manifest"]["path"]
    assert argv[argv.index("--scores") + 1] == outputs["scores"]
    assert argv[argv.index("--score-manifest") + 1] == outputs["score_manifest"]
    assert argv[argv.index("--receipt") + 1] == outputs["receipt"]
    assert "--input-preflight" in argv
    assert "--limit" not in argv


def test_model_preprocessing_fusion_and_threshold_contracts_match_casa_exactly() -> None:
    evanston = load(EVANSTON_PROTOCOL)
    casa = load(CASA_PROTOCOL)

    for section in (
        "released_mars_v3",
        "gaussian_dofa_candidate",
        "spatial_prithvi_posttest_candidate",
    ):
        assert evanston[section] == casa[section]
    evanston_preprocessing = dict(evanston["preprocessing"])
    evanston_preprocessing.pop("site_context_grouping")
    assert evanston_preprocessing == casa["preprocessing"]
    for key in (
        "released_checkpoint",
        "released_config",
        "residual_artifact",
        "current_artifact",
        "gaussian_protocol",
        "gaussian_state",
        "dofa_checkpoint",
        "dofa_deployment_state",
        "spatial_artifact",
        "adaptive_prithvi_artifact",
        "spatial_prithvi_artifact",
        "calibrated_spatial_prithvi_artifact",
    ):
        assert evanston["deployment_dependencies"][key] == casa["deployment_dependencies"][key]


def test_site_group_label_text_is_numerically_inert_for_one_group() -> None:
    generator = np.random.default_rng(20260803)
    names = np.asarray(CONTEXT_BASE_FEATURES)
    features = generator.normal(size=(9, names.size)).astype(np.float64)

    casa, casa_names = augment_site_context(
        features, names, np.asarray(["casa_grande"] * 9)
    )
    evanston, evanston_names = augment_site_context(
        features, names, np.asarray(["evanston_rawhide"] * 9)
    )

    assert casa_names == evanston_names
    assert np.array_equal(casa, evanston)
