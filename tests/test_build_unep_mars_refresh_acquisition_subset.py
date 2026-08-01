from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from build_unep_mars_refresh_acquisition_subset import select_refresh_subset  # noqa: E402


def test_only_new_resolved_auxiliary_rows_are_selected() -> None:
    cohort = [
        {"sample_id": "old", "research_role": "auxiliary_training"},
        {"sample_id": "new", "research_role": "auxiliary_training"},
        {"sample_id": "dev", "research_role": "development"},
        {"sample_id": "missing", "research_role": "auxiliary_training"},
    ]
    assets = [
        {"sample_id": "old", "status": "resolved"},
        {"sample_id": "new", "status": "resolved"},
        {"sample_id": "dev", "status": "resolved"},
        {"sample_id": "missing", "status": "unresolved"},
    ]
    prior = [{"sample_id": "old"}]
    selected_cohort, selected_assets = select_refresh_subset(cohort, assets, prior)
    assert [row["sample_id"] for row in selected_cohort] == ["new"]
    assert [row["sample_id"] for row in selected_assets] == ["new"]
