#!/usr/bin/env python3
"""One-shot evaluation of frozen Stanford Evanston label-free scores.

The dry-run path validates only protocol and score artifacts and cannot download
outcomes. The real path refuses overwrite, acquires exactly nine protocol-listed
Stanford summary CSVs, verifies repository digests, joins by release_ID, and
applies frozen strata, thresholds, metrics, and uncertainty without retuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta, binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from evaluate_stanford_large_controlled_release_scores import (  # noqa: E402
    binary_metrics,
    decisions,
    paired_date_bootstrap,
    superiority_gate,
    validate_score_bundle,
    validate_score_receipt,
)

DEFAULT_PROTOCOL = Path("configs/stanford_evanston_outcome_evaluation_protocol.json")
REQUIRED_SUMMARY_COLUMNS = (
    "release_ID",
    "date",
    "time_UTC",
    "location",
    "lat",
    "lon",
    "ch4_kgh_mean",
    "ch4_kgh_sigma",
    "ci95_lower",
    "ci95_upper",
    "PredInt95_lower",
    "PredInt95_upper",
    "PI_within_10pct_of_mean",
)
MODEL_CONTRACTS = {
    "released_mars_v3": {
        "score": "released_mars_v3_scores",
        "decision": "released_mars_v3_decisions",
        "threshold": 0.5,
        "comparator": ">",
    },
    "gaussian_dofa": {
        "score": "gaussian_dofa_scores",
        "decision": "gaussian_dofa_decisions",
        "threshold": 0.16728139929966007,
        "comparator": ">=",
    },
    "spatial_prithvi_posttest": {
        "score": "calibrated_spatial_prithvi_scores",
        "decision": "calibrated_spatial_prithvi_decisions",
        "threshold": 0.28187603894788654,
        "comparator": ">=",
    },
}


def digest_bytes(payload: bytes, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    digest.update(payload)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(value: str, *, name: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"{name} must remain beneath repository root") from exc
    return path


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    atomic_bytes(path, payload.encode("utf-8"))


def validate_binding(binding: dict[str, Any], *, name: str) -> Path:
    path = repository_path(str(binding["path"]), name=name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen binding: {name}: {path}")
    if sha256(path) != str(binding["sha256"]):
        raise ValueError(f"Frozen binding hash mismatch: {name}")
    return path


def load_protocol(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("Protocol must remain beneath repository root") from exc
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_first_evanston_outcome_access":
        raise ValueError("Evanston outcome protocol is not frozen")
    if int(protocol["cohort"]["expected_rows"]) != 9:
        raise ValueError("Evanston outcome protocol must bind exactly nine rows")
    if protocol["outcome"]["field"] != "ch4_kgh_mean":
        raise ValueError("Unexpected Evanston outcome field")
    if protocol["outcome"]["negative"] != "ch4_kgh_mean == 0":
        raise ValueError("Negative stratum rule changed")
    if protocol["outcome"]["positive"] != "ch4_kgh_mean >= 1000":
        raise ValueError("Positive stratum rule changed")
    if protocol["outcome"]["challenge"] != "0 < ch4_kgh_mean < 1000":
        raise ValueError("Challenge stratum rule changed")
    sources = protocol["sources"]["summary_files"]
    ids = [str(row["event_id"]) for row in sources]
    if len(sources) != 9 or len(set(ids)) != 9:
        raise ValueError("Outcome source inventory must contain nine unique event IDs")
    for row in sources:
        if not str(row["filename"]).endswith(f"/{row['event_id']}_summary.csv"):
            raise ValueError("Outcome source filename and event ID differ")
        if int(row["size"]) <= 0 or len(str(row["sha1"])) != 40 or len(str(row["md5"])) != 32:
            raise ValueError("Outcome source metadata is incomplete")
    for name, binding in protocol["bindings"].items():
        validate_binding(binding, name=f"bindings.{name}")
    for name, value in protocol["outputs"].items():
        repository_path(str(value), name=f"outputs.{name}")
    return protocol


def load_and_validate_scores(
    protocol_path: Path, protocol: dict[str, Any]
) -> tuple[dict[str, np.ndarray], list[str]]:
    bindings = protocol["bindings"]
    scores_path = validate_binding(bindings["scores"], name="bindings.scores")
    receipt_path = validate_binding(bindings["score_receipt"], name="bindings.score_receipt")
    crop_path = validate_binding(bindings["crop_manifest"], name="bindings.crop_manifest")
    scoring_protocol_path = validate_binding(
        bindings["scoring_protocol"], name="bindings.scoring_protocol"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_score_receipt(receipt, scores_path, scoring_protocol_path, 9)
    with np.load(scores_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    event_ids = validate_score_bundle(arrays, crop_path, expected_rows=9)
    for model, contract in MODEL_CONTRACTS.items():
        score = np.asarray(arrays[contract["score"]], dtype=np.float64)
        observed = np.asarray(arrays[contract["decision"]], dtype=np.uint8)
        expected = decisions(score, float(contract["threshold"]), str(contract["comparator"])).astype(
            np.uint8
        )
        if observed.shape != (9,) or not np.array_equal(observed, expected):
            raise ValueError(f"Frozen score decisions differ from threshold contract: {model}")
    score_manifest_path = validate_binding(
        bindings["score_manifest"], name="bindings.score_manifest"
    )
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    if score_manifest.get("rows") != 9 or [row["event_id"] for row in score_manifest["samples"]] != event_ids:
        raise ValueError("Score manifest identity/order differs from score bundle")
    return arrays, event_ids


def verify_source_bytes(payload: bytes, source: dict[str, Any]) -> None:
    if len(payload) != int(source["size"]):
        raise ValueError(f"Stanford source size mismatch: {source['event_id']}")
    if digest_bytes(payload, "sha1") != str(source["sha1"]):
        raise ValueError(f"Stanford source SHA-1 mismatch: {source['event_id']}")
    if digest_bytes(payload, "md5") != str(source["md5"]):
        raise ValueError(f"Stanford source MD5 mismatch: {source['event_id']}")


def acquire_source(source: dict[str, Any], *, base_url: str, output_dir: Path) -> tuple[Path, bytes]:
    filename = str(source["filename"])
    local_path = output_dir / f"{source['event_id']}_summary.csv"
    if local_path.exists():
        payload = local_path.read_bytes()
        verify_source_bytes(payload, source)
        return local_path, payload
    url = base_url.rstrip("/") + "/" + urllib.parse.quote(filename, safe="/")
    request = urllib.request.Request(url, headers={"User-Agent": "ERSRR-Evanston-one-shot/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    verify_source_bytes(payload, source)
    atomic_bytes(local_path, payload)
    return local_path, payload


def parse_summary(payload: bytes, source: dict[str, Any]) -> dict[str, Any]:
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != REQUIRED_SUMMARY_COLUMNS:
        raise ValueError(f"Stanford summary schema differs: {source['event_id']}")
    rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Stanford summary must have exactly one row: {source['event_id']}")
    row = rows[0]
    if str(row["release_ID"]) != str(source["event_id"]):
        raise ValueError(f"Stanford release_ID differs from frozen event ID: {source['event_id']}")
    numeric: dict[str, float] = {}
    for name in (
        "lat",
        "lon",
        "ch4_kgh_mean",
        "ch4_kgh_sigma",
        "ci95_lower",
        "ci95_upper",
        "PredInt95_lower",
        "PredInt95_upper",
    ):
        numeric[name] = float(row[name])
        if not math.isfinite(numeric[name]):
            raise ValueError(f"Non-finite Stanford summary value: {source['event_id']}:{name}")
    if numeric["ch4_kgh_mean"] < 0.0:
        raise ValueError(f"Negative methane rate: {source['event_id']}")
    return {
        "event_id": str(source["event_id"]),
        "date": str(row["date"]),
        "time_UTC": str(row["time_UTC"]),
        "location": str(row["location"]),
        **numeric,
    }


def exact_binomial_interval(
    successes: int, total: int, *, confidence: float = 0.95
) -> list[float | None]:
    if total == 0:
        return [None, None]
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes)
    )
    return [lower, upper]


def stratum(rate: float) -> str:
    if rate == 0.0:
        return "negative"
    if rate >= 1000.0:
        return "positive"
    return "challenge"


def model_metrics(labels: np.ndarray, scores: np.ndarray, contract: dict[str, Any]) -> dict[str, Any]:
    fixed = binary_metrics(
        labels,
        scores,
        threshold=float(contract["threshold"]),
        comparator=str(contract["comparator"]),
    )
    confusion = fixed["confusion"]
    positives = confusion["tp"] + confusion["fn"]
    negatives = confusion["tn"] + confusion["fp"]
    predicted = confusion["tp"] + confusion["fp"]
    fixed["recall_exact_clopper_pearson_95"] = exact_binomial_interval(confusion["tp"], positives)
    fixed["false_positive_rate_exact_clopper_pearson_95"] = exact_binomial_interval(
        confusion["fp"], negatives
    )
    fixed["precision_exact_clopper_pearson_95"] = exact_binomial_interval(
        confusion["tp"], predicted
    )
    classes = np.unique(labels)
    ranking: dict[str, Any]
    if np.array_equal(classes, np.asarray([0, 1], dtype=classes.dtype)):
        ranking = {
            "average_precision": float(average_precision_score(labels, scores)),
            "auroc_descriptive": float(roc_auc_score(labels, scores)),
        }
    else:
        ranking = {
            "average_precision": None,
            "auroc_descriptive": None,
            "undefined_reason": "primary view does not contain both classes",
        }
    return {
        "rows": int(len(labels)),
        "positive": int(np.sum(labels == 1)),
        "negative": int(np.sum(labels == 0)),
        "ranking": ranking,
        "fixed_threshold": fixed,
    }


def challenge_summary(decision: np.ndarray) -> dict[str, Any]:
    total = int(len(decision))
    detected = int(np.sum(decision))
    return {
        "rows": total,
        "detected": detected,
        "detection_fraction": None if total == 0 else detected / total,
        "exact_clopper_pearson_95": exact_binomial_interval(detected, total),
    }


def paired_comparison(
    labels: np.ndarray,
    dates: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    baseline_contract: dict[str, Any],
    candidate_contract: dict[str, Any],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if len(labels) == 0:
        return {
            "paired_date_bootstrap": None,
            "superiority_gate": {
                "passed": False,
                "undefined_reason": "primary view is empty",
            },
            "exact_mcnemar": {
                "baseline_only_positive": 0,
                "candidate_only_positive": 0,
                "discordant": 0,
                "two_sided_p_value": None,
            },
        }
    bootstrap = paired_date_bootstrap(
        labels,
        baseline_scores,
        candidate_scores,
        dates,
        baseline_threshold=float(baseline_contract["threshold"]),
        baseline_comparator=str(baseline_contract["comparator"]),
        candidate_threshold=float(candidate_contract["threshold"]),
        candidate_comparator=str(candidate_contract["comparator"]),
        replicates=replicates,
        seed=seed,
    )
    baseline_decision = decisions(
        baseline_scores,
        float(baseline_contract["threshold"]),
        str(baseline_contract["comparator"]),
    )
    candidate_decision = decisions(
        candidate_scores,
        float(candidate_contract["threshold"]),
        str(candidate_contract["comparator"]),
    )
    baseline_only = int(np.sum(baseline_decision & ~candidate_decision))
    candidate_only = int(np.sum(~baseline_decision & candidate_decision))
    discordant = baseline_only + candidate_only
    exact_p = None if discordant == 0 else float(
        binomtest(candidate_only, discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {
        "paired_date_bootstrap": bootstrap,
        "superiority_gate": superiority_gate(bootstrap),
        "exact_mcnemar": {
            "baseline_only_positive": baseline_only,
            "candidate_only_positive": candidate_only,
            "discordant": discordant,
            "two_sided_p_value": exact_p,
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    cohort = report["cohort"]
    lines = [
        "# Stanford Evanston Independent-Site One-Shot Evaluation",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Cohort",
        "",
        f"- Frozen events: {cohort['rows']}",
        f"- Primary negatives: {cohort['negative']}",
        f"- Primary positives (>=1,000 kg/h): {cohort['positive']}",
        f"- Challenge events (0-1,000 kg/h): {cohort['challenge']}",
        f"- Location values in source: {', '.join(cohort['locations'])}",
        "",
        "## Models",
        "",
    ]
    for name, result in report["models"].items():
        fixed = result["primary"]["fixed_threshold"]
        ranking = result["primary"]["ranking"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- AP: {ranking['average_precision']}",
                f"- AUROC (descriptive): {ranking['auroc_descriptive']}",
                f"- Recall: {fixed['recall']} (exact 95%: {fixed['recall_exact_clopper_pearson_95']})",
                f"- FPR: {fixed['false_positive_rate']} (exact 95%: {fixed['false_positive_rate_exact_clopper_pearson_95']})",
                f"- Precision: {fixed['precision']} (exact 95%: {fixed['precision_exact_clopper_pearson_95']})",
                f"- Confusion: {fixed['confusion']}",
                f"- Challenge detection: {result['challenge']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "This is a post-test, nine-event evaluation at one independent geographic site. Thresholds were unchanged. It is separate from the Casa Grande one-shot result and cannot by itself establish broad geographic generalization or superiority.",
            "",
            "## License and Attribution",
            "",
            "Stanford source dataset: Reuland et al., Large-Scale Controlled Methane Releases for Satellite-Based Detection and Emission Quantification of Point-Sources, Stanford Digital Repository, CC BY 4.0, DOI 10.25740/qh001qt3946.",
            "",
        ]
    )
    atomic_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL.as_posix())
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = repository_path(args.protocol, name="protocol")
    protocol = load_protocol(protocol_path)
    arrays, event_ids = load_and_validate_scores(protocol_path, protocol)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "immutable Evanston score bundle valid; outcomes not opened",
                    "rows": 9,
                    "scores_sha256": protocol["bindings"]["scores"]["sha256"],
                    "source_files_opened": 0,
                },
                sort_keys=True,
            )
        )
        return 0

    outputs = {name: repository_path(value, name=f"outputs.{name}") for name, value in protocol["outputs"].items()}
    for name, path in outputs.items():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite one-shot output: {name}: {path}")

    source_rows: dict[str, dict[str, Any]] = {}
    source_records: list[dict[str, Any]] = []
    source_dir = outputs["source_manifest"].parent / "summary_files"
    for source in protocol["sources"]["summary_files"]:
        local_path, payload = acquire_source(
            source,
            base_url=str(protocol["sources"]["base_url"]),
            output_dir=source_dir,
        )
        parsed = parse_summary(payload, source)
        event_id = parsed["event_id"]
        if event_id in source_rows:
            raise ValueError(f"Duplicate Stanford outcome ID: {event_id}")
        source_rows[event_id] = parsed
        source_records.append(
            {
                "event_id": event_id,
                "repository_filename": source["filename"],
                "local_path": local_path.relative_to(ROOT).as_posix(),
                "bytes": len(payload),
                "md5": digest_bytes(payload, "md5"),
                "sha1": digest_bytes(payload, "sha1"),
                "sha256": digest_bytes(payload, "sha256"),
            }
        )
    if set(source_rows) != set(event_ids):
        raise ValueError("Outcome source IDs do not exactly match immutable score IDs")

    joined: list[dict[str, Any]] = []
    index_by_id = {event_id: index for index, event_id in enumerate(event_ids)}
    for event_id in event_ids:
        index = index_by_id[event_id]
        outcome = source_rows[event_id]
        rate = float(outcome["ch4_kgh_mean"])
        row: dict[str, Any] = {**outcome, "stratum": stratum(rate)}
        for name in (
            "released_mars_v3_scores",
            "released_mars_v3_decisions",
            "current_oof_extratrees_scores",
            "gaussian_dofa_scores",
            "gaussian_dofa_decisions",
            "calibrated_spatial_prithvi_scores",
            "calibrated_spatial_prithvi_decisions",
        ):
            value = np.asarray(arrays[name])[index]
            row[name] = int(value) if name.endswith("_decisions") else float(value)
        joined.append(row)

    strata = np.asarray([row["stratum"] for row in joined])
    primary_mask = np.isin(strata, ["negative", "positive"])
    challenge_mask = strata == "challenge"
    labels = (strata[primary_mask] == "positive").astype(np.int8)
    dates = np.asarray([row["date"] for row in joined])[primary_mask]
    models: dict[str, Any] = {}
    for model, contract in MODEL_CONTRACTS.items():
        scores = np.asarray(arrays[contract["score"]], dtype=np.float64)
        model_decisions = np.asarray(arrays[contract["decision"]], dtype=np.uint8)
        models[model] = {
            "primary": model_metrics(labels, scores[primary_mask], contract),
            "challenge": challenge_summary(model_decisions[challenge_mask]),
        }

    comparisons: dict[str, Any] = {}
    baseline_contract = MODEL_CONTRACTS["released_mars_v3"]
    baseline_scores = np.asarray(arrays[baseline_contract["score"]], dtype=np.float64)[primary_mask]
    for model in ("gaussian_dofa", "spatial_prithvi_posttest"):
        contract = MODEL_CONTRACTS[model]
        candidate = np.asarray(arrays[contract["score"]], dtype=np.float64)[primary_mask]
        comparisons[f"{model}_minus_released_mars_v3"] = paired_comparison(
            labels,
            dates,
            baseline_scores,
            candidate,
            baseline_contract,
            contract,
            replicates=int(protocol["uncertainty"]["paired_bootstrap_replicates"]),
            seed=int(protocol["uncertainty"]["paired_bootstrap_seed"]),
        )

    source_manifest = {
        "schema_version": 1,
        "status": "complete_verified_public_outcomes",
        "repository": protocol["sources"]["repository"],
        "license": protocol["sources"]["license"],
        "files": source_records,
    }
    report = {
        "schema_version": 1,
        "status": "completed independent-site one-shot; thresholds unchanged",
        "scope": protocol["scope"],
        "protocol_sha256": sha256(protocol_path),
        "score_bundle_sha256": protocol["bindings"]["scores"]["sha256"],
        "source_manifest_sha256": None,
        "cohort": {
            "rows": 9,
            "negative": int(np.sum(strata == "negative")),
            "positive": int(np.sum(strata == "positive")),
            "challenge": int(np.sum(strata == "challenge")),
            "primary_rows": int(np.sum(primary_mask)),
            "UTC_date_blocks": int(len(set(dates.tolist()))),
            "locations": sorted(set(str(row["location"]) for row in joined)),
        },
        "models": models,
        "paired_comparisons": comparisons,
        "uncertainty": protocol["uncertainty"],
        "claim_boundary": protocol["claim_boundary"],
        "source": {
            "repository": protocol["sources"]["repository"],
            "version": protocol["sources"]["repository_version"],
            "license": protocol["sources"]["license"],
            "outcome_field": "ch4_kgh_mean",
            "summary_window": "five minutes preceding approximate satellite overpass",
        },
    }

    atomic_json(outputs["source_manifest"], source_manifest)
    report["source_manifest_sha256"] = sha256(outputs["source_manifest"])
    atomic_jsonl(outputs["joined"], joined)
    report["joined_sha256"] = sha256(outputs["joined"])
    atomic_json(outputs["report_json"], report)
    write_markdown(outputs["report_markdown"], report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cohort": report["cohort"],
                "report_sha256": sha256(outputs["report_json"]),
                "joined_sha256": report["joined_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
