#!/usr/bin/env python3
"""Cross-fit AdaBN-style target-normalized Prithvi CLS scene probes."""

from __future__ import annotations
import argparse, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib, numpy as np, sklearn
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "EarthRemoteSensingRapidResponse", ROOT / "tools"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_crossfold_bagged_scene_head import (  # noqa: E402
    FOLDS, DEFAULT_FOLD0_CACHE, DEFAULT_FOLD0_SHA256, DEFAULT_FOLD1_CACHE,
    DEFAULT_FOLD1_SHA256, DEFAULT_INNER_CACHE, DEFAULT_INNER_SHA256,
    DEFAULT_SCORE_CACHE, DEFAULT_SCORE_SHA256, load_development)
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap  # noqa: E402
from train_mars_target_weighted_scene_head import evaluate_candidate  # noqa: E402
from train_mars_scene_ranker import blend_scores  # noqa: E402

DEFAULT_PRITHVI = Path("outputs/mars_prithvi_eo_2_tiny_tl_features_all_folds.npz")
DEFAULT_PRITHVI_SHA256 = "e3e52a9453426e5e048cd753daf2597d59cbe820a18ae584c61a2de7ae405f23"
DEFAULT_ARTIFACT = Path("EarthRemoteSensingRapidResponse/artifacts/mars_adaptive_prithvi_probe.joblib")
DEFAULT_JSON = Path("reports/experiments/mars_adaptive_prithvi_probe.json")
DEFAULT_MARKDOWN = Path("reports/experiments/MARS_ADAPTIVE_PRITHVI_PROBE.md")
FEATURE_SETS = ("cls", "cls_plus_base")
REGULARIZATION = (0.001, 0.01, 0.1)
BLENDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
CLS_WIDTH = 768

def domain_normalize(source: np.ndarray, target: np.ndarray, epsilon: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Normalize source and unlabeled target with their own feature moments."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_scale = np.maximum(source.std(0), epsilon)
    target_scale = np.maximum(target.std(0), epsilon)
    return (source-source_mean)/source_scale, (target-target_mean)/target_scale

def load_features(path: Path, values: dict[str, Any], feature_set: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as cache:
        ids, encoded = cache["sample_ids"].astype(str), cache["features"][:, :CLS_WIDTH].astype(np.float32)
        names = cache["feature_names"][:CLS_WIDTH].astype(str)
    if names.size != CLS_WIDTH or not names[0].startswith("prithvi_block3_cls_") or not names[-1].startswith("prithvi_block12_cls_"):
        raise ValueError("Prithvi CLS schema differs")
    lookup = {value: index for index, value in enumerate(ids)}
    if len(lookup) != ids.size: raise ValueError("Prithvi IDs are not unique")
    aligned = encoded[np.asarray([lookup[value] for value in values["sample_ids"]])]
    return aligned if feature_set == "cls" else np.concatenate((aligned, values["features"].astype(np.float32)), axis=1)

def oof_scores(features: np.ndarray, values: dict[str, Any], c_value: float) -> np.ndarray:
    scores = np.empty(values["labels"].shape, dtype=np.float64)
    for holdout in FOLDS:
        fit, held = values["folds"] != holdout, values["folds"] == holdout
        source, target = domain_normalize(features[fit].astype(np.float64), features[held].astype(np.float64))
        positive_weight = float(np.sqrt((values["labels"][fit] == 0).sum() / (values["labels"][fit] == 1).sum()))
        weights = np.where(values["labels"][fit] == 1, positive_weight, 1.0)
        model = LogisticRegression(C=c_value, max_iter=500, solver="lbfgs", random_state=20261500+holdout).fit(source, values["labels"][fit], sample_weight=weights)
        scores[held] = model.predict_proba(target)[:, 1]
        print(json.dumps({"C": c_value, "completed_holdout": holdout, "features": features.shape[1]}), flush=True)
    return scores

def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8"); os.replace(tmp,path)

def write_markdown(path: Path, report: dict[str, Any]) -> None:
    s=report["selected"]; d=s["versus_current"]["delta"]; i=s["paired_group_bootstrap_ap_delta_vs_current"]
    lines=["# Adaptive Prithvi CLS probe","", "Source and each unlabeled target fold use independent feature moments before a source-label logistic fit.","",
        f"- Feature set / C / blend: `{s['feature_set']}` / {s['C']} / {s['blend_weight']:.2f}",
        f"- AP delta vs current: {d['average_precision']:+.5f}", f"- Recall delta vs current: {d['recall_at_fpr_0_0713']:+.5f}",
        f"- Paired-site AP interval: [{i['lower']:+.5f}, {i['upper']:+.5f}]","",report["decision"]]
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text("\n".join(lines)+"\n",encoding="utf-8")

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prithvi",default=DEFAULT_PRITHVI.as_posix()); p.add_argument("--prithvi-sha256",default=DEFAULT_PRITHVI_SHA256)
    p.add_argument("--inner-cache",default=DEFAULT_INNER_CACHE.as_posix()); p.add_argument("--inner-sha256",default=DEFAULT_INNER_SHA256)
    p.add_argument("--fold0-cache",default=DEFAULT_FOLD0_CACHE.as_posix()); p.add_argument("--fold0-sha256",default=DEFAULT_FOLD0_SHA256)
    p.add_argument("--fold1-cache",default=DEFAULT_FOLD1_CACHE.as_posix()); p.add_argument("--fold1-sha256",default=DEFAULT_FOLD1_SHA256)
    p.add_argument("--score-cache",default=DEFAULT_SCORE_CACHE.as_posix()); p.add_argument("--score-sha256",default=DEFAULT_SCORE_SHA256)
    p.add_argument("--artifact",default=DEFAULT_ARTIFACT.as_posix()); p.add_argument("--output-json",default=DEFAULT_JSON.as_posix()); p.add_argument("--output-markdown",default=DEFAULT_MARKDOWN.as_posix())
    a=p.parse_args(); root=repo_root()
    paths={"prithvi":(root/a.prithvi).resolve(),"inner":(root/a.inner_cache).resolve(),"fold0":(root/a.fold0_cache).resolve(),"fold1":(root/a.fold1_cache).resolve(),"score":(root/a.score_cache).resolve()}
    expected={"prithvi":a.prithvi_sha256,"inner":a.inner_sha256,"fold0":a.fold0_sha256,"fold1":a.fold1_sha256,"score":a.score_sha256}
    for name,digest in expected.items():
        if sha256(paths[name]) != digest: raise ValueError(f"Frozen {name} hash mismatch")
    values=load_development({name:paths[name] for name in ("inner","fold0","fold1")},paths["score"])
    candidates=[]; raw_store={}
    for feature_set in FEATURE_SETS:
        features=load_features(paths["prithvi"],values,feature_set)
        for c_value in REGULARIZATION:
            raw=oof_scores(features,values,c_value); raw_store[(feature_set,c_value)]=raw
            for blend in BLENDS:
                candidate=evaluate_candidate(values,raw,{"feature_set":feature_set,"C":c_value},blend)
                candidate.update({"feature_set":feature_set,"C":c_value,"blend_weight":blend})
                candidates.append(candidate)
    selected=max(candidates,key=lambda value:tuple(value["rank"])); raw=raw_store[(selected["feature_set"],selected["C"])]
    scores=blend_scores(values["current"],raw,selected["blend_weight"])
    selected["paired_group_bootstrap_ap_delta_vs_primary"]=ap_group_bootstrap(values["labels"],values["primary"],scores,values["groups"],replicates=10_000,seed=20261520)
    selected["paired_group_bootstrap_ap_delta_vs_current"]=ap_group_bootstrap(values["labels"],values["current"],scores,values["groups"],replicates=10_000,seed=20261521)
    passed=bool(selected["stable"] and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"]>0 and selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"]>0)
    artifact=(root/a.artifact).resolve(); artifact.parent.mkdir(parents=True,exist_ok=True); tmp=artifact.with_suffix(artifact.suffix+".tmp")
    joblib.dump({"schema_version":1,"kind":"mars_adaptive_prithvi_control","feature_set":selected["feature_set"],"C":selected["C"],"blend_weight":selected["blend_weight"],"label_contract":"target labels excluded from moment estimates and fitting"},tmp,compress=3); os.replace(tmp,artifact)
    report={"schema_version":1,"scope":"five-fold unlabeled-target adaptive Prithvi probe; paper cache not loaded","generated_at_utc":datetime.now(timezone.utc).isoformat(),"feature_sets":list(FEATURE_SETS),"regularization":list(REGULARIZATION),"blends":list(BLENDS),"selected":selected,"all_promotion_gates_pass":passed,"decision":"Freeze adaptive Prithvi probe for one label-free paper adaptation." if passed else "Reject adaptive Prithvi probe before paper scoring.","provenance":{**{f"{n}_sha256":d for n,d in expected.items()},"artifact_sha256":sha256(artifact),"script_sha256":sha256(Path(__file__).resolve()),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),"numpy":np.__version__,"sklearn":sklearn.__version__}}
    write_json((root/a.output_json).resolve(),report); write_markdown((root/a.output_markdown).resolve(),report)
    print(json.dumps({"ok":passed,"feature_set":selected["feature_set"],"C":selected["C"],"blend":selected["blend_weight"],"ap_delta":selected["versus_current"]["delta"]["average_precision"],"recall_delta":selected["versus_current"]["delta"]["recall_at_fpr_0_0713"],"ap_lower":selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"]},indent=2)); return 0 if passed else 2
if __name__=="__main__": raise SystemExit(main())
