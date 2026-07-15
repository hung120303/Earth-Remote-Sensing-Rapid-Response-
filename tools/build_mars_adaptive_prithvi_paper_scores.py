#!/usr/bin/env python3
"""Fit the frozen adaptive Prithvi probe and score label-free paper inputs."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT/"EarthRemoteSensingRapidResponse",ROOT/"tools"):
    if str(path) not in sys.path: sys.path.insert(0,str(path))
from acquire_mars_metadata import repo_root, sha256  # noqa: E402
from train_mars_adaptive_prithvi_probe import CLS_WIDTH, domain_normalize, load_features  # noqa: E402
from train_mars_context_scene_ranker import augment_site_context  # noqa: E402
from train_mars_crossfold_bagged_scene_head import (  # noqa: E402
    DEFAULT_FOLD0_CACHE,DEFAULT_FOLD0_SHA256,DEFAULT_FOLD1_CACHE,DEFAULT_FOLD1_SHA256,
    DEFAULT_INNER_CACHE,DEFAULT_INNER_SHA256,DEFAULT_SCORE_CACHE,DEFAULT_SCORE_SHA256,load_development)
from train_mars_scene_ranker import blend_scores  # noqa: E402

DEFAULT_DEV_PRITHVI=Path("outputs/mars_prithvi_eo_2_tiny_tl_features_all_folds.npz")
DEFAULT_DEV_PRITHVI_SHA256="e3e52a9453426e5e048cd753daf2597d59cbe820a18ae584c61a2de7ae405f23"
DEFAULT_PAPER_BASE=Path("outputs/mars_paper_scene_features_label_free.npz")
DEFAULT_PAPER_BASE_SHA256="8a35e60e7c396e58639f940239020adb36def885124841e0b20901e10db52f33"
DEFAULT_PAPER_PRITHVI=Path("outputs/mars_paper_prithvi_cls_features.npz")
DEFAULT_PAPER_PRITHVI_SHA256="d3d9bfb6423fe9ac6bf53185ea408a476491ca8fa31e941e1782a8c85a016795"
DEFAULT_CONTROL=Path("EarthRemoteSensingRapidResponse/artifacts/mars_adaptive_prithvi_probe.joblib")
DEFAULT_CONTROL_SHA256="a38be1acc8ca425ef5307c0ad2b253274fd36d4c3e6b5ed63ce1bf205f6fb0d5"
DEFAULT_REPORT=Path("reports/experiments/mars_adaptive_prithvi_probe.json")
DEFAULT_REPORT_SHA256="3a0478d9abdb94a76815744cefcf7ac181765c20e2969a7a40f310f272a03a8b"
DEFAULT_OUTPUT=Path("outputs/mars_adaptive_prithvi_paper_scores.npz")
DEFAULT_RECEIPT=Path("reports/acquisition/mars_adaptive_prithvi_paper_scores.json")

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev-prithvi",default=DEFAULT_DEV_PRITHVI.as_posix());p.add_argument("--dev-prithvi-sha256",default=DEFAULT_DEV_PRITHVI_SHA256)
    p.add_argument("--paper-base",default=DEFAULT_PAPER_BASE.as_posix());p.add_argument("--paper-base-sha256",default=DEFAULT_PAPER_BASE_SHA256)
    p.add_argument("--paper-prithvi",default=DEFAULT_PAPER_PRITHVI.as_posix());p.add_argument("--paper-prithvi-sha256",default=DEFAULT_PAPER_PRITHVI_SHA256)
    p.add_argument("--control",default=DEFAULT_CONTROL.as_posix());p.add_argument("--control-sha256",default=DEFAULT_CONTROL_SHA256)
    p.add_argument("--development-report",default=DEFAULT_REPORT.as_posix());p.add_argument("--development-report-sha256",default=DEFAULT_REPORT_SHA256)
    p.add_argument("--inner-cache",default=DEFAULT_INNER_CACHE.as_posix());p.add_argument("--inner-sha256",default=DEFAULT_INNER_SHA256)
    p.add_argument("--fold0-cache",default=DEFAULT_FOLD0_CACHE.as_posix());p.add_argument("--fold0-sha256",default=DEFAULT_FOLD0_SHA256)
    p.add_argument("--fold1-cache",default=DEFAULT_FOLD1_CACHE.as_posix());p.add_argument("--fold1-sha256",default=DEFAULT_FOLD1_SHA256)
    p.add_argument("--score-cache",default=DEFAULT_SCORE_CACHE.as_posix());p.add_argument("--score-sha256",default=DEFAULT_SCORE_SHA256)
    p.add_argument("--output",default=DEFAULT_OUTPUT.as_posix());p.add_argument("--receipt",default=DEFAULT_RECEIPT.as_posix());a=p.parse_args();root=repo_root()
    paths={"dev_prithvi":(root/a.dev_prithvi).resolve(),"paper_base":(root/a.paper_base).resolve(),"paper_prithvi":(root/a.paper_prithvi).resolve(),"control":(root/a.control).resolve(),"development_report":(root/a.development_report).resolve(),"inner":(root/a.inner_cache).resolve(),"fold0":(root/a.fold0_cache).resolve(),"fold1":(root/a.fold1_cache).resolve(),"score":(root/a.score_cache).resolve()}
    expected={"dev_prithvi":a.dev_prithvi_sha256,"paper_base":a.paper_base_sha256,"paper_prithvi":a.paper_prithvi_sha256,"control":a.control_sha256,"development_report":a.development_report_sha256,"inner":a.inner_sha256,"fold0":a.fold0_sha256,"fold1":a.fold1_sha256,"score":a.score_sha256}
    for name,digest in expected.items():
        if sha256(paths[name])!=digest: raise ValueError(f"Frozen {name} hash mismatch")
    report=json.loads(paths["development_report"].read_text(encoding="utf-8"));control=joblib.load(paths["control"])
    if report.get("all_promotion_gates_pass") is not True or any(control[k]!=report["selected"][k] for k in ("feature_set","C","blend_weight")): raise ValueError("Adaptive Prithvi control/report mismatch")
    values=load_development({n:paths[n] for n in ("inner","fold0","fold1")},paths["score"])
    source=load_features(paths["dev_prithvi"],values,"cls_plus_base").astype(np.float64)
    with np.load(paths["paper_base"],allow_pickle=False) as cache:
        ids=cache["sample_ids"].astype(str);groups=cache["groups"].astype(str);base=cache["base_features"].astype(np.float32);names=cache["base_feature_names"].astype(str);current=cache["current_v3_scores"].astype(np.float64)
    augmented,augmented_names=augment_site_context(base,names,groups)
    if augmented_names!=values["augmented_feature_names"]: raise ValueError("Paper base schema mismatch")
    with np.load(paths["paper_prithvi"],allow_pickle=False) as cache:
        if "labels" in cache.files: raise ValueError("Paper Prithvi cache contains labels")
        pids=cache["sample_ids"].astype(str);encoded=cache["features"][:,:CLS_WIDTH].astype(np.float32)
    lookup={value:index for index,value in enumerate(pids)}
    target=np.concatenate((encoded[np.asarray([lookup[value] for value in ids])],augmented.astype(np.float32)),axis=1).astype(np.float64)
    source_norm,target_norm=domain_normalize(source,target);positive_weight=float(np.sqrt((values["labels"]==0).sum()/(values["labels"]==1).sum()));weights=np.where(values["labels"]==1,positive_weight,1.0)
    model=LogisticRegression(C=float(control["C"]),max_iter=500,solver="lbfgs",random_state=20261550).fit(source_norm,values["labels"],sample_weight=weights)
    raw=model.predict_proba(target_norm)[:,1];scores=blend_scores(current,raw,float(control["blend_weight"]))
    output=(root/a.output).resolve();output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_suffix(output.suffix+".tmp")
    with tmp.open("wb") as handle: np.savez_compressed(handle,sample_ids=ids,scores=scores)
    os.replace(tmp,output)
    receipt={"schema_version":1,"scope":"label-free adaptive Prithvi scores for exact available paper rows","generated_at_utc":datetime.now(timezone.utc).isoformat(),"rows":int(ids.size),"labels_accessed":False,"output_sha256":sha256(output),"script_sha256":sha256(Path(__file__).resolve()),"inputs":{f"{n}_sha256":d for n,d in expected.items()},"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()}
    path=(root/a.receipt).resolve();path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(temp,path);print(json.dumps(receipt,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
