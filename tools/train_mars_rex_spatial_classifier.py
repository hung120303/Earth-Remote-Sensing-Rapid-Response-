#!/usr/bin/env python3
"""Train a multi-seed V-REx spatial scene ensemble across MARS environments."""

from __future__ import annotations
import argparse, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT/"EarthRemoteSensingRapidResponse",ROOT/"tools"):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from acquire_mars_metadata import repo_root,sha256  # noqa:E402
from train_mars_oof_scene_ensemble_v2 import ap_group_bootstrap,sample_weights  # noqa:E402
from train_mars_scene_ranker import blend_scores  # noqa:E402
from train_mars_spatial_scene_classifier import (  # noqa:E402
    DEFAULT_IMAGES,DEFAULT_IMAGES_SHA256,DEFAULT_METADATA,DEFAULT_METADATA_SHA256,
    DEFAULT_SCORE_CACHE,DEFAULT_SCORE_SHA256,DEFAULT_INNER_CACHE,DEFAULT_INNER_SHA256,
    DEFAULT_FOLD0_CACHE,DEFAULT_FOLD0_SHA256,DEFAULT_FOLD1_CACHE,DEFAULT_FOLD1_SHA256,
    SpatialSceneClassifier,augment_batch,channel_indices,load_partitions,predict_model)
from train_mars_target_weighted_scene_head import evaluate_candidate  # noqa:E402

DEFAULT_ARTIFACT=Path("EarthRemoteSensingRapidResponse/artifacts/mars_rex_spatial_classifier.pt")
DEFAULT_JSON=Path("reports/experiments/mars_rex_spatial_classifier.json")
DEFAULT_MARKDOWN=Path("reports/experiments/MARS_REX_SPATIAL_CLASSIFIER.md")
FOLDS=(0,1,2,3,4);BETAS=(0.5,2.0);SEEDS=(20261600,20261700);BLENDS=(0.05,0.1,0.2,0.3,0.4)

def rex_objective(losses:torch.Tensor,environments:torch.Tensor,weights:torch.Tensor,beta:float)->tuple[torch.Tensor,torch.Tensor]:
    risks=[]
    for environment in torch.unique(environments):
        rows=environments==environment
        risks.append((losses[rows]*weights[rows]).sum()/weights[rows].sum().clamp_min(1e-8))
    values=torch.stack(risks);return values.mean()+beta*values.var(unbiased=False),values.detach()

def combine_partitions(parts:dict[str,dict[str,np.ndarray]])->dict[str,np.ndarray]:
    order=("fold0","fold1","inner");counts=[parts[n]["labels"].size for n in order]
    return {"image_indices":np.concatenate([parts[n]["image_indices"] for n in order]),"labels":np.concatenate([parts[n]["labels"] for n in order]),"sensors":np.concatenate([parts[n]["sensors"] for n in order]),"groups":np.concatenate([parts[n]["groups"] for n in order]),"primary":np.concatenate([parts[n]["primary"] for n in order]),"current":np.concatenate([parts[n]["new"] for n in order]),"folds":np.concatenate((np.zeros(counts[0],dtype=np.uint8),np.ones(counts[1],dtype=np.uint8),parts["inner"]["folds"].astype(np.uint8)))}

def train_model(images:np.ndarray,partition:dict[str,np.ndarray],fit:np.ndarray,beta:float,seed:int,device:torch.device)->dict[str,Any]:
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    channels=channel_indices("physics_spatial");model=SpatialSceneClassifier(len(channels),0.3).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.003)
    labels=partition["labels"][fit];sensors=partition["sensors"][fit];folds=partition["folds"][fit];indices=partition["image_indices"][fit];groups=partition["groups"][fit]
    row_weights=sample_weights("site_cell",groups,labels,sensors);positive_weight=min(4.0,float(np.sqrt(row_weights[labels==0].sum()/max(row_weights[labels==1].sum(),1e-8))));generator=torch.Generator(device="cpu").manual_seed(seed);model.train()
    for epoch in range(8):
        order=torch.randperm(labels.size,generator=generator).numpy()
        risks=[]
        for start in range(0,labels.size,160):
            rows=order[start:start+160];array=np.asarray(images[indices[rows]][:,channels],dtype=np.float32);values=augment_batch(torch.from_numpy(array),generator).to(device);target=torch.from_numpy(labels[rows].astype(np.float32)).to(device);sensor=torch.from_numpy(sensors[rows].astype(np.int64)).to(device);environment=torch.from_numpy(folds[rows].astype(np.int64)).to(device);weight=torch.from_numpy(row_weights[rows].astype(np.float32)).to(device);class_weight=torch.where(target>.5,positive_weight,1.0);losses=F.binary_cross_entropy_with_logits(model(values,sensor),target,reduction="none");loss,local=rex_objective(losses,environment,weight*class_weight,beta);optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step();risks.append(local.cpu().numpy())
        print(json.dumps({"seed":seed,"beta":beta,"epoch":epoch+1,"mean_environment_risk":float(np.concatenate(risks).mean())}),flush=True)
    return {"state_dict":{n:v.detach().cpu() for n,v in model.state_dict().items()},"input_channels":len(channels),"channel_indices":channels,"dropout":0.3,"positive_weight":positive_weight,"beta":beta,"seed":seed}

def crossfit(images:np.ndarray,partition:dict[str,np.ndarray],beta:float,seed:int,device:torch.device)->tuple[np.ndarray,list[dict[str,Any]]]:
    scores=np.empty(partition["labels"].shape,dtype=np.float64);models=[]
    for holdout in FOLDS:
        fit=partition["folds"]!=holdout;held=~fit;fitted=train_model(images,partition,fit,beta,seed+holdout,device);scores[held]=predict_model(fitted,images,partition["image_indices"][held],partition["sensors"][held],device);models.append({"holdout":holdout,"fitted":fitted});print(json.dumps({"seed":seed,"beta":beta,"completed_holdout":holdout}),flush=True)
    return scores,models

def write_json(path:Path,value:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(tmp,path)
def write_markdown(path:Path,report:dict[str,Any])->None:
    s=report["selected"];d=s["versus_current"]["delta"];i=s["paired_group_bootstrap_ap_delta_vs_current"];lines=["# Multi-environment V-REx spatial classifier","",f"- Beta / blend: {s['beta']} / {s['blend_weight']:.2f}",f"- AP delta vs current: {d['average_precision']:+.5f}",f"- Recall delta vs current: {d['recall_at_fpr_0_0713']:+.5f}",f"- Paired-site AP interval: [{i['lower']:+.5f}, {i['upper']:+.5f}]","",report["decision"]];path.parent.mkdir(parents=True,exist_ok=True);path.write_text("\n".join(lines)+"\n",encoding="utf-8")
def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--images",default=DEFAULT_IMAGES.as_posix());p.add_argument("--images-sha256",default=DEFAULT_IMAGES_SHA256);p.add_argument("--metadata",default=DEFAULT_METADATA.as_posix());p.add_argument("--metadata-sha256",default=DEFAULT_METADATA_SHA256);p.add_argument("--score-cache",default=DEFAULT_SCORE_CACHE.as_posix());p.add_argument("--score-sha256",default=DEFAULT_SCORE_SHA256);p.add_argument("--inner-cache",default=DEFAULT_INNER_CACHE.as_posix());p.add_argument("--inner-sha256",default=DEFAULT_INNER_SHA256);p.add_argument("--fold0-cache",default=DEFAULT_FOLD0_CACHE.as_posix());p.add_argument("--fold0-sha256",default=DEFAULT_FOLD0_SHA256);p.add_argument("--fold1-cache",default=DEFAULT_FOLD1_CACHE.as_posix());p.add_argument("--fold1-sha256",default=DEFAULT_FOLD1_SHA256);p.add_argument("--artifact",default=DEFAULT_ARTIFACT.as_posix());p.add_argument("--output-json",default=DEFAULT_JSON.as_posix());p.add_argument("--output-markdown",default=DEFAULT_MARKDOWN.as_posix());a=p.parse_args();root=repo_root()
    paths={"images":(root/a.images).resolve(),"metadata":(root/a.metadata).resolve(),"score":(root/a.score_cache).resolve(),"inner":(root/a.inner_cache).resolve(),"fold0":(root/a.fold0_cache).resolve(),"fold1":(root/a.fold1_cache).resolve()};expected={"images":a.images_sha256,"metadata":a.metadata_sha256,"score":a.score_sha256,"inner":a.inner_sha256,"fold0":a.fold0_sha256,"fold1":a.fold1_sha256}
    for n,d in expected.items():
        if sha256(paths[n])!=d:raise ValueError(f"Frozen {n} hash mismatch")
    images=np.load(paths["images"],mmap_mode="r",allow_pickle=False);parts=load_partitions(paths["metadata"],paths["score"],{n:paths[n] for n in ("inner","fold0","fold1")});values=combine_partitions(parts);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");raw_store={};model_store={};candidates=[]
    for beta in BETAS:
        seed_scores=[];model_store[beta]=[]
        for seed in SEEDS:
            raw,models=crossfit(images,values,beta,seed,device);seed_scores.append(raw);model_store[beta].extend(models)
        ensemble=np.mean(seed_scores,axis=0);raw_store[beta]=(ensemble,seed_scores)
        for blend in BLENDS:
            candidate=evaluate_candidate(values,ensemble,{"beta":beta},blend);candidate.update({"beta":beta,"blend_weight":blend});candidate_scores=blend_scores(values["current"],ensemble,blend);seed_checks=[]
            for raw in seed_scores:
                local=evaluate_candidate(values,raw,{"beta":beta},blend);seed_checks.append(bool(local["versus_current"]["delta"]["average_precision"]>0 and min(f["versus_current"]["delta"]["recall_at_fpr_0_0713"] for f in local["per_fold"].values())>=0))
            candidate["all_seed_stability_pass"]=all(seed_checks);candidate["stable"]=bool(candidate["stable"] and all(seed_checks));candidate["rank"][0]=int(candidate["stable"]);candidates.append(candidate)
    selected=max(candidates,key=lambda v:tuple(v["rank"]));ensemble,_=raw_store[selected["beta"]];scores=blend_scores(values["current"],ensemble,selected["blend_weight"]);selected["paired_group_bootstrap_ap_delta_vs_primary"]=ap_group_bootstrap(values["labels"],values["primary"],scores,values["groups"],replicates=10_000,seed=20261820);selected["paired_group_bootstrap_ap_delta_vs_current"]=ap_group_bootstrap(values["labels"],values["current"],scores,values["groups"],replicates=10_000,seed=20261821);passed=bool(selected["stable"] and selected["paired_group_bootstrap_ap_delta_vs_primary"]["lower"]>0 and selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"]>0)
    artifact=(root/a.artifact).resolve();digest=None
    if passed:
        artifact.parent.mkdir(parents=True,exist_ok=True);tmp=artifact.with_suffix(artifact.suffix+".tmp");torch.save({"schema_version":1,"kind":"mars_rex_spatial_crossfold_ensemble","beta":selected["beta"],"blend_weight":selected["blend_weight"],"seeds":SEEDS,"members":model_store[selected["beta"]]},tmp);os.replace(tmp,artifact);digest=sha256(artifact)
    report={"schema_version":1,"scope":"two-seed five-fold V-REx spatial development experiment; paper cache not loaded","generated_at_utc":datetime.now(timezone.utc).isoformat(),"betas":list(BETAS),"seeds":list(SEEDS),"blends":list(BLENDS),"selected":selected,"all_promotion_gates_pass":passed,"decision":"Freeze ten-member V-REx ensemble for label-free paper scoring." if passed else "Reject V-REx spatial model before paper scoring.","provenance":{**{f"{n}_sha256":d for n,d in expected.items()},"artifact_sha256":digest,"script_sha256":sha256(Path(__file__).resolve()),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),"numpy":np.__version__,"torch":torch.__version__}}
    write_json((root/a.output_json).resolve(),report);write_markdown((root/a.output_markdown).resolve(),report);print(json.dumps({"ok":passed,"beta":selected["beta"],"blend":selected["blend_weight"],"ap_delta":selected["versus_current"]["delta"]["average_precision"],"recall_delta":selected["versus_current"]["delta"]["recall_at_fpr_0_0713"],"ap_lower":selected["paired_group_bootstrap_ap_delta_vs_current"]["lower"],"artifact_sha256":digest},indent=2));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
