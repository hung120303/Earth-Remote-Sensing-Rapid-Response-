#!/usr/bin/env python3
"""Evaluate frozen adaptive Prithvi scores on the exact MARS-S2L paper contract."""
from __future__ import annotations
import argparse,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT/"EarthRemoteSensingRapidResponse",ROOT/"tools"):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from acquire_mars_metadata import repo_root,sha256  # noqa:E402
from diagnose_mars_scene_stacker_paper_cache import aligned_indices,triplet  # noqa:E402
from evaluate_mars_scene_gated_masks_paper_cache import gate_counts  # noqa:E402
from evaluate_mars_successor_paper_test import bootstrap_view,view_metrics  # noqa:E402
from evaluate_mars_xgboost_scene_head_paper_cache import operational_threshold  # noqa:E402

DEFAULT_SCORES=Path("outputs/mars_adaptive_prithvi_paper_scores.npz");DEFAULT_SCORES_SHA256="37aa4eb2e14bd7265df95a0cf55fc805e2a6e1ee8a2af7aa3e84a23165fe0059"
DEFAULT_RECEIPT=Path("reports/acquisition/mars_adaptive_prithvi_paper_scores.json");DEFAULT_RECEIPT_SHA256="1b76927c30b3614c0c08c4139f25fb4e66f92b517ac4a5fc76d3e4cc06cb5f9f"
DEFAULT_DIAGNOSTIC=Path("outputs/mars_paper_test_v3_diagnostic_cache.npz");DEFAULT_DIAGNOSTIC_SHA256="1624fddc0222f8ffc5137f557c7fc3e465d53b335c82cc8014711baa35bb94a1"
DEFAULT_DEVELOPMENT=Path("reports/experiments/mars_adaptive_prithvi_probe.json");DEFAULT_DEVELOPMENT_SHA256="3a0478d9abdb94a76815744cefcf7ac181765c20e2969a7a40f310f272a03a8b"
DEFAULT_GATE=Path("reports/experiments/mars_scene_gated_masks.json");DEFAULT_GATE_SHA256="c1e5a1497abebba80d42898a8165b30fd255ff252478a0ee1fd90fd32456a51c"
DEFAULT_JSON=Path("reports/experiments/mars_adaptive_prithvi_paper_posttest.json");DEFAULT_MARKDOWN=Path("reports/experiments/MARS_ADAPTIVE_PRITHVI_PAPER_POSTTEST.md")

def write_json(path:Path,value:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(tmp,path)
def write_markdown(path:Path,report:dict[str,Any])->None:
    lines=["# Adaptive Prithvi: exact MARS-S2L paper benchmark","","Transparent post-test replay; this is not an untouched confirmation cohort. Model scores came from a separately sealed label-free adaptation.","","| View | AP | AP delta (95% CI) | Recall delta (95% CI) | FPR delta | IoU delta (95% CI) | Gates |","|---|---:|---:|---:|---:|---:|---|"]
    for name,value in report["views"].items():
        m=value["metrics"];i=value["bootstrap"]["delta_intervals"]
        lines.append(f"| {name} | {m['candidate']['average_precision']:.5f} | {m['delta']['average_precision']:+.5f} ([{i['average_precision']['lower']:+.5f}, {i['average_precision']['upper']:+.5f}]) | {m['delta']['matched_fpr_recall']:+.5f} ([{i['matched_fpr_recall']['lower']:+.5f}, {i['matched_fpr_recall']['upper']:+.5f}]) | {m['delta']['fixed_false_positive_rate']:+.5f} | {m['delta']['pixel_iou']:+.5f} ([{i['pixel_iou']['lower']:+.5f}, {i['pixel_iou']['upper']:+.5f}]) | {'PASS' if value['passed'] else 'FAIL'} |")
    lines.extend(["",report["decision"]]);path.parent.mkdir(parents=True,exist_ok=True);path.write_text("\n".join(lines)+"\n",encoding="utf-8")
def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores",default=DEFAULT_SCORES.as_posix());p.add_argument("--scores-sha256",default=DEFAULT_SCORES_SHA256);p.add_argument("--receipt",default=DEFAULT_RECEIPT.as_posix());p.add_argument("--receipt-sha256",default=DEFAULT_RECEIPT_SHA256);p.add_argument("--diagnostic",default=DEFAULT_DIAGNOSTIC.as_posix());p.add_argument("--diagnostic-sha256",default=DEFAULT_DIAGNOSTIC_SHA256);p.add_argument("--development",default=DEFAULT_DEVELOPMENT.as_posix());p.add_argument("--development-sha256",default=DEFAULT_DEVELOPMENT_SHA256);p.add_argument("--gate",default=DEFAULT_GATE.as_posix());p.add_argument("--gate-sha256",default=DEFAULT_GATE_SHA256);p.add_argument("--replicates",type=int,default=10_000);p.add_argument("--output-json",default=DEFAULT_JSON.as_posix());p.add_argument("--output-markdown",default=DEFAULT_MARKDOWN.as_posix());a=p.parse_args();root=repo_root()
    paths={"scores":(root/a.scores).resolve(),"receipt":(root/a.receipt).resolve(),"diagnostic":(root/a.diagnostic).resolve(),"development":(root/a.development).resolve(),"gate":(root/a.gate).resolve()};expected={"scores":a.scores_sha256,"receipt":a.receipt_sha256,"diagnostic":a.diagnostic_sha256,"development":a.development_sha256,"gate":a.gate_sha256}
    for name,digest in expected.items():
        if sha256(paths[name])!=digest:raise ValueError(f"Frozen {name} hash mismatch")
    receipt=json.loads(paths["receipt"].read_text(encoding="utf-8"));development=json.loads(paths["development"].read_text(encoding="utf-8"));gate=json.loads(paths["gate"].read_text(encoding="utf-8"))
    if receipt["output_sha256"]!=a.scores_sha256 or receipt["labels_accessed"] is not False or development.get("all_promotion_gates_pass") is not True or gate.get("all_selection_and_confirmation_gates_pass") is not True:raise ValueError("Frozen promotion/label-free contract failed")
    with np.load(paths["scores"],allow_pickle=False) as cache:ids=cache["sample_ids"].astype(str);available_scores=cache["scores"].astype(np.float64)
    with np.load(paths["diagnostic"],allow_pickle=False) as cache:values={name:cache[name] for name in cache.files}
    indices=aligned_indices(values["aligned_sample_ids"],ids);candidate=values["candidate_scores"].astype(np.float64).copy();candidate[indices]=available_scores
    labels=values["labels"].astype(np.uint8);baseline=values["baseline_scores"].astype(np.float64);sites=values["sites"].astype(str);baseline_pixels=values["baseline_pixels"].astype(np.int64);cutoff=float(gate["selection"]["selected_cutoff"]);gated=gate_counts(values["candidate_pixels"].astype(np.int64),values["candidate_scores"].astype(np.float64),cutoff);threshold=operational_threshold(development)
    selections={"full":np.ones(labels.shape,dtype=bool),"test_only_sites":values["test_only"].astype(bool)};views={}
    for index,(name,rows) in enumerate(selections.items()):
        metrics=view_metrics(labels[rows],baseline[rows],candidate[rows],triplet(baseline_pixels[rows]),triplet(gated[rows]),threshold)
        bootstrap=bootstrap_view(labels=labels[rows],sites=sites[rows],baseline_scores=baseline[rows],candidate_scores=candidate[rows],baseline_predictions=baseline[rows]>0.5,candidate_predictions=candidate[rows]>threshold,baseline_pixels=triplet(baseline_pixels[rows]),candidate_pixels=triplet(gated[rows]),replicates=a.replicates,seed=20261570+index,confidence=0.95);i=bootstrap["delta_intervals"]
        checks={"ap_point_higher":metrics["delta"]["average_precision"]>0,"ap_lower_positive":i["average_precision"]["lower"]>0,"matched_recall_point_higher":metrics["delta"]["matched_fpr_recall"]>0,"matched_recall_lower_positive":i["matched_fpr_recall"]["lower"]>0,"fixed_fpr_upper_nonpositive":i["fixed_false_positive_rate"]["upper"]<=0,"pixel_iou_point_higher":metrics["delta"]["pixel_iou"]>0,"pixel_iou_lower_positive":i["pixel_iou"]["lower"]>0};views[name]={"metrics":metrics,"bootstrap":bootstrap,"checks":checks,"passed":all(checks.values())}
    passed=all(v["passed"] for v in views.values());report={"schema_version":1,"scope":"transparent post-test adaptive Prithvi replay on exact MARS-S2L comparator","generated_at_utc":datetime.now(timezone.utc).isoformat(),"architecture":{"scene_ranking":"target-normalized Prithvi CLS plus scene/context logistic complement","scene_threshold":threshold,"scene_threshold_rule":"maximum candidate threshold across five development OOF folds","mask_gate_score":"unchanged frozen v3 score","mask_gate_cutoff":cutoff},"available_rows":int(ids.size),"missing_rows_fallback_to_v3":int(labels.size-ids.size),"views":views,"all_exact_paper_gates_pass":passed,"decision":"All exact paper gates pass on both views; independent external confirmation remains required." if passed else "Reject adaptive Prithvi as final successor; at least one exact paper gate fails.","provenance":{**{f"{n}_sha256":d for n,d in expected.items()},"script_sha256":sha256(Path(__file__).resolve())}}
    write_json((root/a.output_json).resolve(),report);write_markdown((root/a.output_markdown).resolve(),report);print(json.dumps({"ok":passed,"views":{n:{"ap":v["metrics"]["candidate"]["average_precision"],"ap_lower":v["bootstrap"]["delta_intervals"]["average_precision"]["lower"],"recall_lower":v["bootstrap"]["delta_intervals"]["matched_fpr_recall"]["lower"],"iou_lower":v["bootstrap"]["delta_intervals"]["pixel_iou"]["lower"],"passed":v["passed"]} for n,v in views.items()}},indent=2));return 0 if passed else 2
if __name__=="__main__":raise SystemExit(main())
