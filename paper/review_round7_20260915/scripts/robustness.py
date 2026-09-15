"""Post-hoc descriptive analyses of stored candidates; no new CLIP inference."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parents[1]
settings=['coco_openai_b16','coco_openai_b32','voc2007_openai_b16','voc2007_openai_b32']
labels=['COCO B/16','COCO B/32','VOC B/16','VOC B/32'];results=[];tr=[];br=[]
for key,label in zip(settings,labels):
 p=root/'experiments/local_rerun_2026-09-13/rerun_workspace/runs'/key/'per_image.csv'
 rows=list(csv.DictReader(p.open()));arr=lambda k:np.array([json.loads(r[k]) for r in rows],dtype=np.float32)
 target=arr('candidate_selection_target');foil=arr('candidate_selection_max_foil');margin=arr('candidate_eval_margin_norm');mean=arr('candidate_eval_pmean_norm');bbox=arr('candidate_bbox_precision');raw=arr('candidate_eval_target_drop_raw')
 idx=np.arange(len(rows));cci=target.argmax(1);A=(bbox[idx,cci]>=.5)&(mean[idx,cci]>0);B=A&(margin[idx,cci]<0)
 ds=[0,.01,.1,.5];rates=[float((margin[idx,cci][A]<-d).mean()*100) for d in ds]
 tr.append(label+' & '+' & '.join(f'{x:.2f}' for x in rates)+r'\\')
 budgets=[]
 for eps in [.01,.02,.05,.1,.2]:
  feasible=target>=target[idx,cci,None]-np.float32(eps);wf=np.where(feasible,target-foil,-np.inf).argmax(1)
  o=float((((margin>=0)&feasible).any(1)&B).sum()/B.sum()*100);r=float((B&(margin[idx,wf]>=0)).sum()/B.sum()*100)
  multi=float((feasible.sum(1)>1).mean()*100);dt=float((raw[idx,wf].astype(float)-raw[idx,cci]).mean());db=float((bbox[idx,wf].astype(float)-bbox[idx,cci]).mean()*100)
  budgets.append(dict(epsilon=eps,multi_pct=multi,oracle_pct=o,repair_pct=r,raw_target_change=dt,bbox_change_pp=db))
  if eps in [.02,.2]:br.append({'COCO B/16':'C16','COCO B/32':'C32','VOC B/16':'V16','VOC B/32':'V32'}[label]+f' & {eps:.2f} & {multi:.2f} & {o:.2f} & {r:.2f} & {dt:+.4f} & {db:+.2f}'+r'\\')
 results.append(dict(setting=key,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),n=int(len(rows)),A=int(A.sum()),B=int(B.sum()),thresholds=ds,failure_pct=rates,budgets=budgets))
(out/'source/tables/threshold_rows.tex').write_text('\n'.join(tr)+'\n');(out/'source/tables/all_budget_rows.tex').write_text('\n'.join(br)+'\n')
(out/'generated/robustness_checks.json').write_text(json.dumps({'scope':'Post-hoc descriptive fixed-A threshold sweep and fixed-B budget sweep; normalized logit units, no interval estimates or new model inference. No raw-vs-normalized full-foil comparison is possible from these summaries.','settings':results},indent=2)+'\n')
print('\n'.join(tr));print('\n'.join(br))
