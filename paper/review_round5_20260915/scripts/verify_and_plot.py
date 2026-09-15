"""Verify retained manuscript counts against existing CSVs; no model inference."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
SETTINGS=['coco_openai_b16','coco_openai_b32','voc2007_openai_b16','voc2007_openai_b32']
EXPECTED=[[11410,10697,695,0,18,12],[11561,10655,875,1,30,25],[511,371,135,0,5,3],[505,347,155,1,2,1]]
report=[];case=None
for setting,expected in zip(SETTINGS,EXPECTED):
 path=ROOT/'experiments/local_rerun_2026-09-13/rerun_workspace/runs'/setting/'per_image.csv'
 rows=list(csv.DictReader(path.open()))
 arr=lambda key:np.asarray([json.loads(r[key]) for r in rows],dtype=np.float32)
 target=arr('candidate_selection_target');foil=arr('candidate_selection_max_foil');margin=arr('candidate_eval_margin_norm');mean=arr('candidate_eval_pmean_norm');positive=arr('candidate_eval_target_drop_norm');raw=arr('candidate_eval_target_drop_raw');bbox=arr('candidate_bbox_precision')
 idx=np.arange(len(rows));cci=target.argmax(1)
 feasible=target>=target[idx,cci,None]-np.float32(.02)
 wf=np.where(feasible,target-foil,-np.inf).argmax(1)
 assert np.array_equal(cci,np.array([int(r['cci_region']) for r in rows]))
 assert np.array_equal(wf,np.array([int(r['wf_region']) for r in rows]))
 A=(bbox[idx,cci]>=.5)&(mean[idx,cci]>0);B=A&(margin[idx,cci]<0)
 passing=margin>=0;unrestricted=passing.any(1);feasiblepass=(passing&feasible).any(1);repair=B&passing[idx,wf]
 J=passing&(bbox>=.5)&(mean>0)&(positive>0)
 counts=[int(B.sum()),int((B&~unrestricted).sum()),int((B&unrestricted&~feasiblepass).sum()),int((B&feasiblepass&~passing[idx,wf]).sum()),int(repair.sum()),int((repair&J[idx,wf]).sum())]
 assert counts==expected,(setting,counts,expected)
 frontier=[]
 for eps in [.01,.02,.05,.1,.2]:
  feas=target>=target[idx,cci,None]-np.float32(eps);choice=np.where(feas,target-foil,-np.inf).argmax(1)
  frontier.append(dict(epsilon=eps,multi_pct=float((feas.sum(1)>1).mean()*100),oracle_pct=float(((passing&feas).any(1)&B).sum()/B.sum()*100),repair_pct=float((B&passing[idx,choice]).sum()/B.sum()*100),raw_target_change=float((raw[idx,choice].astype(float)-raw[idx,cci]).mean()),bbox_change_pp=float((bbox[idx,choice].astype(float)-bbox[idx,cci]).mean()*100)))
 result=dict(setting=setting,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=len(rows),A=int(A.sum()),B_C0_C1_C2_r_j=counts,unrestricted_pct=float((unrestricted&B).sum()/B.sum()*100),feasible_pct=float((feasiblepass&B).sum()/B.sum()*100),repair_raw_target_change=float((raw[idx,wf].astype(float)-raw[idx,cci])[repair].mean()),repair_bbox_change_pp=float((bbox[idx,wf].astype(float)-bbox[idx,cci])[repair].mean()*100),frontier=frontier)
 report.append(result)
 if setting=='coco_openai_b16':
  eligible=repair&(positive[idx,cci]>0); inds=np.flatnonzero(eligible);ordered=inds[np.argsort(margin[inds,cci[inds]],kind='stable')];chosen=int(ordered[len(ordered)//2]);assert str(rows[chosen]['image_id'])=='254807'
  case={'image_id':rows[chosen]['image_id'],'selection_pool':int(eligible.sum()),'endpoints':{name:[float(v[chosen,cci[chosen]]),float(v[chosen,wf[chosen]])] for name,v in [('worst_foil_margin',margin),('mean_foil_margin',mean),('raw_target_drop',raw),('bbox_precision',bbox)]}}
(OUT/'generated/evidence_checks.json').write_text(json.dumps({'scope':'Reanalysis of existing CSVs: strategy choices, counts, means, tolerance rows and case selection. No inference or new bootstrap confidence intervals.','settings':report,'case':case},indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(3.3858,2.4));fig.subplots_adjust(left=.30,right=.975,top=.77,bottom=.23)
y=np.arange(4);u=np.array([r['unrestricted_pct'] for r in report]);f=np.array([r['feasible_pct'] for r in report]);h=.27
ax.barh(y-h/2,u,height=h,color='#4e748b',label='Unrestricted (all 8)')
ax.barh(y+h/2,f,height=h,color='#b16b4d',label='Feasible (epsilon = .02)')
for j,(a,b) in enumerate(zip(u,f)):
 ax.text(a+.55,j-h/2,f'{a:.2f}',va='center',fontsize=9)
 ax.text(b+.55,j+h/2,f'{b:.2f}',va='center',fontsize=9,color='#75422f')
ax.set(yticks=y,yticklabels=['COCO B/16','COCO B/32','VOC B/16','VOC B/32'],xlim=(0,39),xticks=[0,10,20,30],xlabel='Sign-repair capacity (% of B)')
ax.invert_yaxis();ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=4);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.03,1.01),frameon=False,fontsize=9,handlelength=1.5,labelspacing=.3)
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'source/figures/figure2_repair_capacity.{ext}',dpi=200)
print(json.dumps({'verified_records':sum(r['records'] for r in report),'case':case,'counts':[r['B_C0_C1_C2_r_j'] for r in report]},indent=2))

svg = OUT/'source/figures/figure2_repair_capacity.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
