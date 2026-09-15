"""Build paper tables from the pinned traceable release; no model inference."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
E=ROOT/'experiments/priority123_20260915';T=E/'traceable_results';S=['coco_openai_b16','coco_openai_b32','voc2007_openai_b16','voc2007_openai_b32'];L=['COCO B/16','COCO B/32','VOC B/16','VOC B/32']
restricted=pd.read_csv(T/'restricted_foil_failure_rates.csv');matched=pd.read_csv(T/'matched_cardinality_control.csv');budget=pd.read_csv(T/'budget_scan.csv');normal=pd.read_csv(E/'normalization/normalization_fixed_A.csv');audit=pd.read_csv(E/'normalization/normalization_per_image_audit.csv');paired=pd.read_csv(T/'paired_difference_ci_10000.csv')
files={k:[] for k in ['restricted','oracle','all_budget','normalization','endpoint_cost','threshold']};report=[]
for z,(setting,label) in enumerate(zip(S,L)):
 p=T/'per_image'/f'{setting}.csv';d=pd.read_csv(p);A=d.A.to_numpy()==1;B=d.B.to_numpy()==1
 arrays=lambda key:np.array([json.loads(x) for x in d[key]],dtype=float)
 margin=arrays('candidate_eval_margin_norm');bbox=arrays('candidate_bbox_precision');mean=arrays('candidate_eval_pmean_norm');target=arrays('candidate_eval_target_drop_norm');raw=arrays('candidate_eval_target_drop_raw');sel=arrays('candidate_selection_target');maxfoil=arrays('candidate_selection_max_foil');idx=np.arange(len(d));cci=d.cci_region.to_numpy(int);wf=d.wf_region.to_numpy(int);feasible=arrays('feasible_mask').astype(bool)
 assert np.array_equal(A,(bbox[idx,cci]>=.5)&(mean[idx,cci]>0));assert np.array_equal(B,A&(margin[idx,cci]<0))
 assert np.array_equal(wf,np.where(feasible,sel-maxfoil,-np.inf).argmax(1))
 passing=margin>=0;u=passing.any(1);f=(passing&feasible).any(1);repair=B&passing[idx,wf];joint=passing&(bbox>=.5)&(mean>0)&(target>0)
 counts=[int(B.sum()),int((B&~u).sum()),int((B&u&~f).sum()),int((B&f&~passing[idx,wf]).sum()),int(repair.sum()),int((repair&joint[idx,wf]).sum())]
 assert counts==[int(d[k].sum()) for k in ['B','C0','C1','C2','repair','joint_J']]
 v=json.loads((E/'verification'/setting/'same_record_checks.json').read_text());assert v['status']=='PASS' and v['A']==int(A.sum()) and v['B']==int(B.sum())
 rr=restricted[(restricted.dataset_model==setting)&(restricted.subset=='A')].iloc[0];mm=matched[(matched.dataset_model==setting)&(matched.subset=='A')].iloc[0]
 assert rr.n==A.sum() and rr.full_failure_count==B.sum();assert np.isclose(d.loc[A,'exact_matched_random_failure_probability'].mean(),mm.exact_matched_random_failure_rate)
 files['restricted'].append(f"{label} & {100*rr.Pr_F_full:.2f} & {100*rr.Pr_F_absent:.2f} & {100*mm.exact_matched_random_failure_rate:.2f} & \\estci{{{100*mm.absent_minus_matched_random:+.2f}}}{{[{100*mm.absent_minus_matched_random_CI_lo:+.2f},{100*mm.absent_minus_matched_random_CI_hi:+.2f}]}}\\\\")
 files['oracle'].append(label+' & '+' & '.join(map(str,counts[:5]))+r'\\')
 for eps in [.02,.20]:
  b=budget[(budget.dataset_model==setting)&np.isclose(budget.epsilon,eps)].iloc[0]
  files['all_budget'].append(f"{'C' if z<2 else 'V'}{16 if z%2==0 else 32} & {eps:.2f} & {100*b.multi_feasible_pct_all_images:.2f} & {100*b.epsilon_oracle_pct_B:.2f} & {100*b.epsilon_wf_repair_pct_B:.2f} & {b.all_image_raw_target_change:+.4f} & {b.all_image_bbox_change_pp:+.2f}\\\\")
 n=normal[normal.setting==setting].iloc[0];a=audit[(audit.setting==setting)&(audit.A==1)];assert len(a)==n.n_A
 for col,out in [('raw_failure','raw_mean_failure_pct'),('normalized_failure','normalized_mean_failure_pct'),('hardest_foil_agreement','hardest_foil_agreement_pct')]:assert np.isclose(a[col].mean()*100,n[out])
 assert np.isclose((a.raw_failure!=a.normalized_failure).mean()*100,n.sign_flip_pct)
 files['normalization'].append(f"{label} & {int(n.n_A):,} & {n.raw_mean_failure_pct:.3f} & {n.normalized_mean_failure_pct:.3f} & {n.sign_flip_pct:.3f}\\\\")
 costs=[]
 for j,x in enumerate([(raw[idx,wf]-raw[idx,cci])[repair],100*(bbox[idx,wf]-bbox[idx,cci])[repair]]):
  rng=np.random.default_rng(8800+10*z+j);ci=np.percentile(x[rng.integers(len(x),size=(10000,len(x)))].mean(1),[2.5,97.5]);dec=3 if j==0 else 2
  costs.append(f'\\estci{{{x.mean():.{dec}f}}}{{[{ci[0]:.{dec}f},{ci[1]:.{dec}f}]}}')
 files['endpoint_cost'].append(label+f' & {counts[4]} & {counts[5]} & '+' & '.join(costs)+r'\\')
 files['threshold'].append(label+' & '+' & '.join(f'{100*np.mean(margin[A,cci[A]] < -delta):.2f}' for delta in [0,.01,.1,.5])+r'\\')
 for comparison,other in [('wf_minus_mean','mean'),('wf_minus_max_0_1','max_0_1')]:
  rows=paired[(paired.dataset_model==setting)&(paired.comparison==comparison)]
  if len(rows)==0:
   assert comparison=='wf_minus_max_0_1';rows=paired[(paired.dataset_model==setting)&(paired.comparison=='wf_minus_max01')]
  for _,row in rows.iterrows():assert np.isclose((d['wf_'+row.metric]-d[other+'_'+row.metric]).mean(),row.estimate,atol=1e-10)
 report.append(dict(setting=setting,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),records=len(d),A=int(A.sum()),B_C0_C1_C2_r_j=counts,unrestricted_pct=100*(B&u).sum()/B.sum(),feasible_pct=100*(B&f).sum()/B.sum(),joint_oracle=int((B&(joint&feasible).any(1)).sum()),normalization_old_A=int(n.n_A)))
for key,rows in files.items():(OUT/'source/tables'/f'{key}_rows.tex').write_text('\n'.join(rows)+'\n')
(OUT/'generated/evidence_checks.json').write_text(json.dumps(dict(experiment_commit='7ee2b6080aa6752790e17c910c5c8a3111235d4d',release_commit='78387d76f271d84b508373c603a5eff75c09299b',scope='CSV-derived counts and normalization audit independently checked. Physical masks/tensors not downloaded; same-batch PASS reports are supplied by experiment runner. Table 5 conditional bootstrap: 10000 image resamples, seeds 8800+10*setting+metric.',settings=report),indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42,
 'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
 'axes.spines.right':False,'axes.spines.left':False,'axes.linewidth':.6,
 'mathtext.fontset':'dejavuserif'})
fig,ax=plt.subplots(figsize=(3.3858,2.65))
fig.subplots_adjust(left=.245,right=.975,top=.80,bottom=.21)
y=np.array([0.,1.,2.4,3.4])
u=np.array([r['unrestricted_pct'] for r in report])
f=np.array([r['feasible_pct'] for r in report])
blue='#255b78';orange='#a34c22'
ax.axhspan(1.7,4.0,color='#f3f5f6',zorder=0)
for yy,a,b in zip(y,u,f):
 ax.plot([b,a],[yy,yy],color='#9ba7ad',lw=1.4,zorder=2)
 ax.plot(a,yy,'o',color=blue,ms=5.3,zorder=3)
 ax.plot(b,yy,'D',color=orange,ms=4.5,zorder=3)
 ax.annotate(f'{a:.2f}',(a,yy),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9,color=blue)
 ax.annotate(f'{b:.2f}',(b,yy),xytext=(4,-12),textcoords='offset points',ha='left',fontsize=9,color=orange)
ax.set(yticks=y,yticklabels=['COCO B/16','COCO B/32','VOC B/16','VOC B/32'],
 xlim=(-.7,36),ylim=(4.05,-.65),xticks=[0,10,20,30],xlabel=r'Sign-repair capacity (% of $B$)')
ax.tick_params(axis='y',length=0,pad=5,labelsize=9)
ax.tick_params(axis='x',length=3,color='#899298',labelsize=9)
ax.grid(axis='x',color='#dce1e4',lw=.5);ax.set_axisbelow(True)
from matplotlib.lines import Line2D
handles=[Line2D([],[],marker='o',ls='none',color=blue,ms=5,label='Unrestricted (8 candidates)'),
 Line2D([],[],marker='D',ls='none',color=orange,ms=4.5,label=r'Feasible ($\epsilon=0.02$)')]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.02,1.015),frameon=False,
 fontsize=9,handlelength=1.1,handletextpad=.5,labelspacing=.4)
for ext in ['pdf','svg','png']:
 fig.savefig(OUT/f'source/figures/figure2_repair_capacity.{ext}',dpi=300,
  metadata={'Date':None} if ext=='svg' else None)
svg=OUT/'source/figures/figure2_repair_capacity.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
