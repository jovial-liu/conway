"""Regenerate added statistics from recovered, frozen GitHub CSV records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
D=R/'evidence/recovered'
b=pd.read_csv(D/'coco_b16_baseline.csv');r=pd.read_csv(D/'coco_b16_repair_e002.csv')
assert len(b)==len(r)==27708 and b.image_id.is_unique and r.image_id.is_unique
assert np.array_equal(b.image_id,r.image_id)
assert np.allclose(b.aggregate_pmax,r.cci_pmax_agg,rtol=0,atol=1e-12)
assert np.allclose(b.bbox_precision,r.cci_bbox,rtol=0,atol=1e-12)
A=(b.bbox_precision>=.5)&(b.aggregate_pmean_all>0);F=b.aggregate_pmax<0
N=len(b); counts=np.array([(A&F).sum(),(A&~F).sum(),(~A&F).sum(),(~A&~F).sum()])
# Multinomial resampling of the four cells is exactly the image-bootstrap
# distribution of statistics that depend only on these two binary indicators.
rng=np.random.default_rng(1701);boot=rng.multinomial(N,counts/N,size=10000)
def ci(v):return np.quantile(v,[.025,.975]).tolist()
out={'N':N,'cells_A_F':counts.tolist(),'screen_pass_count':int(A.sum()),'failure_count':int(F.sum()),'joint_count':int((A&F).sum()),
 'screen_pass':[float(A.mean()),ci((boot[:,0]+boot[:,1])/N)],
 'failure':[float(F.mean()),ci((boot[:,0]+boot[:,2])/N)],
 'joint':[float((A&F).mean()),ci(boot[:,0]/N)],
 'failure_given_pass':[float((A&F).sum()/A.sum()),ci(boot[:,0]/(boot[:,0]+boot[:,1]))]}
sw=r.switched.astype(bool);assert int(sw.sum())==663
for c in ['delta_pmax_agg','delta_eval_target','delta_bbox']:assert np.allclose(r.loc[~sw,c],0)
q=r.loc[sw];rng=np.random.default_rng(1702);indices=rng.integers(0,len(q),size=(10000,len(q)))
out['switched']={'n':len(q),'sign_repairs':int(((q.cci_pmax_agg<0)&(q.tp_pmax_agg>=0)).sum()),'feasible_mean':float(r.selection_feasible_count.mean()),'multiple_feasible_count':int((r.selection_feasible_count>1).sum())}
out['switched']['switch_given_multiple']=len(q)/out['switched']['multiple_feasible_count']
out['feasible_histogram']={str(k):int(v) for k,v in r.selection_feasible_count.value_counts().sort_index().items()}
for c in ['delta_pmax_agg','delta_eval_target','delta_bbox']:
 v=q[c].to_numpy();out['switched'][c]={'mean':float(v.mean()),'ci':ci(v[indices].mean(axis=1)),'q10_q25_median_q75_q90':np.quantile(v,[.1,.25,.5,.75,.9]).tolist()}
out['switched']['gain_and_bbox_loss_count']=int(((q.delta_pmax_agg>0)&(q.delta_bbox<0)).sum())
out['failure_subset']={'target_raw_mean':float(b.loc[F,'eval_target_response_pm'].mean()),'target_raw_positive_rate':float((b.loc[F,'eval_target_response_pm']>0).mean())}
baseline=pd.read_csv(D/'12_feasible_baselines.csv');sparse=pd.read_csv(D/'14_sparse_repair_summary.csv')
rows=[]
for _,v in baseline[(baseline.method=='rtp')&(baseline.protocol=='full_foil')].iterrows():
 z=sparse[(sparse.dataset==v.dataset)&(sparse.model==v.model)].iloc[0]
 assert v.switch_count==z.switch_count
 assert abs(v.delta_pmax_agg_norm/v.switch_rate-z.among_switched_mean_delta_pmax)<1e-9
 rows.append({'dataset':v.dataset,'model':v.model,'n_images':int(v.n_images),'switch_count':int(v.switch_count),
 'switched_mean_delta_bbox':v.delta_bbox_precision/v.switch_rate,
 'switched_mean_delta_target_raw':v.delta_target_response_raw/v.switch_rate,
 'gain_fraction':z.among_switched_p_delta_positive,'sign_repair_fraction':z.among_switched_fail_to_pass,
 'note':'Conditional means algebraically derived from unrounded archived full-sample means and exact switch counts; no conditional CI available except separately computed COCO B/16.'})
out['four_setting_conditional']=rows
(R/'recovered_statistics.json').write_text(json.dumps(out,indent=2))
pd.DataFrame(rows).to_csv(R/'four_setting_conditional.csv',index=False)
print(json.dumps(out,indent=2))
