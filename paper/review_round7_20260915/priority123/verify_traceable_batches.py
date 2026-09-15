"""Verify same-batch spatial/response records and derive core counts from them."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('--setting-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 root=a.setting_dir
 for line in (root/'SHA256SUMS.txt').read_text().splitlines():
  digest,name=line.split('  ',1);h=hashlib.sha256()
  with (root/name).open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  assert h.hexdigest()==digest,name
 meta=pd.read_csv(root/'metadata.csv');z=np.load(next(root.glob('*_cci_full_class_responses.npz')),allow_pickle=False)
 classes=z['category_ids'];cat={int(v):i for i,v in enumerate(classes)};rows=[];seen=[]
 for f in sorted((root/'traceable_batches').glob('*.npz')):
  b=np.load(f,allow_pickle=False);ids=b['sample_index'].astype(int);seen.extend(ids.tolist());m=b['masks'];raw=b['response_raw'];norm=b['response_norm'];sel=b['selection_all']
  assert np.isin(m,[0,1]).all() and (m.sum(1)==1).all() and (m.sum(2)>0).all()
  bbox=(m*b['inside'][:,None,:]).sum(2)/m.sum(2)
  assert np.allclose(bbox,b['bbox'],atol=1e-6)
  assert np.allclose(b['response_per_prompt'].mean(2),raw,atol=1e-5)
  assert np.allclose(raw/b['text_norm'][None,None,:],norm,atol=1e-5)
  assert np.array_equal(raw,z['response_raw'][ids]) and np.array_equal(norm,z['response_norm'][ids])
  for j,i in enumerate(ids):
   row=meta.iloc[i];assert str(row.image_id)==str(b['image_id'][j]);ti=cat[int(row.target_id)];r=int(np.argmax(sel[j,:,ti]));assert r==row.cci_region
   foil=np.arange(len(classes))!=ti;target=norm[j,:,ti];margin=target-norm[j][:,foil].max(1)
   mean=target-norm[j][:,foil].mean(1);A=bool(bbox[j,r]>=.5 and mean[r]>0);B=A and margin[r]<0
   present={int(v) for v in json.loads(row.annotated_category_ids)};absent=np.array([int(c) not in present and int(c)!=int(row.target_id) for c in classes]);assert absent.any()
   absent_margin=target[r]-norm[j,r,absent].max();N=int(foil.sum());k=int(absent.sum());outrank=int((norm[j,r,foil]>target[r]).sum());q=1-(math.comb(N-outrank,k)/math.comb(N,k) if k<=N-outrank else 0)
   feasible=sel[j,:,ti]>=sel[j,r,ti]-.02;sf=sel[j].copy();sf[:,ti]=-np.inf;objective=sel[j,:,ti]-sf.max(1);wf=int(np.argmax(np.where(feasible,objective,-np.inf)))
   passing=margin>=0;C0=B and not passing.any();C1=B and passing.any() and not (passing&feasible).any();C2=B and (passing&feasible).any() and not passing[wf];repair=B and passing[wf]
   assert not B or sum(map(int,[C0,C1,C2,repair]))==1
   rows.append(dict(sample_index=int(i),image_id=row.image_id,A=int(A),Aplus=int(A and target[r]>0),full_failure=int(margin[r]<0),absent_failure=int(absent_margin<0),random_failure=q,B=int(B),C0=int(C0),C1=int(C1),C2=int(C2),r=int(repair),cci_region=r,wf_region=wf,cci_bbox=bbox[j,r],wf_bbox=bbox[j,wf]))
 assert seen==list(range(len(meta))),'Missing, duplicate or reordered capture batches'
 a.out.mkdir(parents=True,exist_ok=True);d=pd.DataFrame(rows);d.to_csv(a.out/'same_record_diagnostics.csv',index=False);s=d[d.A==1]
 assert len(s)>0,'No screened images; report requires a nonempty A'
 result=dict(status='PASS',n=len(d),A=len(s),full_failure_pct=float(s.full_failure.mean()*100),absent_failure_pct=float(s.absent_failure.mean()*100),random_failure_pct=float(s.random_failure.mean()*100),**{v:int(d[v].sum()) for v in ['B','C0','C1','C2','r']},scope='Same-generation counts; confidence intervals and remaining paper tables require analysis before release.')
 (a.out/'same_record_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
