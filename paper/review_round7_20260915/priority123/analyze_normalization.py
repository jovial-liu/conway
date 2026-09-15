"""Fixed-original-A normalization audit. Real NPZs required; no inferred responses."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

SETTINGS = ['coco_openai_b16','coco_openai_b32','voc2007_openai_b16','voc2007_openai_b32']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def metrics(raw,norm,target,region,A):
 n,k,c=raw.shape
 assert raw.shape==norm.shape and np.isfinite(raw).all() and np.isfinite(norm).all()
 assert target.shape==region.shape==A.shape==(n,) and A.any()
 rows=np.arange(n);out=[]
 for v in [raw,norm]:
  foil=v.copy();foil[rows,:,target]=-np.inf
  margin=v[rows,:,target]-foil.max(2)
  worst=foil[rows,region].argmax(1)
  out.append((margin,margin[rows,region]<0,worst))
 mr,fr,wr=out[0];mn,fn,wn=out[1]
 B=A&fn
 pct=lambda v:float(np.mean(v[A])*100)
 summary=dict(n_A=int(A.sum()),raw_failure_pct=pct(fr),normalized_failure_pct=pct(fn),sign_flip_pct=pct(fr!=fn),raw_pass_norm_fail_pct=pct(~fr&fn),raw_fail_norm_pass_pct=pct(fr&~fn),hardest_foil_agreement_pct=pct(wr==wn),fixed_normalized_B=int(B.sum()),raw_unrestricted_pass_pct_of_fixed_B=float(np.mean((mr>=0).any(1)[B])*100) if B.any() else None,normalized_unrestricted_pass_pct_of_fixed_B=float(np.mean((mn>=0).any(1)[B])*100) if B.any() else None)
 return summary, mn[rows,region],fn

def main():
 p=argparse.ArgumentParser();p.add_argument('--responses-root',type=Path,required=True);p.add_argument('--diagnostics',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 diag=pd.read_csv(a.diagnostics,low_memory=False);manifest=pd.read_csv(a.manifest);results=[];provenance={}
 for setting in SETTINGS:
  name=f'{setting}_cci_full_class_responses.npz';path=a.responses_root/setting/name
  expected=manifest[manifest.path.str.endswith('/'+name)]
  assert len(expected)==1,setting
  digest=sha(path);assert digest==expected.iloc[0].sha256,'SHA mismatch: '+setting
  with np.load(path,allow_pickle=False) as z:
   raw=z['response_raw'];norm=z['response_norm'];ids=z['image_id'].astype(str);targets=z['target_id'];categories=z['category_ids'];regions=z['cci_region'].astype(int)
   d=diag[diag.setting==setting].sort_values('sample_index')
   assert len(d)==len(ids) and np.array_equal(d.sample_index,np.arange(len(ids)))
   assert np.array_equal(d.image_id.astype(str).values,ids),'image order mismatch'
   assert np.array_equal(d.target_id,targets) and np.array_equal(d.cci_region,regions)
   index={int(v):i for i,v in enumerate(categories)};target=np.array([index[int(v)] for v in targets]);A=d.A.to_numpy()==1
   s,margin,failure=metrics(raw,norm,target,regions,A)
   assert np.allclose(margin,d.full_margin_recomputed_norm,atol=1e-5,rtol=0),'margin mismatch'
   assert np.array_equal(failure,d.full_failure.to_numpy()==1),'failure mismatch'
   results.append(dict(setting=setting,**s));provenance[setting]=dict(file=name,sha256=digest,rows=len(ids))
 a.out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(results).to_csv(a.out/'normalization_fixed_A.csv',index=False)
 (a.out/'provenance.json').write_text(json.dumps(dict(inputs=provenance,diagnostics_sha256=sha(a.diagnostics),scope='Original normalized A and CCI region fixed; oracle pass capacity uses original normalized B. No rescreening.',status='computed'),indent=2)+'\n')
 print(a.out/'normalization_fixed_A.csv')
if __name__=='__main__':main()
