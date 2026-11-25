"""Audit the saved L3 midpoint changes, including finite-range censoring."""
import json, os
from pathlib import Path
import numpy as np

assert os.environ.get('SLURM_JOB_ID')
O=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/m7')
with np.load(O/'pka-arms.npz',allow_pickle=False) as z:
    ids=sorted(k.split('/')[1] for k in z.files if k.startswith('teacher/'))
    assert len(ids)==104
    result=dict(job=os.environ['SLURM_JOB_ID'],states=len(ids),seeds={})
    for seed in (1,2):
        delta=[];outliers=[];lost=gained=both_missing=0
        for sid in ids:
            d=z[f'delphi_seed{seed}/{sid}'];c=z[f'converged_seed{seed}/{sid}']
            fd=np.isfinite(d);fc=np.isfinite(c);shared=fd&fc
            lost+=int(np.sum(fd&~fc));gained+=int(np.sum(~fd&fc))
            both_missing+=int(np.sum(~fd&~fc))
            for site in np.flatnonzero(shared):
                change=float(c[site]-d[site]);delta.append(change)
                outliers.append(dict(state_id=sid,site_index=int(site),delphi_pka=float(d[site]),converged_pka=float(c[site]),delta=change))
        a=np.abs(delta)
        result['seeds'][str(seed)]=dict(shared_finite=len(a),lost_finite_midpoints=lost,gained_finite_midpoints=gained,both_missing=both_missing,mae=float(a.mean()),max=float(a.max()),signed_mean=float(np.mean(delta)),changes_above_pK={str(t):int(np.sum(a>t)) for t in (.01,.05,.1,.25,.5,1.)},largest_changes=sorted(outliers,key=lambda x:abs(x['delta']),reverse=True)[:10])
result['completed']=True
(O/'label-details.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
