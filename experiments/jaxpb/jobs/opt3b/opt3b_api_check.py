"""Validate existing public-API shards in both stored and reader orientations. Slurm only."""
import hashlib
import json
import os
from pathlib import Path
import sys
assert os.environ.get('SLURM_JOB_ID')
TP='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'
sys.path.insert(0,TP+'/opt3b-final/src')
sys.path.insert(0,TP+'/opt3b-collect3/jobs')
import numpy as np
from jaxpb import api as API, titrate as T
import bench_jaxpb as BJ
O=Path(TP)/'opt3b'
f=json.load(open(O/'fresh-final2.json'))
E=BJ.load_energies([TP+'/bench/gpu/energies_gpu.npz'])
def digest(x):
    a=np.ascontiguousarray(x)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()
r=dict(job=os.environ['SLURM_JOB_ID'],source=str(Path(__file__)),original_job=f['job'],checks={},details={},
       correction='load_shard returns curves [73,n]; writer ragged arrays are [n,73]. Compare the same orientation.',
       exact_count_recovery='Stored f64 curves=integer prot/sweeps, with prot<=sweeps<2**24. Rounding curves*sweeps uniquely recovers prot (error far below 0.5). A successful MC diagnostic implies tot==sweeps, enforced by the driver.')
for label,sw in [('50k',50000),('200k',200000)]:
    ref=next(x for x in f['runs'] if x['label']==label)
    loaded=API.load_shard(str(O/('fresh-'+label+'-labels.npz')),full_W=False)
    got={};fields={};diag=True
    for d in loaded:
        sid=d['alias'];e=E[sid];curves=d['curves'];prot=np.rint(curves*sw).astype(np.int64)
        assert (prot>=0).all() and (prot<=sw).all()
        mp,promoted=T.fit_bucket(len(e.g),len(e.nst),len(e.pairs))
        diag=diag and d['status']=='ok' and d['diag']==dict(method='mc',flag=None,Mp=mp,promoted=promoted)
        got[sid]={k:digest(v) for k,v in dict(prot=prot,tot=np.full_like(prot,sw),curves=curves,pka=d['pka']).items()}
        fields[sid]={k:digest(d[k].T if k=='curves' else d[k]) for k in API.RAGGED}
    detail=dict(outputs=got==ref['outputs'],stored_ragged_fields=fields==ref['shard_fields'],diagnostics_and_status=diag)
    r['details'][label]=detail;r['checks'][label]=all(detail.values())
r['passed']=all(r['checks'].values());BJ.jdump(r,str(O/'api-check.json'));print(json.dumps(r,indent=2),flush=True)
assert r['passed']
