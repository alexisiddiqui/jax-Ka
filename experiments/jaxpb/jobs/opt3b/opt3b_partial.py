"""Early review of completed OPT3B arms; Slurm-only, reuses frozen aggregation functions."""
import json
import os
from pathlib import Path
import sys
assert os.environ.get('SLURM_JOB_ID')
T='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'
sys.path.insert(0,T+'/opt3b-collect/jobs')
import numpy as np
from opt3b_results import diff, load, run, parity
import bench_jaxpb as BJ
O=Path(T)/'opt3b'
g,c=load('guard2'),load('cpu-final2')
z=np.load(O/'guard2-arrays.npz');zc=np.load(O/'cpu-final2-arrays.npz');teach=np.load(O/'teacher.npz')
r=dict(job=os.environ['SLURM_JOB_ID'],cpu_repeats=c['repeat_exact'],flat_compiles=c['compiles'],rebaseline={},cross={},teacher={})
for label in ('bench200k','l3'):
    ids=run(g,'new-'+label)['states']
    r['rebaseline'][label]=dict(change=diff(z,'new-'+label,'old-'+label,ids),noise=diff(z,'noise-'+label,'old-'+label,ids))
for label in ('50k','200k'):
    a,b,k=run(g,'pallas-'+label),run(g,'ref-'+label),run(c,label)
    r['cross'][label]=dict(pallas_gpu_ref=parity(a,b),pallas_cpu=parity(a,k),gpu_ref_cpu=parity(b,k),
        differences=dict(pallas_gpu_ref=diff(z,'pallas-'+label,'ref-'+label,k['states']),
                         pallas_cpu=diff(z,'pallas-'+label,label,k['states'],other=zc),
                         gpu_ref_cpu=diff(z,'ref-'+label,label,k['states'],other=zc)))
for label in ('old','new','noise'):
    e=[];mm=0
    for sid in teach.files:
        a=z[f'{label}-l3/{sid}/pka'];b=teach[sid];finite=np.isfinite(a)&np.isfinite(b)
        e.extend(np.abs(a[finite]-b[finite]));mm+=int(np.count_nonzero(np.isfinite(a)!=np.isfinite(b)))
    r['teacher'][label]=dict(n=len(e),mae=float(np.mean(e)),midpoint_mismatches=mm)
BJ.jdump(r,str(O/'partial.json'));print(json.dumps(r,indent=2),flush=True)
