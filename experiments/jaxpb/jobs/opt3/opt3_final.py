"""Fresh-process public auto/API validation on the final source, after CPU gate 1."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch
assert os.environ.get('SLURM_JOB_ID')
import numpy as np
import jax
from jaxpb import api as API, titrate as T
from jaxpb.compile_budget import CompileCounter
import bench_jaxpb as BJ
from opt3_gate import CPU5, digest

O=Path(BJ.TP)/'opt3'
old=json.load(open(O/'baseline12.json'))
prior=json.load(open(O/'network12.json'))
assert old['passed'] and prior['passed']
assert T.resolve_backend()=='cpu' and API.mc_backend(API.Config())=='cpu'
assert API.mc_backend(API.Config(backend='xla'))=='ref'
with patch.object(T,'pallas_available',return_value=True):
    assert T.resolve_backend()=='pallas' and API.mc_backend(API.Config())=='pallas'
with patch.object(T,'pallas_available',return_value=False), patch.object(jax,'default_backend',return_value='gpu'):
    assert T.resolve_backend()=='ref' and API.mc_backend(API.Config())=='ref'
E=BJ.load_energies([BJ.TP+'/bench/gpu/energies_gpu.npz'])
r=dict(job=os.environ['SLURM_JOB_ID'],frozen=os.getcwd(),env=BJ.env_info(),selection=True,runs=[])
with CompileCounter() as cc:
    for label,ids,sw in [('50k',BJ.STATES,50000),('200k',CPU5,200000)]:
        n0=cc.n_compiled;t=time.perf_counter()
        out=API.titrate([E[s] for s in ids],API.Config(mc_sweeps=sw),chain_ids=[API.chain_id_of(s) for s in ids])
        rows=[]
        got={s:{k:digest(a[k]) for k in ('prot','tot','curves','pka')} for s,a in zip(ids,out)}
        for sid,a in zip(ids,out):
            rows.append(dict(state_id=sid,alias=sid,input='saved bench energies',status='ok',
                             arrays=API._state_arrays(len(E[sid].nst),None,a),
                             diag={k:a[k] for k in ('method','flag','Mp','promoted')}))
        path=O/('final-auto-'+label+'-labels.npz')
        API.write_shard(str(path),rows,dict(backend='auto',resolved='cpu'),{})
        fields={d['state_id']:{**{k:digest(d[k]) for k in API.RAGGED},'diag':d['diag'],'status':d['status']} for d in API.load_shard(str(path),full_W=False)}
        checks={}
        for name,ref in [('baseline',old),('prior_process',prior)]:
            other=next(x for x in ref['runs'] if x['label']==label)
            checks[name]=got==other['outputs'] and fields==other['shard_fields']
        row=dict(sweeps=sw,states=ids,checks=checks,compiles=cc.n_compiled-n0,seconds=time.perf_counter()-t,
                 timings=T.titrate.last_timing,shard=str(path))
        r['runs'].append(row)
        BJ.jdump(r,str(O/'final.json'));print('final',label,row,flush=True)
    r['production_compiles']=cc.summary()
sys.path.insert(0,str(Path.cwd()/'tests'))
import test_opt3, test_titrate
r['new_tests']=test_opt3.test_cpu_ref()
r['existing_tests']=test_titrate.local()
work=Path('/tmp')/('jaxpb-opt3-api-'+os.environ['SLURM_JOB_ID'])
try:
    subprocess.run([sys.executable,'tests/test_api.py',str(work)],check=True)
    r['api_tests']=True
finally:
    shutil.rmtree(work,ignore_errors=True)
r['passed']=all(all(x['checks'].values()) for x in r['runs']) and r['new_tests']['passed'] and r['existing_tests']['passed'] and r['api_tests'] and r['runs'][0]['compiles']==5 and r['runs'][1]['compiles']==0
BJ.jdump(r,str(O/'final.json'));assert r['passed'];print('FINAL PASS',flush=True)
