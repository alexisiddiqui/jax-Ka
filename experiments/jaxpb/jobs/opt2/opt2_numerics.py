"""Quantify a rejected CPU variant against existing exact energy references."""
import json,os,sys
assert os.environ.get('SLURM_JOB_ID')
from pathlib import Path
root='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'
tag=sys.argv[1]
sys.path.insert(0,f'{root}/opt2-{tag}-20261122/src')
import numpy as np
import jax
from jaxpb import api as API
import bench_jaxpb as BJ
import prof2
import prof_geomaps as PG
from opt2 import digest
sid='4066edfcdb3ea8e3_A'
ref=prof2.ref_energies(sorted(str(x) for x in Path(root+'/bench/jaxcpu').glob('energies_cpu*.npz')))[sid]
baseline=json.load(open(root+'/opt1b/s5-cpu.json'))['passes'][0]['rows'][sid]['hashes']
capture={};solve=API.solve_pb

def cap(*a,**k):
 pb,tm=solve(*a,**k)
 capture.update(sweeps_bitwise=digest(pb.sweeps)==baseline['sweeps'], maxchange_bitwise=digest(pb.maxchange)==baseline['maxchange'],
                sweeps=np.asarray(pb.sweeps).tolist())
 return pb,tm
API.solve_pb=cap
r=PG.load(BJ.INPUTS,[sid])[0]
e=API.energies([r],API.Config(backend='xla'))[0]
got=dict(g=e.g,W=e.W,pkint=e.diag['pkint'],**{k:e.diag['pb'][k] for k in ('esolvM','esolvS','esolvM0')})
metrics={}
for k,a in got.items():
 if k not in ref:continue
 x=np.asarray(a);y=ref[k];d=np.abs(x-y);scale=float(np.max(np.abs(y)))
 metrics[k]=dict(max_abs=float(np.max(d)),relative_to_max_reference=float(np.max(d))/max(scale,1e-30),
                 max_elementwise_relative=float(np.max(d/np.maximum(np.abs(y),1e-30))))
res=dict(tag=tag,state=sid,job=os.environ['SLURM_JOB_ID'],measured_source=f'{root}/opt2-{tag}-20261122',
         reference='bench/jaxcpu, exact OPT1B energy reference; s5-cpu sweep/maxchange fingerprints',energies=metrics,**capture)
BJ.jdump(res,root+f'/opt2/rejected-numerics-{tag}.json')
print(json.dumps(res),flush=True)
