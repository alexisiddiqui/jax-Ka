"""Read the missing reference sweep array in one CPU allocation, without changing its frozen code."""
import json,os,sys
assert os.environ.get('SLURM_JOB_ID')
T='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'
sys.path.insert(0,T+'/opt1b-s5-20261122/src')
sys.path.insert(0,os.path.join(os.getcwd(),'jobs'))
import numpy as np
from jaxpb import api as API
import bench_jaxpb as BJ
import prof_geomaps as PG
from opt2 import digest
sid='4066edfcdb3ea8e3_A'
r=PG.load(BJ.INPUTS,[sid])[0]
e=API.energies([r],API.Config(backend='xla'))[0]
sw=np.asarray(e.diag['pb']['sweeps'])
ref=json.load(open(T+'/opt1b/s5-cpu.json'))['passes'][0]['rows'][sid]['hashes']['sweeps']
assert digest(sw)==ref,'reference rerun did not match the saved 10a3ddc sweep fingerprint'
rows={}
for tag in ('s1d','s2'):
 data=json.load(open(T+'/opt2/rejected-numerics-'+tag+'.json'));trial=np.asarray(data['sweeps'])
 rows[tag]=dict(entries=int(sw.size),equal=int(np.sum(sw==trial)),agreement=float(np.mean(sw==trial)),
                differences=[dict(index=ix,reference=int(sw[tuple(ix)]),trial=int(trial[tuple(ix)])) for ix in np.argwhere(sw!=trial).tolist()])
BJ.jdump(dict(job=os.environ['SLURM_JOB_ID'],reference_frozen=T+'/opt1b-s5-20261122',state=sid,reference_fingerprint_matched=True,rows=rows),T+'/opt2/rejected-sweep-localization.json')
print(rows,flush=True)
