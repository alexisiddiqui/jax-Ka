"""Measured solver selection on a real one-site PB problem, Slurm only."""
import os,time,json
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import jax,numpy as np
from jaxpb import api as A,pb as B,pcg as P,records as R
import bench_jaxpb as BJ,prof_geomaps as PG
O=Path(BJ.TP)/'m7'; r=PG.load(BJ.INPUTS,['4066edfcdb3ea8e3_A'])[0]
original=P.run_pool; capture=[]
def pool(*args,**kw):
    if not capture:
        capture.append(jax.tree.map(lambda x:np.asarray(x),args))
    return original(*args,**kw)
P.run_pool=pool
out=B.solve_state(r,B.PBConfig(pb_mode='converged',backend='xla',B=1,specrad='estimate'),sites=[0])
args=capture[0]; results={}; potentials={}
for method in ('pcg','pcg_restart50','sor'):
    times=[]
    for rep in range(2):
        t=time.perf_counter(); stats=[]; phi,it,res=original(*args,stats=stats,method=method); jax.block_until_ready(phi)
        times.append(time.perf_counter()-t)
    results[method]=dict(seconds=times,iterations=it.tolist(),residual=res.tolist(),stats=stats);potentials[method]=np.asarray(phi)
BJ.jdump(dict(env=BJ.env_info(),state=r.state_id,site=0,results=results,flags=out.flags,standard_vs_restart50_max_dphi=float(np.max(np.abs(potentials['pcg']-potentials['pcg_restart50'])))),str(O/('probe-'+jax.default_backend()+'.json')))
print(results,flush=True)
