"""Isolate the rounding risk of reassociating a pair's four energy terms."""
import json
import os
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import jax
import jax.numpy as jnp
import numpy as np
import bench_jaxpb as BJ

rng=np.random.default_rng(81)
W=1024; N=4096
E=rng.normal(0,20,(N,W)).astype(np.float32)
ix=np.array([rng.choice(W,4,replace=False) for _ in range(N)],np.int32)
@jax.jit
def calc(E,ix):
    def one(e,i):
        na,oa,nb,ob=i
        x=jnp.arange(W)
        ref=jnp.sum(jnp.where(x==na,e,0)-jnp.where(x==oa,e,0)+jnp.where(x==nb,e,0)-jnp.where(x==ob,e,0))
        direct=(e[na]-e[oa])+(e[nb]-e[ob])
        return ref,direct
    return jax.vmap(one)(E,ix)
a,b=map(np.asarray,calc(E,ix));bad=np.flatnonzero(a!=b);k=int(bad[0])
r=dict(job=os.environ['SLURM_JOB_ID'],frozen=os.getcwd(),pairs=N,differences=len(bad),example=dict(indices=ix[k].tolist(),values=E[k,ix[k]].tolist(),reference=float(a[k]),reassociated=float(b[k]),reference_bits=int(a[k].view(np.uint32)),reassociated_bits=int(b[k].view(np.uint32))))
BJ.jdump(r,str(Path(BJ.TP)/'opt3/pair-rounding.json'));print(json.dumps(r),flush=True)
