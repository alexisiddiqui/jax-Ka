"""Test fixed array-reduction tiles without changing the chain batching."""
import json
import os
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import bench_jaxpb as BJ
r=dict(job=os.environ['SLURM_JOB_ID'],frozen=os.getcwd(),rows={})
rng=np.random.default_rng(55)
for mp in (64,128,256,512,1024):
    N=73
    E=rng.normal(0,20,(N,mp)).astype(np.float32)
    ix=np.array([rng.choice(mp,4,replace=False) for _ in range(N)],np.int32)
    def vals(e,i):
        na,oa,nb,ob=i;m=jnp.arange(len(e));z=jnp.float32(0)
        return jnp.where(m==na,e,z)-jnp.where(m==oa,e,z)+jnp.where(m==nb,e,z)-jnp.where(m==ob,e,z)
    ref=np.asarray(jax.jit(jax.vmap(lambda e,i:jnp.sum(vals(e,i))))(E,ix))
    row={}
    for size in (1,2,4,8,16,32,73):
        def make(size):
            def fn(e,i):
                v=lax.optimization_barrier(jnp.broadcast_to(vals(e,i),(1,size,len(e))))
                return lax.optimization_barrier(jnp.sum(v,axis=2))[0,0]
            return fn
        fn=make(size)
        a=np.asarray(jax.jit(lambda e,i:lax.map(lambda z:fn(*z),(e,i)))(E,ix))
        row[str(size)]=int(np.sum(ref!=a))
    r['rows'][str(mp)]=row;print(mp,row,flush=True)
BJ.jdump(r,str(Path(BJ.TP)/'opt3/tile.json'))
