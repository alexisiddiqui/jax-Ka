"""Check four-leaf pruning on the bounded PyPKA site domain against ref lowering."""
import importlib.util
import json
import os
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxpb import api as API, titrate as T
import bench_jaxpb as BJ
src=Path(BJ.TP)/'opt3-sparse/src/jaxpb/titrate.py'
spec=importlib.util.spec_from_file_location('jaxpb._opt3_pruned',src)
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
r=dict(job=os.environ['SLURM_JOB_ID'],frozen=os.getcwd(),pruned_source=str(src),rows={})
rng=np.random.default_rng(81)
for mp in (64,128,256,512,1024,2048,4096):
 def ref(e,i):
    na,oa,nb,ob=i;m=jnp.arange(len(e));z=jnp.float32(0)
    return jnp.sum(jnp.where(m==na,e,z)-jnp.where(m==oa,e,z)+jnp.where(m==nb,e,z)-jnp.where(m==ob,e,z))
 rf=jax.jit(jax.vmap(ref))
 sf=jax.jit(lambda e,i:lax.map(lambda z:P._cpu_pair_energy(z[0],*z[1]),(e,i)))
 def dense(e,i):
    na,oa,nb,ob=i;m=jnp.arange(len(e));z=jnp.float32(0)
    v=jnp.where(m==na,e,z)-jnp.where(m==oa,e,z)+jnp.where(m==nb,e,z)-jnp.where(m==ob,e,z)
    w=len(e)
    while w>1:
        w//=2;v=lax.optimization_barrier(v[:w]+v[w:])
    return v[0]
 df=jax.jit(jax.vmap(dense))
 row=dict(cases=0,ref_vs_pruned=0,ref_vs_dense=0,pruned_vs_dense=0)
 for rep in range(32):
    N=73;e=rng.normal(0,20,(N,mp)).astype(np.float32);ix=[]
    for _ in range(N):
        a=int(rng.integers(mp-16+1));b=int(rng.integers(a+8,mp-8+1))
        aa=a+rng.integers(8,size=2);bb=b+rng.integers(8,size=2)
        i=np.concatenate([aa,bb] if rng.integers(2) else [bb,aa]);ix.append(i)
    ix=np.asarray(ix,np.int32)
    a=np.asarray(rf(e,ix));b=np.asarray(sf(e,ix));c=np.asarray(df(e,ix))
    row['cases']+=N;row['ref_vs_pruned']+=int(np.sum(a!=b));row['ref_vs_dense']+=int(np.sum(a!=c));row['pruned_vs_dense']+=int(np.sum(b!=c))
    if np.any(a!=b) and 'example' not in row:
        k=int(np.flatnonzero(a!=b)[0]);row['example']=dict(indices=ix[k].tolist(),values=e[k,ix[k]].tolist(),ref=float(a[k]),pruned=float(b[k]),dense=float(c[k]))
 r['rows'][str(mp)]=row;print(mp,row,flush=True)
 BJ.jdump(r,str(Path(BJ.TP)/'opt3/physical-tree.json'))
E=BJ.load_energies([BJ.TP+'/bench/gpu/energies_gpu.npz'])
r['max_nst']={s:int(E[s].nst.max()) for s in BJ.STATES}
r['passed']=all(x['ref_vs_pruned']==0 for x in r['rows'].values())
BJ.jdump(r,str(Path(BJ.TP)/'opt3/physical-tree.json'))
