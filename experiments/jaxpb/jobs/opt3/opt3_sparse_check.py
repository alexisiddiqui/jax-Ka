"""Validate sparse bit-reversed pair trees, then the two 200k failure states."""
import json
import os
import time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import jax
import jax.numpy as jnp
import numpy as np
from jaxpb import api as API, titrate as T
import bench_jaxpb as BJ
from opt3_gate import digest
O=Path(BJ.TP)/'opt3'
r=dict(job=os.environ['SLURM_JOB_ID'],frozen=os.getcwd(),math={},states={})
rng=np.random.default_rng(31)
for mp in (64,128,256,512,1024):
    N=73*16
    E=rng.normal(0,20,(N,mp)).astype(np.float32)
    ids=[]
    for _ in range(N):
        aa=int(rng.integers(mp-16+1));bb=int(rng.integers(aa+8,mp-8+1))
        ids.append(np.concatenate((aa+rng.integers(8,size=2),bb+rng.integers(8,size=2))))
    ix=np.asarray(ids,np.int32)
    # Real pair sites have disjoint blocks; include unchanged-site proposals.
    ix[:100,1]=ix[:100,0];ix[100:200,3]=ix[100:200,2]
    def ref(e,i):
        na,oa,nb,ob=i;m=jnp.arange(len(e));z=jnp.float32(0)
        return jnp.sum(jnp.where(m==na,e,z)-jnp.where(m==oa,e,z)+jnp.where(m==nb,e,z)-jnp.where(m==ob,e,z))
    a=np.asarray(jax.jit(jax.vmap(ref))(E,ix))
    b=np.asarray(jax.jit(lambda e,i:jax.lax.map(lambda z:T._cpu_pair_energy(z[0],*z[1]),(e,i)))(E,ix))
    bad=np.flatnonzero(a!=b)
    r['math'][str(mp)]=dict(cases=N,differences=len(bad),maxdiff=float(np.abs(a-b).max()))
    if len(bad):
        k=int(bad[0]);r['math'][str(mp)]['example']=dict(indices=ix[k].tolist(),values=E[k,ix[k]].tolist(),ref=float(a[k]),sparse=float(b[k]))
    print(mp,r['math'][str(mp)],flush=True)
    BJ.jdump(r,str(O/'qualified-check.json'))
assert all(x['differences']==0 for x in r['math'].values())
base=next(x for x in json.load(open(O/'baseline12.json'))['runs'] if x['label']=='200k')
E=BJ.load_energies([BJ.TP+'/bench/gpu/energies_gpu.npz'])
for sid in ('4066edfcdb3ea8e3_AB','70c9c3eaa126c736_AB'):
    e=E[sid];mp=T.fit_bucket(len(e.g),len(e.nst),len(e.pairs))[0]
    keys=T.chain_keys(1,np.int64(API.chain_id_of(sid))*T.NPH+np.arange(T.NPH)).reshape(1,T.NPH)
    b=T.pack_bucket([e],[0],mp,200,keys,S=32)
    t=time.perf_counter();cnt=T.mc_cpu(T.PH_TERM,*(b[k] for k in ('g','occ','W','start','nst','pa','pb','meta','keys')))[0]
    prot,tot=T.counts_to_prot(e,cnt);curves=prot/200000
    got={k:digest(v) for k,v in dict(prot=prot,tot=tot,curves=curves,pka=T.curves_to_pka(curves)).items()}
    r['states'][sid]=dict(outputs_equal=got==base['outputs'][sid],raw_equal=digest(cnt[:,:len(e.g)])==base['raw_counts'][sid],seconds=time.perf_counter()-t)
    BJ.jdump(r,str(O/'qualified-check.json'));print(sid,r['states'][sid],flush=True)
r['passed']=all(x['raw_equal'] and x['outputs_equal'] for x in r['states'].values())
BJ.jdump(r,str(O/'qualified-check.json'));assert r['passed']
