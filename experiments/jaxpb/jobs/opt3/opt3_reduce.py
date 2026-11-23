"""Identify the CPU batched reduction order from sparse f32 sums."""
import json
import os
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import bench_jaxpb as BJ

rng=np.random.default_rng(13)
r=dict(job=os.environ['SLURM_JOB_ID'],frozen=os.getcwd(),rows={})
for Mp in (64,128,256,512,1024):
    N=73
    E=rng.normal(0,20,(N,Mp)).astype(np.float32)
    ix=[]
    widths=[w for w in (1,2,4,8,16,32,64,128,256) if 4*w<=Mp]
    for i in range(N):
        w=widths[i%len(widths)]
        p=np.arange(4,dtype=np.int32)*w+int(rng.integers(w))
        rng.shuffle(p);ix.append(p)
    ix=np.array(ix,np.int32)
    def energy(e,i):
        na,oa,nb,ob=i;idx=jnp.arange(Mp)
        return jnp.sum(jnp.where(idx==na,e,0)-jnp.where(idx==oa,e,0)+jnp.where(idx==nb,e,0)-jnp.where(idx==ob,e,0))
    target=np.asarray(jax.jit(jax.vmap(energy))(E,ix))
    scalar=np.asarray(jax.jit(lambda e,i:lax.map(lambda z:energy(*z),(e,i)))(E,ix))
    vals=E[np.arange(N)[:,None],ix]*np.array([1,-1,1,-1],np.float32)
    order=np.argsort(ix,axis=1);positions=np.take_along_axis(ix,order,axis=1);values=np.take_along_axis(vals,order,axis=1)
    matches=[];scores={}
    for width in (1,2,4,8,16,32,64,128,256,512,1024):
        if width>Mp:continue
        a=np.zeros((N,width),np.float32)
        for k in range(4):
            loc=positions[:,k]%width
            a[np.arange(N),loc]=a[np.arange(N),loc]+values[:,k]
        for mode in ('adjacent','halves','sequential'):
            b=a.copy()
            if mode=='sequential':
                z=np.zeros(N,np.float32)
                for k in range(width):z=z+b[:,k]
            else:
                while b.shape[1]>1:
                    n=b.shape[1]//2
                    b=b[:,::2]+b[:,1::2] if mode=='adjacent' else b[:,:n]+b[:,n:]
                z=b[:,0]
            key=f'{width}/{mode}';scores[key]=int(np.sum(z!=target))
            if np.array_equal(z,target):matches.append(key)
    unroll_matches=[]
    for vector in (4,8,16,32):
        for unroll in (1,2,4,8):
            width=vector*unroll
            if width>Mp:continue
            a=np.zeros((N,width),np.float32)
            for k in range(4):
                loc=positions[:,k]%width
                a[np.arange(N),loc]=a[np.arange(N),loc]+values[:,k]
            for combine in ('seq','halves','adjacent'):
                chunks=a.reshape(N,unroll,vector)
                if combine=='seq':
                    v=np.zeros((N,vector),np.float32)
                    for k in range(unroll):v=v+chunks[:,k]
                else:
                    while chunks.shape[1]>1:
                        n=chunks.shape[1]//2
                        chunks=chunks[:,::2]+chunks[:,1::2] if combine=='adjacent' else chunks[:,:n]+chunks[:,n:]
                    v=chunks[:,0]
                while v.shape[1]>1:
                    n=v.shape[1]//2;v=v[:,:n]+v[:,n:]
                score=int(np.sum(v[:,0]!=target))
                if not score:unroll_matches.append(f'{vector}x{unroll}/{combine}')
    r['rows'][str(Mp)]=dict(unroll_matches=unroll_matches,scalar_differences=int(np.sum(scalar!=target)),exact_sparse_orders=matches,best=sorted(scores.items(),key=lambda x:x[1])[:8])
    print(Mp,r['rows'][str(Mp)],flush=True)
BJ.jdump(r,str(Path(BJ.TP)/'opt3/reduce-unroll.json'))
