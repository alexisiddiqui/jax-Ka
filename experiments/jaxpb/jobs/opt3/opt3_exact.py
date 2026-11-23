"""Localise the 200k scalar-chain mismatch and test conservative reductions."""
import inspect
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxpb import titrate as T, api as API
import bench_jaxpb as BJ
from opt3_gate import digest

O=Path(BJ.TP)/'opt3'
base=next(x for x in json.load(open(O/'baseline12.json'))['runs'] if x['label']=='200k')
trial=next(x for x in json.load(open(O/'cpu12.json'))['runs'] if x['label']=='200k')
E=BJ.load_energies([BJ.TP+'/bench/gpu/energies_gpu.npz'])
names=['4066edfcdb3ea8e3_AB','70c9c3eaa126c736_AB']
r=dict(job=os.environ['SLURM_JOB_ID'],frozen=os.getcwd(),variants={},curve_differences={})
bsh={x['state_id']:x for x in API.load_shard(base['shard'],full_W=False)}
csh={x['state_id']:x for x in API.load_shard(trial['shard'],full_W=False)}
for s in names:
    a,b=bsh[s]['curves'],csh[s]['curves'];ix=np.argwhere(a!=b)
    r['curve_differences'][s]=[dict(ph_index=int(i),site=int(j),old=float(a[i,j]),trial=float(b[i,j]),count_delta=int(np.rint((b[i,j]-a[i,j])*200000))) for i,j in ix]

# Same expression under a batched vs scalar reduction, and scalar acceptance.
rng=np.random.default_rng(13);N=4096;Mp=512
v=rng.normal(0,20,(N,Mp)).astype(np.float32)
ix=np.array([rng.choice(Mp,4,replace=False) for _ in range(N)],np.int32)
x=rng.uniform(0,50,N).astype(np.float32)
def energy(e,i):
    na,oa,nb,ob=i;idx=jnp.arange(Mp)
    return jnp.sum(jnp.where(idx==na,e,0)-jnp.where(idx==oa,e,0)+jnp.where(idx==nb,e,0)-jnp.where(idx==ob,e,0))
a=np.asarray(jax.jit(jax.vmap(energy))(v,ix))
b=np.asarray(jax.jit(lambda v,i:lax.map(lambda z:energy(*z),(v,i)))(v,ix))
c=np.asarray(jax.jit(T.invexp)(x));d=np.asarray(jax.jit(lambda x:lax.map(T.invexp,x))(x))
r['isolated']=dict(pair_reduce_batched_vs_scalar=int(np.sum(a!=b)),invexp_vector_vs_scalar=int(np.sum(c!=d)))
BJ.jdump(r,str(O/'exact-tree.json'));print('DIAG',r,flush=True)

original=inspect.getsource(T.chain).replace('def chain(', 'def candidate(')
indexed=original.replace('return jnp.sum(jnp.where(sidx == i, curv, 0))','return curv[i]').replace('jnp.where(sidx == i, nw, curv)','curv.at[i].set(nw)').replace('jnp.where(sidx == a, na, jnp.where(sidx == b, nb, curv))','curv.at[a].set(na).at[b].set(nb)')
site=inspect.getsource(T._chain_cpu).replace('def _chain_cpu(', 'def candidate(')
pattern='''jnp.sum(jnp.where(idx == na, E, zero) - jnp.where(idx == oa, E, zero)
                     + jnp.where(idx == nb, E, zero) - jnp.where(idx == ob, E, zero))'''
ordered='''pair_ordered(E, na, oa, nb, ob)'''
def pair_ordered(E,na,oa,nb,ob):
    z=jnp.float32(0)
    aa=jnp.where(na==oa,z,jnp.where(na<oa,E[na],-E[oa]))
    ab=jnp.where(na==oa,z,jnp.where(na<oa,-E[oa],E[na]))
    ba=jnp.where(nb==ob,z,jnp.where(nb<ob,E[nb],-E[ob]))
    bb=jnp.where(nb==ob,z,jnp.where(nb<ob,-E[ob],E[nb]))
    # select_pairs orders sites; support reversed input pairs as well.
    first,second,third,fourth=(jnp.where(na<nb,x,y) for x,y in ((aa,ba),(ab,bb),(ba,aa),(bb,ab)))
    return lax.optimization_barrier(lax.optimization_barrier(first+second)+third)+fourth
assert pattern in site
def pair_tree(E,na,oa,nb,ob):
    idx=jnp.arange(len(E));z=jnp.float32(0)
    v=jnp.where(idx==na,E,z)-jnp.where(idx==oa,E,z)+jnp.where(idx==nb,E,z)-jnp.where(idx==ob,E,z)
    width=1 << (len(E)-1).bit_length()
    v=jnp.pad(v,(0,width-len(E)))
    while width>1:
        width//=2
        v=lax.optimization_barrier(v[:width]+v[width:])
    return v[0]
r['odd_tree_check']={}
for mp in (23,31,64,65,128,256,512,1024):
    e=rng.normal(0,20,(73,mp)).astype(np.float32)
    inds=np.array([rng.choice(mp,4,replace=False) for _ in range(73)],np.int32)
    def ref(ee,ii):
        na,oa,nb,ob=ii;i=jnp.arange(len(ee));z=jnp.float32(0)
        return jnp.sum(jnp.where(i==na,ee,z)-jnp.where(i==oa,ee,z)+jnp.where(i==nb,ee,z)-jnp.where(i==ob,ee,z))
    a=np.asarray(jax.jit(jax.vmap(ref))(e,inds))
    b=np.asarray(jax.jit(jax.vmap(lambda e,i:pair_tree(e,*i)))(e,inds))
    r['odd_tree_check'][str(mp)]=int(np.sum(a!=b))
print('TREE MATH',r['odd_tree_check'],flush=True)
variants={'tree':site.replace(pattern,'pair_tree(E, na, oa, nb, ob)')}

for name,src in variants.items():
    ns=dict(T.__dict__,pair_ordered=pair_ordered,pair_tree=pair_tree);exec(src,ns);chain=ns['candidate']
    def make(chain):
        @jax.jit
        def one(ph,g,occ,W,start,nst,pa,pb,meta,key):
            acc=dict(row=lambda i:W[i],w=lambda i,j:W[i,j],start=lambda i:start[i],nst=lambda i:nst[i],pa=lambda p:pa[p],pb=lambda p:pb[p],u=ph*occ+g,start_vec=start,n=meta[0],P=meta[1])
            return chain(acc,key.astype(jnp.uint32),len(g),len(start),meta[2])
        return one
    fn=make(chain);rows={}
    for sid in names:
        e=E[sid];mp=T.fit_bucket(len(e.g),len(e.nst),len(e.pairs))[0]
        keys=T.chain_keys(1,np.int64(API.chain_id_of(sid))*T.NPH+np.arange(T.NPH)).reshape(1,T.NPH)
        b=T.pack_bucket([e],[0],mp,200,keys,S=1)
        args=tuple(jax.device_put(b[k][0]) for k in ('g','occ','W','start','nst','pa','pb','meta'))
        fn(T.PH_TERM[0],*args,keys[0,0]).block_until_ready()
        t=time.perf_counter()
        with ThreadPoolExecutor(max_workers=12) as pool:
            cnt=np.stack(list(pool.map(lambda j:np.asarray(fn(T.PH_TERM[j],*args,keys[0,j])),range(T.NPH))))
        prot,tot=T.counts_to_prot(e,cnt);curves=prot/200000
        got={k:digest(z) for k,z in dict(prot=prot,tot=tot,curves=curves,pka=T.curves_to_pka(curves)).items()}
        raw=digest(cnt[:,:len(e.g)])
        rows[sid]=dict(seconds=time.perf_counter()-t,outputs_equal=got==base['outputs'][sid],raw_equal=raw==base['raw_counts'][sid],raw=raw,
                       differing_curve_counts=int(np.sum(curves!=bsh[sid]['curves'])))
        r['variants'][name]=rows;BJ.jdump(r,str(O/'exact-tree.json'));print(name,sid,rows[sid],flush=True)
    if all(x['raw_equal'] and x['outputs_equal'] for x in rows.values()):
        r['first_matching_variant']=name;BJ.jdump(r,str(O/'exact-tree.json'));break
