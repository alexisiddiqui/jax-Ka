"""Measure a four-key sorting network against lax.sort in the gated pair tree."""
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
from jaxpb import api as API, titrate as T
import bench_jaxpb as BJ
from opt3_gate import digest
O=Path(BJ.TP)/'opt3';E=BJ.load_energies([BJ.TP+'/bench/gpu/energies_gpu.npz'])
base=next(x for x in json.load(open(O/'baseline12.json'))['runs'] if x['label']=='200k')
helper=inspect.getsource(T._cpu_pair_energy)
network='''    def swap(ka, va, kb, vb):
        flip = ka > kb
        return jnp.where(flip, kb, ka), jnp.where(flip, vb, va), jnp.where(flip, ka, kb), jnp.where(flip, va, vb)
    k0, k1, k2, k3 = rank[0], rank[1], rank[2], rank[3]
    a, b, c, d = values[0], values[1], values[2], values[3]
    k0, a, k1, b = swap(k0, a, k1, b)
    k2, c, k3, d = swap(k2, c, k3, d)
    k0, a, k2, c = swap(k0, a, k2, c)
    k1, b, k3, d = swap(k1, b, k3, d)
    k1, b, k2, c = swap(k1, b, k2, c)
'''
pattern='''    rank, values = lax.sort_key_val(rank, values)
    a, b, c, d = values[0], values[1], values[2], values[3]
'''
assert pattern in helper
fast=helper.replace(pattern,network).replace('rank[0] ^ rank[1]','k0 ^ k1').replace('rank[1] ^ rank[2]','k1 ^ k2').replace('rank[2] ^ rank[3]','k2 ^ k3')
r=dict(job=os.environ['SLURM_JOB_ID'],frozen=os.getcwd(),variants={})
for name,src in [('sort',helper),('network',fast)]:
 ns=dict(T.__dict__);exec(src,ns);exec(inspect.getsource(T._chain_cpu),ns);chain=ns['_chain_cpu']
 def make(chain):
  @jax.jit
  def one(ph,g,occ,W,start,nst,pa,pb,meta,key):
   acc=dict(row=lambda i:W[i],w=lambda i,j:W[i,j],start=lambda i:start[i],nst=lambda i:nst[i],pa=lambda p:pa[p],pb=lambda p:pb[p],u=ph*occ+g,start_vec=start,n=meta[0],P=meta[1])
   return chain(acc,key.astype(jnp.uint32),len(g),len(start),meta[2])
  return one
 fn=make(chain);rows={}
 for sid in ('4066edfcdb3ea8e3_AB','70c9c3eaa126c736_AB'):
  e=E[sid];mp=T.fit_bucket(len(e.g),len(e.nst),len(e.pairs))[0]
  keys=T.chain_keys(1,np.int64(API.chain_id_of(sid))*T.NPH+np.arange(T.NPH)).reshape(1,T.NPH)
  b=T.pack_bucket([e],[0],mp,200,keys,S=1)
  args=tuple(jax.device_put(b[k][0]) for k in ('g','occ','W','start','nst','pa','pb','meta'))
  fn(T.PH_TERM[0],*args,keys[0,0]).block_until_ready()
  t=time.perf_counter()
  with ThreadPoolExecutor(max_workers=12) as pool:
   cnt=np.stack(list(pool.map(lambda j:np.asarray(fn(T.PH_TERM[j],*args,keys[0,j])),range(T.NPH))))
  prot,tot=T.counts_to_prot(e,cnt);curves=prot/200000
  got={k:digest(v) for k,v in dict(prot=prot,tot=tot,curves=curves,pka=T.curves_to_pka(curves)).items()}
  rows[sid]=dict(seconds=time.perf_counter()-t,outputs_equal=got==base['outputs'][sid],raw_equal=digest(cnt[:,:len(e.g)])==base['raw_counts'][sid])
  r['variants'][name]=rows;BJ.jdump(r,str(O/'sort.json'));print(name,sid,rows[sid],flush=True)
