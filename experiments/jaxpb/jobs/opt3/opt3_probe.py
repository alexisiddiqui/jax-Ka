"""Measure each scalar-XLA formulation separately; never changes production code."""
import concurrent.futures
import inspect
import json
import os
import time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import numpy as np
import jax
from jaxpb import titrate as T
import bench_jaxpb as BJ

O = Path(BJ.TP) / 'opt3'
r = dict(job=os.environ['SLURM_JOB_ID'], frozen=os.getcwd(), env=BJ.env_info(), candidates={})
try:
    import numba
    r['numba'] = numba.__version__
except ImportError as e:
    r['numba'] = repr(e)
BJ.jdump(r, str(O / 'trials.json'))
E = BJ.load_energies([BJ.TP + '/bench/gpu/energies_gpu.npz'])
ph = jax.device_put(T.PH_TERM)
src = inspect.getsource(T.chain).replace('def chain(', 'def candidate(')
variants = {'branch': src}
src = src.replace('return jnp.sum(jnp.where(sidx == i, curv, 0))', 'return curv[i]')
src = src.replace('jnp.where(sidx == i, nw, curv)', 'curv.at[i].set(nw)')
src = src.replace('jnp.where(sidx == a, na, jnp.where(sidx == b, nb, curv))', 'curv.at[a].set(na).at[b].set(nb)')
variants['index'] = src
src = src.replace('jnp.sum(jnp.where(idx == nw, E, zero) - jnp.where(idx == oc, E, zero))', 'E[nw] - E[oc]')
variants['site'] = src
src = src.replace('jnp.sum(jnp.where(idx == na, E, zero) - jnp.where(idx == oa, E, zero)\n                     + jnp.where(idx == nb, E, zero) - jnp.where(idx == ob, E, zero))', '(E[na] - E[oa]) + (E[nb] - E[ob])')
variants['pair'] = src
fns = {}
for name, source in variants.items():
    ns = dict(T.__dict__); exec(source, ns); chain = ns['candidate']
    def make(chain):
        @jax.jit
        def one(ph, g, occ, W, start, nst, pa, pb, meta, key):
            u = ph * occ + g
            acc = dict(row=lambda i: W[i], w=lambda i,j: W[i,j], start=lambda i:start[i], nst=lambda i:nst[i],
                       pa=lambda p:pa[p], pb=lambda p:pb[p], u=u, start_vec=start, n=meta[0], P=meta[1])
            return chain(acc, key.astype('uint32'), len(g), len(start), meta[2])
        return one
    fns[name] = make(chain)
for Mp in (64,128,256,512,1024):
    ids = [s for s in BJ.STATES if T.fit_bucket(len(E[s].g),len(E[s].nst),len(E[s].pairs))[0] == Mp]
    states = [E[s] for s in ids]
    keys = T.chain_keys(1, np.arange(len(ids)*T.NPH)).reshape(len(ids),T.NPH)
    B = T.pack_bucket(states,list(range(len(ids))),Mp,2,keys,S=32)
    ref = np.zeros((32,T.NPH,Mp),np.float32)
    # Baseline only: use S=1 to avoid paying for 29-30 padded lanes.
    t=time.perf_counter()
    for i in range(len(ids)):
        b=T.pack_bucket(states,[i],Mp,2,keys[i:i+1],S=1)
        args=(ph,)+tuple(b[k] for k in ('g','occ','W','start','nst','pa','pb','meta','keys'))
        ref[i]=np.asarray(T.mc_ref(*args))[0]
    rt=time.perf_counter()-t
    # Make host/device views before timing; no tensor indexing gets compiled
    # while a candidate is measured. All variants run twice after compilation.
    shared=[tuple(jax.device_put(B[k][i]) for k in ('g','occ','W','start','nst','pa','pb','meta')) for i in range(len(ids))]
    for name,fn in fns.items():
        fn(T.PH_TERM[0],*shared[0],keys[0,0]).block_until_ready()
        def run(ij):
            i,j=ij
            return i,j,np.asarray(fn(T.PH_TERM[j],*shared[i],keys[i,j]))
        times=[]
        for rep in range(2):
            t=time.perf_counter();out=np.zeros_like(ref)
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                for i,j,c in pool.map(run,[(i,j) for i in range(len(ids)) for j in range(T.NPH)]): out[i,j]=c
            times.append(time.perf_counter()-t)
        row=dict(states=ids,seconds=times[-1],warm_runs=times,ref_cold_seconds=rt,equal=bool(np.array_equal(out,ref)),
                 differing_counts=int(np.sum(out!=ref)),maxdiff=float(np.abs(out-ref).max()))
        r['candidates'].setdefault(name,{})[str(Mp)]=row
        print(name,Mp,row,flush=True);BJ.jdump(r,str(O/'trials.json'))
