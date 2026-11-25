"""Prepare real coarse/model lane fields on A40; native comparisons run on amd96.
The standalone CPU sweep references consume these exact same f32 inputs. The
full native phi/g gate separately prepares its inputs through the CPU PB path.
"""
import os
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import jax,jax.numpy as jnp,numpy as np
import bench_jaxpb as BJ,prof_geomaps as PG
from jaxpb import api as A,pb as P,sor as S
O=Path(BJ.TP)/'opt4'; coefs=None
original=P.setup1
@jax.jit
def fields(mc,ms):
 a=jax.vmap(lambda m:P._coefs(m,1.0))(mc); b=jax.vmap(lambda m:P._coefs(m,.25))(ms)
 return tuple(jnp.concatenate([x,y]) for x,y in zip(a,b))
def setup(*args,**kw):
 global coefs
 out=original(*args,**kw); coefs=fields(args[0],args[1]); return out
class Done(Exception):pass
def capture(phi,rhs,om,op,valid,original_coefs,**kw):
 ids=np.flatnonzero(valid); ids=np.concatenate([ids[:8],ids[len(ids)//2:len(ids)//2+8]]); ops=np.asarray(op)[ids]; unique,inv=np.unique(ops,return_inverse=True)
 np.savez_compressed(O/'gpu-native-lanes.npz',phi=np.asarray(phi)[ids],rhs=np.asarray(rhs)[ids],om=np.asarray(om)[ids],op=inv.astype(np.int32),**{k:np.asarray(v)[unique] for k,v in zip(('ex','ey','ez','dinv'),coefs)})
 BJ.jdump(dict(job=os.environ['SLURM_JOB_ID'],state='4066edfcdb3ea8e3_AB',source='A40 maps, float fields from pb._coefs, real coarse and model lanes; replayed by native CPU and XLA CPU on identical inputs',lanes=len(ids),env=BJ.env_info(),completed=True),str(O/'gpu-native-lanes.json')); raise Done()
P.setup1=setup;S.run_pool_pallas=capture
try:A.energies(PG.load(BJ.INPUTS,['4066edfcdb3ea8e3_AB']),A.Config(backend='pallas'))
except Done:print('Prepared real lane inputs',flush=True)
