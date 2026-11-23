"""Compile-only inspection of the actual reference CPU reduction topology."""
import json
import os
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import jax
import numpy as np
from jaxpb import titrate as T
import bench_jaxpb as BJ
E=BJ.load_energies([BJ.TP+'/bench/gpu/energies_gpu.npz'])
r=dict(job=os.environ['SLURM_JOB_ID'],frozen=os.getcwd(),programs={})
for sid in ('4066edfcdb3ea8e3_AB','70c9c3eaa126c736_AB'):
 e=E[sid];mp=T.fit_bucket(len(e.g),len(e.nst),len(e.pairs))[0]
 b=T.pack_bucket([e],[0],mp,1,np.zeros((1,T.NPH),np.int32),S=1)
 args=(T.PH_TERM,)+tuple(b[k] for k in ('g','occ','W','start','nst','pa','pb','meta','keys'))
 text=T.mc_ref.lower(*args).compile().as_text()
 r['programs'][str(mp)]=text
 print(mp,'\n'.join(x for x in text.splitlines() if 'reduce' in x.lower() or 'window' in x.lower()),flush=True)
BJ.jdump(r,str(Path(BJ.TP)/'opt3/reduction-hlo.json'))
