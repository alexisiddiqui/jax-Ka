"""Fast, actual-Triton capability check before rerunning the experiment."""
import os
assert os.environ.get('SLURM_JOB_ID')
import jax
import numpy as np
from jaxpb import titrate as T
s=T.toy([3,2,4],3,6.)
b=T.pack_bucket([s],[0],64,2,T.chain_keys(71,np.arange(T.NPH)).reshape(1,T.NPH),S=1)
args=(T.PH_TERM,)+tuple(b[k] for k in ('g','occ','W','start','nst','pa','pb','meta','keys'))
a=np.asarray(T.mc_pallas(*args));r=np.asarray(T.mc_ref(*args))
print('INDEXED',np.array_equal(a,r),flush=True)
assert a.tobytes()==r.tobytes()
