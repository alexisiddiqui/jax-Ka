"""Small diagnostic for existing CPU pure_callback and OpenMP interaction."""
import os,ctypes as C,faulthandler,threading
assert os.environ.get('SLURM_JOB_ID')
faulthandler.dump_traceback_later(30)
watchdog=threading.Timer(120,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
import jax,numpy as np
from pathlib import Path
T=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/opt4')
def identity(x):
 print('identity enter',type(x),flush=True);out=np.asarray(x)+np.float32(1);print('identity return',flush=True);return out
f=jax.jit(lambda x:jax.pure_callback(identity,jax.ShapeDtypeStruct(x.shape,x.dtype),x))
print('before identity',flush=True); print(np.asarray(f(jax.device_put(np.ones(1,np.float32)))),flush=True)
lib=C.CDLL(str(T/'lib-sor-off-811943.so')); n=9; L=1
vals=[np.zeros((L,n,n,n),np.float32),np.ones((L,n,n,n),np.float32)]+[np.full((1,n,n,n),80.,np.float32) for _ in range(3)]+[np.full((1,n,n,n),1/480,np.float32),np.ones(L,np.float32),np.zeros(L,np.int32)]
def cb(*x):
 print('C callback enter',flush=True);q=np.array(x[0],copy=True,order='C');args=[q]+[np.ascontiguousarray(z) for z in x[1:]];it=np.zeros(L,np.int32);mc=np.zeros(L,np.float32);print('C callback converted',flush=True)
 lib.opt4_sor(*[C.c_void_p(z.ctypes.data) for z in args+[it,mc]],C.c_int(L),C.c_int(n),C.c_int(10),C.c_int(0),C.c_int(1));print('C callback returned',flush=True);return q,it,mc
spec=(jax.ShapeDtypeStruct(vals[0].shape,np.float32),jax.ShapeDtypeStruct((L,),np.int32),jax.ShapeDtypeStruct((L,),np.float32))
g=jax.jit(lambda *x:jax.pure_callback(cb,spec,*x))
print('direct C enter',flush=True); it=np.zeros(L,np.int32);mc=np.zeros(L,np.float32);lib.opt4_sor(*[C.c_void_p(z.ctypes.data) for z in vals+[it,mc]],C.c_int(L),C.c_int(n),C.c_int(10),C.c_int(0),C.c_int(1));print('direct C returned',flush=True)
print('before C callback',flush=True);out=jax.block_until_ready(g(*[jax.device_put(x) for x in vals]));print('done',np.asarray(out[1]),np.asarray(out[2]),flush=True)

watchdog.cancel();faulthandler.cancel_dump_traceback_later()
