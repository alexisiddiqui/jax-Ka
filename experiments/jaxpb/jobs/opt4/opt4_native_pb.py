"""Converged native phi/g comparison on a complete real PB state, arm only."""
import argparse,ctypes as C,os,time,json,faulthandler,threading
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
faulthandler.dump_traceback_later(120,repeat=True)
watchdog=threading.Timer(600,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
p=argparse.ArgumentParser(); p.add_argument('--tag',required=True); p.add_argument('--libraries',required=True); p.add_argument('--state',default='4066edfcdb3ea8e3_A'); a=p.parse_args()
import jax,numpy as np
import bench_jaxpb as BJ,prof_geomaps as PG
from jaxpb import api as A,sor as S
from opt2 import FIELDS,digest
O=Path(BJ.TP)/'opt4'; refs={}; lib=C.CDLL(a.libraries); original=S.run_pool; phase=0

def compare(x,y):
 x=np.asarray(x);y=np.asarray(y); delta=np.abs(x.astype(np.float64)-y)
 return dict(bitwise=x.tobytes()==y.tobytes(),different=int(np.count_nonzero(x!=y)),max_abs=float(delta.max(initial=0)),max_relative=float(np.max(delta/np.maximum(np.abs(y),1e-30),initial=0)))

def capture(*args,**kw):
 global phase
 out=original(*args,**kw); refs[phase]=tuple(np.asarray(x)[np.asarray(args[4],bool)].copy() for x in out); phase+=1; return out
S.run_pool=capture
record=PG.load(BJ.INPUTS,[a.state])[0]; cfg=A.Config(backend='xla'); t=time.perf_counter(); baseline=A.energies([record],cfg)[0]; basewall=time.perf_counter()-t
result=dict(tag=a.tag,job=os.environ['SLURM_JOB_ID'],state=a.state,library=a.libraries,baseline_wall_s=basewall,pools=[]); phase=0

def native(phi,rhs,om,op,valid,coefs,stats=None):
 global phase
 NI,n=phi.shape[:2]; ids=np.flatnonzero(valid).astype(np.int32); args=(phi,rhs,om,op,*coefs)
 def callback(p,r,o,op,*cf):
  q=np.ascontiguousarray(np.asarray(p)[ids]); rr=np.ascontiguousarray(np.asarray(r)[ids]); oo=np.ascontiguousarray(np.asarray(o)[ids]); operators=np.ascontiguousarray(np.asarray(op)[ids],dtype=np.int32); fields=[np.ascontiguousarray(x) for x in cf]; it=np.zeros(len(ids),np.int32); mc=np.full(len(ids),1e9,np.float32)
  arrays=[q,rr]+fields+[oo,operators,it,mc]
  lib.opt4_sor(*[C.c_void_p(x.ctypes.data) for x in arrays],C.c_int(len(ids)),C.c_int(n),C.c_int(500),C.c_int(1),C.c_int(len(os.sched_getaffinity(0))))
  out=np.zeros_like(p); out[ids]=q; its=np.zeros(NI,np.int32);mcs=np.zeros(NI,np.float32); its[ids]=it;mcs[ids]=mc;return out,its,mcs
 spec=(jax.ShapeDtypeStruct(phi.shape,np.float32),jax.ShapeDtypeStruct((NI,),np.int32),jax.ShapeDtypeStruct((NI,),np.float32))
 fn=jax.jit(lambda *x:jax.pure_callback(callback,spec,*x)); t=time.perf_counter(); out=jax.tree_util.tree_map(np.asarray,jax.block_until_ready(fn(*args))); elapsed=time.perf_counter()-t
 result['pools'].append(dict(phase=phase,seconds=elapsed,comparisons={k:compare(x[valid],y) for k,x,y in zip(('phi','sweeps','maxchange'),out,refs[phase])})); phase+=1
 if stats is not None:stats.append(dict(blocks=0,lane_blocks=int(out[1].sum()//10),needed=int(out[1].sum()//10)))
 return jax.device_put(out[0]),out[1],out[2]
S.run_pool=native; t=time.perf_counter(); got=A.energies([record],cfg)[0]; result['native_wall_s']=time.perf_counter()-t
result['energy']={k:compare(getattr(got,k),getattr(baseline,k)) for k in ('g','W')};result['pb']={k:compare(got.diag['pb'][k],baseline.diag['pb'][k]) for k in ('esolvM','esolvS','esolvM0','sweeps','maxchange','rho')}; result['completed']=True; result['host_hwm_kb']=BJ.status('VmHWM'); BJ.jdump(result,str(O/(a.tag+'.json')));print(json.dumps(result),flush=True)

watchdog.cancel();faulthandler.cancel_dump_traceback_later()
