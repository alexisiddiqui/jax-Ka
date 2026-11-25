"""Native SOR prototype, standalone arm only. Compile solely using existing node tools."""
import argparse, ctypes as C, json, os, shutil, subprocess, time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
p=argparse.ArgumentParser(); p.add_argument('--tag',required=True); p.add_argument('--pin',type=int); p.add_argument('--lanes',default='native-lanes.npz'); p.add_argument('--skip-callback',action='store_true'); p.add_argument('--source-c',default='jobs/opt4_native.c'); a=p.parse_args()
if a.pin: os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:a.pin]); os.environ['OMP_NUM_THREADS']=str(a.pin)
import jax, jax.numpy as jnp, numpy as np
import bench_jaxpb as BJ
from jaxpb import sor as S, geometry as G, records as R
O=Path(BJ.TP)/'opt4'; z=np.load(O/a.lanes); vals={k:np.ascontiguousarray(z[k]) for k in z.files}; p0=vals['phi']; n=p0.shape[-1]; L=len(p0); result=dict(tag=a.tag,env=BJ.env_info(),source_lanes=str(O/a.lanes),variants={},xla={})
# Existing real operator fields, rhs, boundary phi and omega, no regenerated numerics.
coefs=tuple(jax.device_put(vals[k]) for k in ('ex','ey','ez','dinv')); rhs=jax.device_put(vals['rhs']); om=jax.device_put(vals['om']); op=jax.device_put(vals['op']); ca,cb=(jnp.asarray(x) for x in S.colour_masks(n))
def native(lib,steps,conv=False,threads=1):
 q=p0.copy(); it=np.zeros(L,np.int32); mc=np.full(L,1e9,np.float32)
 args=[q,vals['rhs']]+[vals[k] for k in ('ex','ey','ez','dinv','om','op')]+[it,mc]
 t=time.perf_counter(); lib.opt4_sor(*[C.c_void_p(x.ctypes.data) for x in args],C.c_int(L),C.c_int(n),C.c_int(steps),C.c_int(conv),C.c_int(threads)); return q,it,mc,time.perf_counter()-t
refs={}
with G.x32():
 def fixed(steps):
  if steps==1:
   return one(jax.device_put(p0.copy()),rhs,*[x[op] for x in coefs],om)
  phi=jax.device_put(p0.copy()); it=jax.device_put(np.zeros(L,np.int32)); mc=jax.device_put(np.full(L,1e9,np.float32))
  for block in range(steps//R.CHECK):
   # Use the actual checked 10-sweep executable repeatedly. Force the requested
   # fixed sweep count by clearing done between blocks, without splitting a sweep.
   phi,it,mc,done=S._cpu_block(phi,rhs,it,mc,jax.device_put(np.zeros(L,bool)),om,op,*coefs,ca,cb)
  return phi,it,mc,done
 one=jax.jit(jax.vmap(lambda phi,rhs,ex,ey,ez,dv,om:S.sor_chunk(phi,rhs,ex,ey,ez,dv,ca,cb,om,jnp.int32(0),jnp.float32(1e9),jnp.bool_(False),check=1)))
 for steps in (1,10,100):
  jax.block_until_ready(fixed(steps)); t=time.perf_counter(); out=jax.block_until_ready(fixed(steps)); dt=time.perf_counter()-t
  refs[steps]=np.asarray(out[0]); result['xla'][str(steps)]=dict(seconds=dt,seconds_per_lane_sweep=dt/(L*steps),check_schedule='1 isolated sweep' if steps==1 else 'production 10-sweep blocks, fixed-count arm')
  print('XLA',a.tag,steps,dt,flush=True)
 # Production compaction/check schedule and full converged phi.
 def full():
  return S.device_solve(jax.device_put(p0.copy()),rhs,jnp.zeros(L,jnp.int32),jnp.full(L,1e9,jnp.float32),jnp.zeros(L,bool),om,op,*coefs,n=n,widths=tuple(w for w in R.CPU_LANE_WIDTHS if w<=L))
 jax.block_until_ready(full()); t=time.perf_counter(); fullref=jax.tree_util.tree_map(np.asarray,jax.block_until_ready(full())); result['xla']['converged']=dict(seconds=time.perf_counter()-t,sweeps=fullref[1].tolist(),maxchange=fullref[2].tolist()); print('XLA converged',a.tag,result['xla']['converged']['seconds'],flush=True)
 for mode in ('off','fast'):
  BJ.reset_hwm()
  compiler=shutil.which('gcc') or shutil.which('clang'); libpath=O/f'lib-sor-{a.tag}-{mode}-{os.environ["SLURM_JOB_ID"]}.so'; cmd=[compiler,'-O3','-march=native','-shared','-fPIC','-fopenmp',f'-ffp-contract={mode}',a.source_c,'-o',str(libpath)]
  row=dict(command=cmd,compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]); proc=subprocess.run(cmd,capture_output=True,text=True); row['compile_returncode']=proc.returncode; row['stderr']=proc.stderr
  if proc.returncode: result['variants'][mode]=row; continue
  lib=C.CDLL(str(libpath)); row['checks_by_threads']={}
  print('Native starting',mode,flush=True)
  for threads in sorted(set((1,len(os.sched_getaffinity(0))))):
   checks={};row['checks_by_threads'][str(threads)]=checks
   for steps in (1,10,100):
    got,_,_,_=native(lib,steps,threads=threads); delta=np.abs(got.astype(np.float64)-refs[steps]); checks[str(steps)]=dict(bitwise=got.tobytes()==refs[steps].tobytes(),different=int(np.count_nonzero(got!=refs[steps])),max_abs=float(delta.max()))
  row['checks']=row['checks_by_threads']['1']
  print('Native checks',mode,row['checks'],flush=True)
  row['timing']={}
  for threads in sorted(set((1,len(os.sched_getaffinity(0))))):
   native(lib,100,threads=threads); ts=[native(lib,100,threads=threads)[3] for _ in range(3)]; row['timing'][str(threads)]=dict(seconds=ts,seconds_per_lane_sweep=float(np.median(ts))/(L*100),speedup=result['xla']['100']['seconds']/float(np.median(ts)),speedup_reference_xla_threads=result['env']['affinity'])
  print('Native timings',mode,row['timing'],flush=True)
  q,it,mc,dt=native(lib,500,True,len(os.sched_getaffinity(0))); row['converged']=dict(phi_bitwise=q.tobytes()==fullref[0].tobytes(),sweeps_bitwise=it.tobytes()==fullref[1].tobytes(),maxchange_bitwise=mc.tobytes()==fullref[2].tobytes(),max_abs=float(np.max(np.abs(q.astype(np.float64)-fullref[0]))),seconds=dt,sweeps=it.tolist(),maxchange=mc.tolist())
  result['variants'][mode]=row; BJ.jdump(result,str(O/(a.tag+'.json'))); print('Native convergence',mode,row['converged'],flush=True)
  if not a.skip_callback:
   def callback(*inputs):
    q=np.array(inputs[0],copy=True,order='C'); arrays=[q]+[np.ascontiguousarray(x) for x in inputs[1:]]; it=np.zeros(L,np.int32); mc=np.full(L,1e9,np.float32)
    lib.opt4_sor(*[C.c_void_p(x.ctypes.data) for x in arrays+[it,mc]],C.c_int(L),C.c_int(n),C.c_int(500),C.c_int(1),C.c_int(len(os.sched_getaffinity(0))))
    return q,it,mc
   cbargs=tuple(jax.device_put(vals[k]) for k in ('phi','rhs','ex','ey','ez','dinv','om','op'))
   spec=(jax.ShapeDtypeStruct(p0.shape,np.float32),jax.ShapeDtypeStruct((L,),np.int32),jax.ShapeDtypeStruct((L,),np.float32))
   cbfn=jax.jit(lambda *xs:jax.pure_callback(callback,spec,*xs))
   jax.block_until_ready(cbfn(*cbargs)); t=time.perf_counter(); cbout=jax.tree_util.tree_map(np.asarray,jax.block_until_ready(cbfn(*cbargs))); cbdt=time.perf_counter()-t
   row['pure_callback']=dict(seconds=cbdt,phi_bitwise=cbout[0].tobytes()==fullref[0].tobytes(),sweeps_bitwise=cbout[1].tobytes()==fullref[1].tobytes(),maxchange_bitwise=cbout[2].tobytes()==fullref[2].tobytes(),speedup=result['xla']['converged']['seconds']/cbdt)
  else:
   row['pure_callback']=dict(deferred='Separate full CPU PB phi/g process avoids mixing host/OpenMP and callback timings')
  row['host_hwm_kb']=BJ.status('VmHWM')
  result['variants'][mode]=row; BJ.jdump(result,str(O/(a.tag+'.json')))
result['host_hwm_kb']=BJ.status('VmHWM'); result['completed']=True; BJ.jdump(result,str(O/(a.tag+'.json'))); print(json.dumps(result),flush=True)
