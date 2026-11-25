"""CPU production baseline + separate warm JIT decomposition on every real batch.
Decomposition timings exclude compilation, include synchronization, and are NOT additive
estimates of the fused production executable. Source and candidate arithmetic are unedited.
"""
import argparse, collections, inspect, json, os, shutil, subprocess, time, types
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
p=argparse.ArgumentParser(); p.add_argument('--tag',required=True); p.add_argument('--pin',type=int); a=p.parse_args()
if a.pin: os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:a.pin]); os.environ['OMP_NUM_THREADS']=str(a.pin)
import numpy as np
import jax
import jax.numpy as jnp
import bench_jaxpb as BJ, prof_geomaps as PG, prof2, speed as SP
from opt2 import CPU_IDS,digest
from jaxpb import maps as M, geometry as G, pb as P, api as API, sor as S, records as R
PG.d_finish=jax.jit(lambda pts,valid,b,ba,occ:G._finish(pts,valid,b,ba,occ,jnp.arange(len(pts),dtype=jnp.int64))[:4])
O=Path(BJ.TP)/'opt4'; ct=BJ.CompileTime(); times=collections.defaultdict(float); calls=collections.Counter(); context={}; cache={}
def timed(label,f,*args,**kw):
 t=time.perf_counter(); c0=ct.s; out=f(*args,**kw); jax.block_until_ready(out); times[label]+=max(0,time.perf_counter()-t-(ct.s-c0)); calls[label]+=1; return out

def frozen(f):
 if f.__closure__ is None:return f
 cell=lambda value:(lambda:value).__closure__[0]
 return types.FunctionType(f.__code__,f.__globals__,f.__name__,f.__defaults__,tuple(cell(c.cell_contents) for c in f.__closure__))

def V(f,*axes):
 def run(*args):
  name=f.__name__; key=(context['grid'],context['params'],name,f.__code__.co_firstlineno,context.get('ax',0),str(axes),str([(getattr(x,'shape',None),getattr(x,'dtype',None)) for x in jax.tree_util.tree_leaves(args)]))
  # The only varying closed-over array is offv (one fixed axis per stage).
  if key not in cache: cache[key]=jax.jit(jax.vmap(frozen(f),in_axes=axes))
  if name=='stage0':
   pos,rad,probes,ac=args; h,n,Ka,Kp,nca,ncp,PB,brute,shared=context['params']; B=len(ac); pa=None if shared else 0
   def cells(pos,rad,probes,ac):
    lo=ac-(n-1)*h/2-4.5; lop=ac-(n-1)*h/2-1.9
    ta=M.build_cells(pos,rad,lo,nca,4.01,Ka,batch_size=B) if not brute else ()
    tp=M.build_cells(probes,jnp.zeros(probes.shape[0],pos.dtype),lop,ncp,M.PCELL,Kp,batch_size=B)
    return ta,tp
   ck=(context['grid'],context['params'],'cells',str(axes))
   if ck not in cache: cache[ck]=jax.jit(jax.vmap(cells,in_axes=(pa,pa,pa,0)))
   timed('cells',cache[ck],*args)
  return timed(name,cache[key],*args)
 return run

def dyn(fn,Q,cnt,chunk,args,axes):
 i=context['q']; context['q']+=1; label='probe_projection' if i==9 else 'atom_queries'
 key=(context['grid'],context['params'],'dyn',i,str(Q.shape),str(axes)); fn=frozen(fn)
 if key not in cache: cache[key]=jax.jit(lambda Q,cnt,args:M._dyn_rows(fn,Q,cnt,chunk,args,axes))
 return timed(label,cache[key],Q,cnt,args)

def reach(*args):
 key=(context['grid'],'reach',args[7])
 if key not in cache: cache[key]=jax.jit(M.reach_mask_b,static_argnums=(4,5,6,7,8,9))
 return timed('reach_masks',cache[key],*args)
# Host copy retains production culling, dynamic query counts, B4 arithmetic and padding.
source=inspect.getsource(M.build_maps_b).replace('def build_maps_b(', 'def decompose(').replace('    V = lambda f, *ax: jax.vmap(f, in_axes=ax)','    V = _V').replace('    for ax, name in enumerate(("ex", "ey", "ez")):', '    for ax, name in enumerate(("ex", "ey", "ez")):\n        _context["ax"] = ax')
ns=dict(vars(M)); ns.update(_V=V,_dyn_rows=dyn,reach_mask_b=reach,_context=context); exec(source,ns); decompose=ns['decompose']
orig=M.maps_batch; current={}; result=dict(tag=a.tag,env=BJ.env_info(),rows={},method=__doc__,compilers={c:shutil.which(c) for c in ('gcc','clang','cc')})
try:
 import jax.ffi
 result['ffi_include']=jax.ffi.include_dir()
 result['ffi_headers']={name:(Path(result['ffi_include'])/name).is_file() for name in ('xla/ffi/api/ffi.h','xla/ffi/api/c_api.h')}
except Exception as e: result['ffi_include']=repr(e)
for c in ('gcc','clang'):
 if shutil.which(c): result['compilers'][c]=subprocess.check_output([c,'--version'],text=True).splitlines()[0]

def maps(*args,**kw):
 tbase=time.perf_counter(); cb=ct.s; out=orig(*args,**kw); jax.block_until_ready(out); current.setdefault('production_maps_s',[]).append(max(0,time.perf_counter()-tbase-(ct.s-cb))); pos,rad,pr,ac,sp,sr,spr=args; sz=kw['sz']; active=kw.get('active',jnp.ones(len(ac),bool)); B=len(ac); eq=True
 for start in range(0,B,4):
  sl=slice(start,start+4)
  if not bool(np.asarray(active[sl]).any()): continue
  for gi,(grid,h,NS,NB,PB,brute,shared) in enumerate((('coarse',1.,sz.NSc,sz.NBc,len(pr),False,True),('focused',.25,sz.NSf,sz.NBf,sz.PBf,False,True),('model',.25,sz.NSs,sz.NBs,sz.PBs,True,False))):
   n=R.GSIZE; nca=M.ncells(h,n,'a'); ncp=M.ncells(h,n,'p'); context.update(grid=grid,q=0,ax=0,params=(h,n,sz.Ka,sz.Kp,nca,ncp,PB,brute,shared))
   oo,cc=decompose(pos if shared else sp[sl],rad if shared else sr[sl],pr if shared else spr[sl],ac[sl],h,n,sz.Ka,sz.Kp,nca,ncp,NS,NB,PB,brute,h,shared,active=active[sl])
   differences={}
   for key,old,new in [(k,out[gi][k][sl],oo[k]) for k in oo]+[('counters',out[3][sl,gi],cc)]:
    old=np.asarray(old);new=np.asarray(new);same=old.tobytes()==new.tobytes();eq &= same
    if not same:
     row=dict(different=int(np.count_nonzero(old!=new)),shape=old.shape,dtype=str(old.dtype))
     if old.dtype.kind=='f':row['max_abs']=float(np.max(np.abs(old.astype(np.float64)-new)))
     differences[key]=row
   if differences: current.setdefault('maps_decomp_differences',[]).append(dict(group=start,grid=grid,fields=differences))
 current.setdefault('maps_decomp_exact',[]).append(eq); return out
M.maps_batch=maps
origpool=S.run_pool
saved=False
def pool(phi,rhs,om,op,valid,coefs,**kw):
 global saved
 if not saved:
  ids=np.flatnonzero(valid)[:16]; ops=np.asarray(op)[ids]; unique,inv=np.unique(ops,return_inverse=True)
  np.savez_compressed(O/('native-lanes-'+a.tag+'.npz'),phi=np.asarray(phi)[ids],rhs=np.asarray(rhs)[ids],om=np.asarray(om)[ids],op=inv.astype(np.int32),**{k:np.asarray(v)[unique] for k,v in zip(('ex','ey','ez','dinv'),coefs)})
  saved=True
 t=time.perf_counter(); c0=ct.s; out=origpool(phi,rhs,om,op,valid,coefs,**kw); jax.block_until_ready(out); wall=max(0,time.perf_counter()-t-(ct.s-c0)); sweeps=int(np.asarray(out[1])[valid].sum()); current.setdefault('sor',[]).append(dict(seconds=wall,lane_sweeps=sweeps,seconds_per_lane_sweep=wall/max(1,sweeps),hashes=[digest(x) for x in out])); return out
S.run_pool=pool
recs=SP.triple_order(PG.load(BJ.INPUTS,CPU_IDS),True); gc={}; inst=prof2.Inst(ct,True); inst.install()
for rec in recs:
 current={}; times.clear(); calls.clear(); inst.begin(rec); t=time.perf_counter(); tl=[]; e=API.energies([rec],API.Config(backend='xla'),timing=tl,geo_cache=gc)[0]; current.update(wall=time.perf_counter()-t,timing=tl[0],production=inst.end(),energy_hashes={k:digest(getattr(e,k)) for k in ('g','W','occ','start','nst','pairs')},pb_hashes={k:digest(e.diag['pb'][k]) for k in ('esolvM','esolvS','esolvM0','sweeps','maxchange','rho')},maps_ops=dict(times),maps_op_calls=dict(calls))
 result['current']=dict(state=rec.state_id,**current); BJ.jdump(result,str(O/(a.tag+'.json')))
 # Original geometry decomposition, retaining f64 and original candidate order.
 oldpc=PG.pc; PG.pc=lambda:time.perf_counter()-ct.s
 try: gt,info,par,_=PG.geo_run(rec,rec.size_class,'f64',decomp=True)
 finally: PG.pc=oldpc
 current.update(geometry_ops=gt,geometry_info=info,geometry_parity=par,host_hwm_kb=BJ.status('VmHWM'))
 result['rows'][rec.state_id]=current; result.pop('current',None); cache.clear(); jax.clear_caches(); BJ.jdump(result,str(O/(a.tag+'.json'))); print(a.tag,rec.state_id,current['maps_ops'],flush=True)
result['completed']=True; BJ.jdump(result,str(O/(a.tag+'.json')))
