"""Per-step real-state timing and bitwise gates, independent arms, no default changes."""
import argparse,collections,json,os,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
p=argparse.ArgumentParser(); p.add_argument('--tag',required=True); p.add_argument('--pin',type=int); a=p.parse_args()
if a.pin: os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:a.pin]); os.environ['OMP_NUM_THREADS']=str(a.pin)
import numpy as np
import jax
import bench_jaxpb as BJ,prof_geomaps as PG,prof2,speed as SP
from jaxpb import maps as M,geometry as G,api as API,records as R,pb as P
from opt2 import CPU_IDS,digest
from opt4_arms import maps_arm,geometry_arm,pb_arm
O=Path(BJ.TP)/'opt4'; ct=BJ.CompileTime(); ma={i:maps_arm(i) for i in (1,2,3,5)}; ga={i:geometry_arm(i) for i in (1,2,4)}; rows={}; current={}; pa=pb_arm(); last_hwm=0
def diff(old,new):
 d=[]
 for x,y in zip(jax.tree_util.tree_leaves(old),jax.tree_util.tree_leaves(new)):
  x=np.asarray(x); y=np.asarray(y)
  if x.tobytes()!=y.tobytes():
   item=dict(shape=x.shape,dtype=str(x.dtype),different=int(np.count_nonzero(x!=y)))
   if x.dtype.kind=='f':item.update(max_abs=float(np.max(np.abs(x.astype(np.float64)-y))),max_relative=float(np.max(np.abs(x.astype(np.float64)-y)/np.maximum(np.abs(x),1e-30))))
   d.append(item)
 return d

warmed=set()
def measure(fn,*args,**kw):
 global last_hwm
 # Warm a static signature once, then time each real batch without redundant
 # warm execution. A compile event in a timed call forces a warm retry.
 signature=tuple((tuple(x.shape),str(x.dtype),getattr(x,'weak_type',False)) if hasattr(x,'shape') else repr(x) for x in jax.tree_util.tree_leaves((args,kw)))
 key=(id(fn),bool(jax.config.x64_enabled),signature)
 if key not in warmed:jax.block_until_ready(fn(*args,**kw));warmed.add(key)
 while True:
  BJ.reset_hwm(); c0=ct.s; t=time.perf_counter(); out=jax.block_until_ready(fn(*args,**kw)); elapsed=time.perf_counter()-t
  if ct.s==c0:break
 last_hwm=BJ.status('VmHWM'); return out,elapsed
origmap=M.maps_batch; origgeo=G.geometry_state; origsite=G.geometry_sites

def maps(*args,**kw):
 baseline,dt=measure(origmap,*args,**kw); row=dict(baseline_s=dt,baseline_hwm_kb=last_hwm,arms={})
 if not (O/'worker-inputs.npz').exists():
  vals={k:np.asarray(v) for k,v in zip(('pos','rad','probes','ac','sp','sr','spr'),args)}; vals['active']=np.asarray(kw['active']); vals.update({'sz_'+k:v for k,v in vars(kw['sz']).items()}); np.savez_compressed(O/'worker-inputs.npz',**vals)
 for i,m in ma.items():
  out,sec=measure(m.maps_batch,*args,sz=m.Sizes(**vars(kw['sz'])),active=kw['active']); changes=diff(baseline,out); row['arms'][str(i)]=dict(seconds=sec,speedup=dt/sec,bitwise=not changes,differences=changes,host_hwm_kb=last_hwm)
 print(a.tag,'maps',len(current.get('maps',[])),{k:(v['seconds'],v['bitwise']) for k,v in row['arms'].items()},flush=True); current.setdefault('maps',[]).append(row); return baseline

def geo(*args,**kw):
 baseline=origgeo(*args,**kw); row=dict(arms={}); t=time.perf_counter(); base2=origgeo(*args,**kw); jax.block_until_ready(base2.probes); row['baseline_s']=time.perf_counter()-t
 for i,m in ga.items():
  t=time.perf_counter(); out=m.geometry_state(*args,**kw); jax.block_until_ready(out.probes); warm=time.perf_counter()-t
  BJ.reset_hwm(); t=time.perf_counter(); out=m.geometry_state(*args,**kw); jax.block_until_ready(out.probes); sec=time.perf_counter()-t
  changes=diff(baseline.probes[:R.NP_C[baseline.c]],out.probes[:R.NP_C[out.c]])
  row['arms'][str(i)]=dict(seconds=sec,cold_s=warm,speedup=row['baseline_s']/sec,bitwise=not changes and baseline.flags==out.flags,differences=changes,flags_equal=baseline.flags==out.flags,host_hwm_kb=BJ.status('VmHWM'))
 print(a.tag,'geometry',{k:(v['seconds'],v['bitwise']) for k,v in row['arms'].items()},flush=True); current['geometry']=row; return baseline

def sites(*args,**kw):
 baseline,dt=measure(origsite,*args,**kw); row=dict(baseline_s=dt,baseline_hwm_kb=last_hwm,arms={})
 out,sec=measure(ga[4].geometry_sites,*args,**kw); changes=diff(baseline,out); row['arms']['3']=dict(seconds=sec,speedup=dt/sec,bitwise=not changes,differences=changes,host_hwm_kb=last_hwm)
 row['baseline_memory']=prof2.memana(origsite,args,kw); row['candidate_memory']=prof2.memana(ga[4].geometry_sites,args,kw)
 print(a.tag,'geometry_sites',sec,not changes,flush=True); current.setdefault('geometry_sites',[]).append(row); return baseline
origpost=P.post

def post(*args,**kw):
 baseline,dt=measure(origpost,*args,**kw); out,sec=measure(pa.post,*args,**kw); changes=diff(baseline,out)
 current.setdefault('post',[]).append(dict(baseline_s=dt,seconds=sec,speedup=dt/sec,bitwise=not changes,differences=changes)); return baseline
M.maps_batch=maps; G.geometry_state=geo; G.geometry_sites=sites; P.post=post
result=dict(tag=a.tag,env=BJ.env_info(),rows=rows,frozen=os.getcwd())
recs=SP.triple_order(PG.load(BJ.INPUTS,CPU_IDS),True); gc={}
for rec in recs:
 current={}; t=time.perf_counter(); API.energies([rec],API.Config(backend='xla'),geo_cache=gc); current.update(wall=time.perf_counter()-t,host_hwm_kb=BJ.status('VmHWM')); rows[rec.state_id]=current; BJ.jdump(result,str(O/(a.tag+'.json'))); print(a.tag,rec.state_id,current,flush=True); jax.clear_caches(); warmed.clear()
result['completed']=True; BJ.jdump(result,str(O/(a.tag+'.json')))
