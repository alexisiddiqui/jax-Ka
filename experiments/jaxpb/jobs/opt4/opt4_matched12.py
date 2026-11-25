"""Matched original/retained CPU maps timings on all 15 saved real CPU5 batches."""
import argparse,ctypes,gc,json,os,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import jax,numpy as np
import bench_jaxpb as BJ,prof_geomaps as PG,speed as SP
from jaxpb import maps as M,geometry as G
from opt2 import CPU_IDS
from opt4_arms import maps_arm
p=argparse.ArgumentParser();p.add_argument('--tag',default='matched12');a=p.parse_args()
O=Path(BJ.TP)/'opt4';fixtures=json.load(open(O/'fixtures.json'));assert fixtures['completed']
arm=maps_arm(5);query=arm._compact_query
arm._compact_query=lambda q,P,w,table,lo,ncell,cell,offs:query(q,P,w,table,lo,ncell,cell,offs,slab=32)
ct=BJ.CompileTime();warmed=set();result=dict(env=BJ.env_info(),rows={},method=__doc__,arm='occupied slots, slab32, query8192/B capped4096')
def measure(fn,*args,**kw):
 key=(id(fn),tuple((tuple(x.shape),str(x.dtype),getattr(x,'weak_type',False)) if hasattr(x,'shape') else repr(x) for x in jax.tree_util.tree_leaves((args,kw))))
 if key not in warmed:jax.block_until_ready(fn(*args,**kw));warmed.add(key)
 for _ in range(3):
  BJ.reset_hwm();c0=ct.s;t=time.perf_counter();out=jax.block_until_ready(fn(*args,**kw));dt=time.perf_counter()-t
  if ct.s==c0:return out,dt,BJ.status('VmHWM')
 raise RuntimeError('compilation persisted in measured call')
for rec in SP.triple_order(PG.load(BJ.INPUTS,CPU_IDS),True):
 row=[];result['rows'][rec.state_id]=row
 with np.load(fixtures['rows'][rec.state_id]['file']) as z,G.x32():
  for b,md in enumerate(fixtures['rows'][rec.state_id]['batches']):
   args=tuple(jax.device_put(z[f'{b}/{k}']) for k in ('pos','rad','probes','ac','sp','sr','spr'));active=jax.device_put(z[f'{b}/active'])
   old,base,brss=measure(M.maps_batch,*args,sz=M.Sizes(**md['sizes']),active=active)
   new,secs,rss=measure(arm.maps_batch,*args,sz=arm.Sizes(**md['sizes']),active=active)
   changes=[]
   for x,y in zip(jax.tree_util.tree_leaves(old),jax.tree_util.tree_leaves(new)):
    x=np.asarray(x);y=np.asarray(y)
    if x.tobytes()!=y.tobytes():changes.append(dict(shape=x.shape,dtype=str(x.dtype),different=int(np.count_nonzero(x!=y))))
   r=dict(baseline_s=base,selected_s=secs,speedup=base/secs,bitwise=not changes,differences=changes,baseline_hwm_kb=brss,selected_hwm_kb=rss);row.append(r);BJ.jdump(result,str(O/(a.tag+'.json')));print(rec.state_id,b,r,flush=True)
 warmed.clear();jax.clear_caches();gc.collect();ctypes.CDLL(None).malloc_trim(0)
result['completed']=True;result['passed']=all(r['bitwise'] for rows in result['rows'].values() for r in rows);BJ.jdump(result,str(O/(a.tag+'.json')))
