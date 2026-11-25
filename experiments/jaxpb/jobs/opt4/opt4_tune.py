"""Tune occupied slabs and conservative inverse-reach cell pruning on real batches."""
import os,json,time,gc,ctypes,runpy
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import numpy as np,jax
import bench_jaxpb as BJ
from jaxpb import geometry as G,maps as M
from opt4_arms import maps_arm
from opt4_kernels import compact_query
O=Path(BJ.TP)/'opt4';ns=runpy.run_path('tests/test_opt4.py')
for key in ('test_compact_queries','test_reverse_reach'):
 ns[key]();print(key,'PASS',flush=True)
fixtures=json.loads((O/'fixtures.json').read_text());assert fixtures['completed']
# Two representative shapes; full CPU5/CPU20 production gates follow selection.
ids=['4066edfcdb3ea8e3_AB','4de4ae106157408f_A']
mods={5:maps_arm(5),6:maps_arm(3),7:maps_arm(5)}
mods[7]._compact_query=lambda *args:compact_query(*args,slab=32)
ct=BJ.CompileTime();warmed=set();rows={};result=dict(env=BJ.env_info(),rows=rows,units_passed=True,arms={'5':'occupied128, wide chunk, original reach','6':'occupied128, wide chunk, cell-by-cell pruned inverse reach','7':'occupied32, wide chunk, original reach'},source='real saved A40 inputs; matched CPU execution')
def measure(fn,args,kw):
 sig=(id(fn),repr([(x.shape,str(x.dtype)) for x in args]),repr(kw['sz']))
 if sig not in warmed:jax.block_until_ready(fn(*args,**kw));warmed.add(sig)
 for _ in range(3):
  BJ.reset_hwm();c0=ct.s;t=time.perf_counter();out=jax.block_until_ready(fn(*args,**kw));dt=time.perf_counter()-t
  if c0==ct.s:return out,dt,BJ.status('VmHWM')
 raise RuntimeError('timed call kept compiling')
for sid in ids:
 z=np.load(fixtures['rows'][sid]['file']);rr=[];rows[sid]=rr
 # First full batch per state is enough for this tuning decision.
 for b,md in enumerate(fixtures['rows'][sid]['batches'][:1]):
  args=tuple(jax.device_put(z[f'{b}/{k}']) for k in ('pos','rad','probes','ac','sp','sr','spr'));active=jax.device_put(z[f'{b}/active'])
  with G.x32():
   old,dt,rss=measure(M.maps_batch,args,dict(sz=M.Sizes(**md['sizes']),active=active));row=dict(baseline_s=dt,baseline_hwm_kb=rss,arms={});rr.append(row)
   for arm,m in mods.items():
    new,sec,rss=measure(m.maps_batch,args,dict(sz=m.Sizes(**md['sizes']),active=active));differences=[]
    for field,(x,y) in enumerate(zip(jax.tree_util.tree_leaves(old),jax.tree_util.tree_leaves(new))):
     x=np.asarray(x);y=np.asarray(y)
     if x.tobytes()!=y.tobytes():differences.append(dict(field=field,shape=x.shape,dtype=str(x.dtype),different=int(np.count_nonzero(x!=y)),max_abs=float(np.max(np.abs(x.astype(np.float64)-y)))))
    row['arms'][str(arm)]=dict(seconds=sec,speedup=dt/sec,bitwise=not differences,differences=differences,host_hwm_kb=rss);BJ.jdump(result,str(O/'tune5.json'));print(sid,arm,row['arms'][str(arm)],flush=True)
 z.close();warmed.clear();jax.clear_caches();gc.collect();ctypes.CDLL(None).malloc_trim(0)
result['completed']=True;BJ.jdump(result,str(O/'tune5.json'))
