"""One real model-site batch per CPU5 state, independently varied burial tiles."""
import os,inspect,json,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import numpy as np,jax,jax.numpy as jnp
import bench_jaxpb as BJ,prof2
from opt2 import CPU_IDS
from jaxpb import geometry as G
from opt4_arms import geometry_arm,site_burial
O=Path(BJ.TP)/'opt4';fixtures=json.loads((O/'fixtures.json').read_text());assert fixtures['completed']
mods={};source=inspect.getsource(site_burial)
for chunk in (4096,16384,65536):
 ns=dict(jax=jax,jnp=jnp,G=G);exec(source.replace('ch=4096',f'ch={chunk}'),ns)
 m=geometry_arm(4);m._site_burial=ns['site_burial'];mods[chunk]=m
rows={};warmed=set();ct=BJ.CompileTime();result=dict(env=BJ.env_info(),rows=rows,source='exact captured f64 geometry_sites inputs; one full batch from each CPU5 state')
def measure(fn,args,kw):
 key=(id(fn),repr([(x.shape,str(x.dtype)) for x in args]),repr(kw))
 if key not in warmed:jax.block_until_ready(fn(*args,**kw));warmed.add(key)
 for _ in range(3):
  BJ.reset_hwm();c0=ct.s;t=time.perf_counter();out=jax.block_until_ready(fn(*args,**kw));dt=time.perf_counter()-t
  if ct.s==c0:return out,dt,BJ.status('VmHWM')
 raise RuntimeError('timed call kept compiling')
for sid in CPU_IDS:
 md=fixtures['rows'][sid]['batches'][0];z=np.load(fixtures['rows'][sid]['file'])
 with G.x64():
  args=tuple(jax.device_put(z[f'0/site{i}']) for i in range(4));kw=md['site_kw'];old,dt,rss=measure(G.geometry_sites,args,kw)
  row=dict(baseline_s=dt,baseline_hwm_kb=rss,arms={});rows[sid]=row
  for chunk,m in mods.items():
   new,sec,rss=measure(m.geometry_sites,args,kw);differences=[]
   for field,(x,y) in enumerate(zip(old,new)):
    x=np.asarray(x);y=np.asarray(y)
    if x.tobytes()!=y.tobytes():differences.append(dict(field=field,shape=x.shape,dtype=str(x.dtype),different=int(np.count_nonzero(x!=y)),max_abs=float(np.max(np.abs(x.astype(np.float64)-y)))))
   row['arms'][str(chunk)]=dict(seconds=sec,speedup=dt/sec,bitwise=not differences,differences=differences,host_hwm_kb=rss)
   if sid==CPU_IDS[0]:row['arms'][str(chunk)]['memory_analysis']=prof2.memana(m.geometry_sites,args,kw)
   BJ.jdump(result,str(O/'site-tune5.json'));print(sid,chunk,row['arms'][str(chunk)],flush=True)
 z.close()
result['completed']=True;BJ.jdump(result,str(O/'site-tune5.json'))
