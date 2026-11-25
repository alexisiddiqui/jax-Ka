"""Matched independent CPU arms on saved real inputs; no full PB intermediates."""
import os,argparse,subprocess,sys,json,time,gc,ctypes
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
p=argparse.ArgumentParser();p.add_argument('--mode',choices=('driver','maps','sites'),default='driver');p.add_argument('--state');p.add_argument('--resume',action='store_true');p.add_argument('--arms',default='1,2,3,5');a=p.parse_args()
T=Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline');O=T/'opt4'
if a.mode=='driver':
 for mode in ('maps','sites'):
  cmd=[sys.executable,__file__,'--mode',mode,'--arms',a.arms]+(['--resume'] if a.resume else [])
  subprocess.run(cmd,check=True)
 parts=[json.loads((O/('ab5-'+mode+'.json')).read_text()) for mode in ('maps','sites')]
 rows=parts[0]['rows']
 for sid,r in parts[1]['rows'].items():rows[sid].update(r)
 result=dict(tag='ab5',env=parts[0]['env'],rows=rows,completed=True,method='Separate maps/whole geometry and model-site processes; matched warm timings on saved real inputs. Source physics unchanged.')
 (O/'ab5.json').write_text(json.dumps(result,indent=2)+'\n');sys.exit(0)
import numpy as np,jax
import bench_jaxpb as BJ,prof_geomaps as PG,speed as SP
from opt2 import CPU_IDS
from jaxpb import maps as M,geometry as G,api as A,records as R
from opt4_arms import maps_arm,geometry_arm
ct=BJ.CompileTime();warmed=set();last_hwm=0
fixtures=json.loads((O/'fixtures.json').read_text());assert fixtures['completed']
def diff(old,new):
 d=[]
 for x,y in zip(jax.tree_util.tree_leaves(old),jax.tree_util.tree_leaves(new)):
  x=np.asarray(x);y=np.asarray(y)
  if x.tobytes()!=y.tobytes():
   v=dict(shape=x.shape,dtype=str(x.dtype),different=int(np.count_nonzero(x!=y)))
   if x.dtype.kind=='f':v['max_abs']=float(np.max(np.abs(x.astype(np.float64)-y)))
   d.append(v)
 return d

def measure(fn,*args,**kw):
 global last_hwm
 sig=tuple((tuple(x.shape),str(x.dtype),getattr(x,'weak_type',False)) if hasattr(x,'shape') else repr(x) for x in jax.tree_util.tree_leaves((args,kw)));key=(id(fn),bool(jax.config.x64_enabled),sig)
 if key not in warmed:jax.block_until_ready(fn(*args,**kw));warmed.add(key)
 for _ in range(3):
  BJ.reset_hwm();c0=ct.s;t=time.perf_counter();out=jax.block_until_ready(fn(*args,**kw));dt=time.perf_counter()-t
  if ct.s==c0:break
 else:raise RuntimeError('repeated compilation in timed call')
 last_hwm=BJ.status('VmHWM');return out,dt

def trim():
 warmed.clear();jax.clear_caches();gc.collect();ctypes.CDLL(None).malloc_trim(0)
path=O/('ab5-'+a.mode+'.json')
prior=json.loads(path.read_text()) if a.resume and path.exists() else None
rows=prior['rows'] if prior else {};result=prior or dict(env=BJ.env_info(),rows=rows,mode=a.mode)
if prior:result.setdefault('resume_envs',[]).append(BJ.env_info())
cache={}
ma={int(i):maps_arm(int(i)) for i in a.arms.split(',')} if a.mode=='maps' else {};ga={i:geometry_arm(i) for i in (1,2,4)} if a.mode=='maps' else {4:geometry_arm(4)}
for rec in SP.triple_order(PG.load(BJ.INPUTS,CPU_IDS),True):
 row=rows.setdefault(rec.state_id,{});already=len(row.get('maps' if a.mode=='maps' else 'geometry_sites',[]));z=np.load(fixtures['rows'][rec.state_id]['file'])
 if a.mode=='maps' and 'geometry' not in row:
  cid,role=A.triple_role(rec.state_id);kwargs=dict(parent=cache.get(cid) if role!='AB' else None,want_reuse=role=='AB')
  old=G.geometry_state(rec.pos,rec.rad,rec.size_class,**kwargs);t=time.perf_counter();G.geometry_state(rec.pos,rec.rad,rec.size_class,**kwargs);dt=time.perf_counter()-t
  rr=dict(baseline_s=dt,arms={});row['geometry']=rr
  for i,m in ga.items():
   m.geometry_state(rec.pos,rec.rad,rec.size_class,**kwargs);BJ.reset_hwm();t=time.perf_counter();new=m.geometry_state(rec.pos,rec.rad,rec.size_class,**kwargs);sec=time.perf_counter()-t
   changes=diff(old.probes[:R.NP_C[old.c]],new.probes[:R.NP_C[new.c]])
   rr['arms'][str(i)]=dict(seconds=sec,speedup=dt/sec,bitwise=not changes and old.flags==new.flags,differences=changes,flags_equal=old.flags==new.flags,host_hwm_kb=BJ.status('VmHWM'))
  if role=='AB':cache[cid]=old.reuse
  print(rec.state_id,'geometry',rr,flush=True);row['maps']=[]
 for b,md in enumerate(fixtures['rows'][rec.state_id]['batches']):
  if b<already:continue
  if a.mode=='maps':
   args=tuple(jax.device_put(z[f'{b}/{k}']) for k in ('pos','rad','probes','ac','sp','sr','spr'));active=jax.device_put(z[f'{b}/active'])
   with G.x32():
    old,dt=measure(M.maps_batch,*args,sz=M.Sizes(**md['sizes']),active=active);rr=dict(baseline_s=dt,baseline_hwm_kb=last_hwm,arms={});row['maps'].append(rr)
    if not (O/'worker-inputs.npz').exists():
     vals={k:np.asarray(v) for k,v in zip(('pos','rad','probes','ac','sp','sr','spr'),args)};vals['active']=np.asarray(active);vals.update({'sz_'+k:v for k,v in md['sizes'].items()});np.savez_compressed(O/'worker-inputs.npz',**vals)
    for i,m in ma.items():
     out,sec=measure(m.maps_batch,*args,sz=m.Sizes(**md['sizes']),active=active);changes=diff(old,out);rr['arms'][str(i)]=dict(seconds=sec,speedup=dt/sec,bitwise=not changes,differences=changes,host_hwm_kb=last_hwm)
    print(rec.state_id,'maps',b,rr,flush=True)
  else:
   import prof2
   with G.x64():
    args=tuple(jax.device_put(z[f'{b}/site{j}']) for j in range(4));kw=md['site_kw'];old,dt=measure(G.geometry_sites,*args,**kw);rr=dict(baseline_s=dt,baseline_hwm_kb=last_hwm,arms={})
    new,sec=measure(ga[4].geometry_sites,*args,**kw);changes=diff(old,new);rr['arms']['3']=dict(seconds=sec,speedup=dt/sec,bitwise=not changes,differences=changes,host_hwm_kb=last_hwm)
    if b==0:rr.update(baseline_memory=prof2.memana(G.geometry_sites,args,kw),candidate_memory=prof2.memana(ga[4].geometry_sites,args,kw))
   row.setdefault('geometry_sites',[]).append(rr);print(rec.state_id,'sites',b,rr,flush=True)
  BJ.jdump(result,str(O/('ab5-'+a.mode+'.json')))
 z.close();trim();BJ.jdump(result,str(O/('ab5-'+a.mode+'.json')))
result['completed']=True;BJ.jdump(result,str(O/('ab5-'+a.mode+'.json')))
