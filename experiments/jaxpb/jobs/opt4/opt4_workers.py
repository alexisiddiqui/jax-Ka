"""Two persistent, pinned three-physical-core maps workers. Standalone trial."""
import os,time,json,argparse,concurrent.futures,multiprocessing
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
T='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'; O=Path(T)/'opt4'

def worker(index,cpus,arm,slab):
 os.sched_setaffinity(0,cpus); os.environ['OMP_NUM_THREADS']=str(len(cpus))
 import numpy as np,jax
 from jaxpb import maps as M,geometry as G
 from opt4_arms import maps_arm
 from opt4_kernels import compact_query
 import bench_jaxpb as BJ
 global INPUT,FUNCTION,SIZES,CC
 from jaxpb.compile_budget import CompileCounter
 CC=CompileCounter();CC.__enter__()
 z=np.load(O/'worker-inputs.npz'); INPUT={k:(jax.device_put(z[k]) if k in ('pos','rad','probes') else np.asarray(z[k])) for k in ('pos','rad','probes','ac','sp','sr','spr','active')}; mod=maps_arm(arm); mod._compact_query=lambda *args:compact_query(*args,slab=slab); SIZES=mod.Sizes(**{k[3:]:int(z[k]) for k in z.files if k.startswith('sz_')}); FUNCTION=mod.maps_batch

def initialize(groups,arm,slab):
 worker(0,groups[(multiprocessing.current_process()._identity[0]-1)%len(groups)],arm,slab)

def task(start):
 mask=False
 if isinstance(start,tuple):start,mask=start
 import numpy as np,jax
 import bench_jaxpb as BJ
 v=INPUT; sl=slice(start,start+4); live=v['active'][sl]; c0=CC.n_compiled
 if mask and start>=len(v['active'])-4:live=np.zeros_like(live)
 with __import__('jaxpb.geometry',fromlist=['x32']).x32(): out=FUNCTION(v['pos'],v['rad'],v['probes'],v['ac'][sl],v['sp'][sl],v['sr'][sl],v['spr'][sl],sz=SIZES,active=live)
 return jax.tree_util.tree_map(np.asarray,out),BJ.status('VmHWM'),os.getpid(),CC.n_compiled-c0

def main():
 p=argparse.ArgumentParser(); p.add_argument("--arm",type=int,required=True); p.add_argument("--tag",required=True); p.add_argument("--slab",type=int,choices=(32,128),required=True); a=p.parse_args()
 import numpy as np,jax
 import bench_jaxpb as BJ
 from jaxpb import geometry as G
 from opt4_arms import maps_arm
 from opt4_kernels import compact_query
 # Slurm HT pairs: choose six distinct physical cores from this allocation.
 cores={}
 for cpu in sorted(os.sched_getaffinity(0)):
  top=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology'); key=(top.joinpath('physical_package_id').read_text().strip(),top.joinpath('core_id').read_text().strip()); cores.setdefault(key,cpu)
 assert len(cores)>=6,cores
 cpus=list(cores.values())[:6]; groups=[cpus[:3],cpus[3:]]
 z=np.load(O/'worker-inputs.npz'); mod=maps_arm(a.arm); mod._compact_query=lambda *args:compact_query(*args,slab=a.slab); sz=mod.Sizes(**{k[3:]:int(z[k]) for k in z.files if k.startswith('sz_')}); args=tuple(jax.device_put(z[k]) for k in ('pos','rad','probes','ac','sp','sr','spr')); active=jax.device_put(z['active'])
 with G.x32():
  jax.block_until_ready(mod.maps_batch(*args,sz=sz,active=active)); t=time.perf_counter(); baseline=jax.tree_util.tree_map(np.asarray,jax.block_until_ready(mod.maps_batch(*args,sz=sz,active=active))); base=time.perf_counter()-t
  result=dict(env=BJ.env_info(),groups=groups,arm=a.arm,slab=a.slab,baseline_s=base,runs=[])
  with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(groups,a.arm,a.slab)) as pool:
   for rep in range(3):
    t=time.perf_counter(); outs=list(pool.map(task,range(0,len(active),4))); elapsed=time.perf_counter()-t
    combined=jax.tree_util.tree_map(lambda *xs:np.concatenate(xs,axis=0),*[x[0] for x in outs]); exact=all(x.tobytes()==y.tobytes() for x,y in zip(jax.tree_util.tree_leaves(baseline),jax.tree_util.tree_leaves(combined)))
    differences=[]
    for field,(x,y) in enumerate(zip(jax.tree_util.tree_leaves(baseline),jax.tree_util.tree_leaves(combined))):
     if x.tobytes()!=y.tobytes(): differences.append(dict(field=field,shape=x.shape,dtype=str(x.dtype),different=int(np.count_nonzero(x!=y)),max_abs=float(np.max(np.abs(x.astype(np.float64)-y))) if x.dtype.kind in 'fi' else None))
    rss={pid:max(x[1] for x in outs if x[2]==pid) for _,_,pid,_ in outs}; result['runs'].append(dict(seconds=elapsed,speedup=base/elapsed,bitwise=exact,differences=differences,new_worker_compiles=sum(x[3] for x in outs),worker_hwm_kb=rss,aggregate_hwm_upper_bound_kb=BJ.status('VmHWM')+sum(rss.values()))); BJ.jdump(result,str(O/(a.tag+'.json')))
   # A fully inactive last group must reproduce maps_batch's empty branch.
   inactive=active.at[-4:].set(False)
   masked=jax.tree_util.tree_map(np.asarray,jax.block_until_ready(mod.maps_batch(*args,sz=sz,active=inactive)))
   outs=list(pool.map(task,[(i,True) for i in range(0,len(active),4)]))
   combined=jax.tree_util.tree_map(lambda *xs:np.concatenate(xs,axis=0),*[x[0] for x in outs])
   result['inactive_group_bitwise']=all(x.tobytes()==y.tobytes() for x,y in zip(jax.tree_util.tree_leaves(masked),jax.tree_util.tree_leaves(combined)))
  result['completed']=True; result['passed']=all(r['bitwise'] for r in result['runs']) and result['inactive_group_bitwise']; BJ.jdump(result,str(O/(a.tag+'.json')))
if __name__=='__main__':main()
