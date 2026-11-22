"""Compile-only memory comparison of every GPU SOR width and the original W256 block."""
import importlib.util,json,os
assert os.environ.get('SLURM_JOB_ID')
import jax
import jax.numpy as jnp
from jaxpb import sor as SOR
T='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'
shape=jax.ShapeDtypeStruct;n=81;E=32;m=(n**3+1)//2
rows=[]
if jax.default_backend() == 'cpu':
 for w in (4,8,16):
  args=(shape((w,n,n,n),jnp.float32),)*2+(shape((w,),jnp.int32),shape((w,),jnp.float32),shape((w,),jnp.bool_),shape((w,),jnp.float32),shape((w,),jnp.int32))+(shape((E,n,n,n),jnp.float32),)*4
  ma=SOR.device_solve.lower(*args,n=n,widths=tuple(x for x in (4,8,16) if x<=w)).compile().memory_analysis()
  rows.append(dict(variant='OPT2',width=w,**{k:int(getattr(ma,k)) for k in ('argument_size_in_bytes','output_size_in_bytes','alias_size_in_bytes','temp_size_in_bytes')}))
else:
 for w in (16,64,128,256):
  args=(shape((w,2,m),jnp.float32),)*2+(shape((w,),jnp.int32),shape((w,),jnp.float32),shape((w,),jnp.bool_),shape((w,),jnp.float32),shape((w,),jnp.int32),shape((E,n**3),jnp.uint8),shape((E,16),jnp.float32))
  ma=SOR.device_solve.lower(*args,n=n,widths=tuple(x for x in (16,64,128,256) if x<=w)).compile().memory_analysis()
  rows.append(dict(variant='OPT2',width=w,**{k:int(getattr(ma,k)) for k in ('argument_size_in_bytes','output_size_in_bytes','alias_size_in_bytes','temp_size_in_bytes')}))
spec=importlib.util.spec_from_file_location('jaxpb._opt2_baseline_sor',T+'/opt1b-s5-20261122/src/jaxpb/sor.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
w=256
args=(shape((w,n,n,n),jnp.float32),)*2+(shape((w,),jnp.int32),shape((w,),jnp.float32),shape((w,),jnp.bool_),shape((w,),jnp.float32),shape((w,),jnp.int32))+(shape((E,n,n,n),jnp.float32),)*4+(shape((n,n,n),jnp.bool_),)*2
ma=old.sor_block.lower(*args).compile().memory_analysis()
rows.append(dict(variant='10a3ddc/OPT1B',width=w,**{k:int(getattr(ma,k)) for k in ('argument_size_in_bytes','output_size_in_bytes','alias_size_in_bytes','temp_size_in_bytes')}))
res=dict(job=os.environ['SLURM_JOB_ID'],frozen=os.getcwd(),reference=T+'/opt1b-s5-20261122',rows=rows,backend=jax.default_backend(),compute='compile-only Slurm allocation; excluded from numerical compile gates')
with open(T+'/opt2/sor-buffer-sizes-'+jax.default_backend()+'.json','w') as f:json.dump(res,f,indent=2)
print(json.dumps(res),flush=True)
