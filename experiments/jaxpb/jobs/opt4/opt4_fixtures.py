"""Save real CPU-arm inputs using unchanged PB on the allowed A40 allocation."""
import os,json
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import numpy as np,jax
import bench_jaxpb as BJ,prof_geomaps as PG,speed as SP
from opt2 import CPU_IDS
from jaxpb import api as A,maps as M,geometry as G
O=Path(BJ.TP)/'opt4';origmap=M.maps_batch;origsite=G.geometry_sites
rows={};data={};lastsite=None;meta=[]
def site(*args,**kw):
 global lastsite
 lastsite=(tuple(np.asarray(x) for x in args),dict(kw));return origsite(*args,**kw)
def maps(*args,**kw):
 i=len(meta)
 for k,x in zip(('pos','rad','probes','ac','sp','sr','spr','active'),args+(kw['active'],)):data[f'{i}/{k}']=np.asarray(x)
 for j,x in enumerate(lastsite[0]):data[f'{i}/site{j}']=x
 meta.append(dict(sizes=vars(kw['sz']),site_kw=lastsite[1]))
 return origmap(*args,**kw)
G.geometry_sites=site;M.maps_batch=maps
result=dict(job=os.environ['SLURM_JOB_ID'],env=BJ.env_info(),rows=rows,source='unchanged bdffc91; all actual batches and XL attempts')
gc={}
for r in SP.triple_order(PG.load(BJ.INPUTS,CPU_IDS),True):
 data={};meta=[];A.energies([r],A.Config(backend='pallas'),geo_cache=gc)
 path=O/('fixture-'+r.state_id+'.npz');np.savez_compressed(path,**data)
 rows[r.state_id]=dict(file=str(path),batches=meta);BJ.jdump(result,str(O/'fixtures.json'));print('fixture',r.state_id,len(meta),flush=True)
result['completed']=True;BJ.jdump(result,str(O/'fixtures.json'))
