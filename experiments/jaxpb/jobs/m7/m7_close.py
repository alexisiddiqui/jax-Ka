"""All solves on bench20 (contains OPT2 five): two modes vs independent f64 CG."""
import argparse,os,time,json
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import numpy as np,jax
from jaxpb import api as A,pb as B,pcg as P,sor as S
import bench_jaxpb as BJ,prof_geomaps as PG,speed as SP
from opt2 import CPU_IDS
import m7_reference as REF
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--selection',choices=('bench','cpu'),default='bench');a=p.parse_args()
O=Path(BJ.TP)/'m7';ids=BJ.STATES if a.selection=='bench' else CPU_IDS;recs=SP.triple_order(PG.load(BJ.INPUTS,ids),True)
orig_setup1=B.setup1;orig_setup2=B.setup2;orig_p=P.run_pool;orig_x=S.run_pool;orig_s=S.run_pool_pallas
capture={};arm=[None];dphi={};residual=[0.]

def setup1(*args,**kw):
 out=orig_setup1(*args,**kw)
 if arm[0]=='delphi':capture['coefs']=orig_setup1(*args,backend='pcg')[1]
 else:capture['coefs']=out[1]
 capture['phase']=1
 return out

def setup2(*args,**kw):
 out=orig_setup2(*args,**kw)
 if arm[0]=='delphi':capture['coefs']=orig_setup2(*args,backend='pcg')[1]
 else:capture['coefs']=out[1]
 capture['phase']=2
 return out
B.setup1=setup1;B.setup2=setup2

def wrapper(fn):
 def run(phi,rhs,om,op,valid,coefs,stats=None,**kw):
  if arm[0]=='reference':
   out=REF.pool(phi,rhs,om,op,valid,coefs,stats=stats);residual[0]=max(residual[0],out[4]);return out[:3]
  out=fn(phi,rhs,om,op,valid,coefs,stats=stats,**kw)
  ref=REF.pool(phi,rhs,om,op,valid,capture['coefs'],compare=(out[0],capture['phase']))
  residual[0]=max(residual[0],ref[4])
  for k,v in ref[3].items():dphi[k]=max(dphi.get(k,0.),v)
  return out
 return run
P.run_pool=wrapper(orig_p);S.run_pool=wrapper(orig_x);S.run_pool_pallas=wrapper(orig_s)
result=dict(env=BJ.env_info(),frozen=os.getcwd(),states=ids,rows={},reference='independent unpreconditioned JAX SciPy CG, f64, L2 rtol 1e-13/atol 1e-12; sliced interior stencil')
for r in recs:
 E={};row={}
 for name in ('reference','converged','delphi'):
  arm[0]=name;dphi.clear();residual[0]=0.;t=time.perf_counter();tim=[]
  e=A.energies([r],A.Config(pb_mode='delphi' if name=='delphi' else 'converged'),timing=tim)[0];E[name]=e
  row[name]=dict(seconds=time.perf_counter()-t,timing=tim[0],max_dphi_same_boundary=dict(dphi),reference_residual_max=residual[0],flags=e.diag['pb']['flags'])
  print(a.tag,r.state_id,name,round(row[name]['seconds'],2),dphi,flush=True)
 ref=E['reference']
 for name in ('converged','delphi'):
  e=E[name];row[name]['errors']={k:dict(max=float(np.max(np.abs(np.asarray(e.diag['pb'][k],np.float64)-ref.diag['pb'][k]))),mae=float(np.mean(np.abs(np.asarray(e.diag['pb'][k],np.float64)-ref.diag['pb'][k])))) for k in ('esolvM0','esolvM','esolvS')}
  row[name]['errors'].update({k:dict(max=float(np.max(np.abs(np.asarray(getattr(e,k),np.float64)-getattr(ref,k)))),mae=float(np.mean(np.abs(np.asarray(getattr(e,k),np.float64)-getattr(ref,k))))) for k in ('g','W')})
 result['rows'][r.state_id]=row;BJ.jdump(result,str(O/(a.tag+'.json')))
result['completed']=True;BJ.jdump(result,str(O/(a.tag+'.json')))
