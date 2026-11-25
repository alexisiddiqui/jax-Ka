"""Energy, determinism and compile gates on real states; Slurm only."""
import argparse,json,os,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import numpy as np
import jax
from jaxpb import api as A,records as R
from jaxpb.compile_budget import CompileCounter,cache_open,cache_close
import bench_jaxpb as BJ,prof_geomaps as PG,speed as SP
from opt2 import CPU_IDS,digest
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--selection',choices=('bench','all','cpu'),default='bench');p.add_argument('--mode',default='delphi');p.add_argument('--passes',type=int,default=2);p.add_argument('--reference');p.add_argument('--cache');a=p.parse_args()
O=Path(BJ.TP)/'m7'; cfg=A.Config(pb_mode=a.mode)
plan=json.load(open(BJ.TP+'/a40/m6/m6/runA/plan.json'));allids=[it['state_id'] for sh in plan['shards'] for it in sh]
ids={'bench':BJ.STATES,'all':allids,'cpu':CPU_IDS}[a.selection];recs=SP.triple_order(PG.load(BJ.INPUTS,ids),True)
assert len(recs)==len(ids)
cache=cache_open(a.cache) if a.cache else None
result=dict(env=BJ.env_info(),frozen=os.getcwd(),cfg=A.cfg_dict(cfg),code_sha=A.code_sha(),states=ids,passes=[],reference=a.reference)
ref=json.load(open(a.reference)) if a.reference else None
arr={}
with CompileCounter() as cc:
 for rep in range(a.passes):
  rows={};gc={};cstart=cc.n_compiled
  for r in recs:
   t=time.perf_counter();tl=[];c0=cc.n_compiled
   e=A.energies([r],cfg,timing=tl,geo_cache=gc)[0]
   pb=e.diag['pb']; fields={k:np.asarray(getattr(e,k)) for k in ('g','W','occ','start','nst','pairs')}
   fields.update({k:np.asarray(e.diag[k]) for k in ('pkint','e_back')});fields.update({k:np.asarray(pb[k]) for k in ('esolvM','esolvS','esolvM0','sweeps','maxchange','rho')})
   hashes={k:digest(v) for k,v in fields.items()};hashes['sitpotM']=pb['sitpot_sha']
   checks={}
   if ref: checks['reference']=hashes==ref['passes'][0]['rows'][r.state_id]['hashes']
   if rep: checks['repeat']=hashes==result['passes'][0]['rows'][r.state_id]['hashes']
   rows[r.state_id]=dict(hashes=hashes,checks=checks,seconds=time.perf_counter()-t,timing=tl[0],compiles=cc.n_compiled-c0,flags=pb['flags'],iterations={name:dict(mean=float(fields['sweeps'][:,k].mean()),max=int(fields['sweeps'][:,k].max()),total=int(fields['sweeps'][:,k].sum())) for k,name in enumerate(('coarse','focused','model'))},host_rss_kb=BJ.status('VmRSS'),gpu=A.gpu_mem())
   if rep==0:
    for k,v in fields.items():arr[r.state_id+'/'+k]=v
   np.savez_compressed(O/(a.tag+'-energies.npz'),**arr)
   result['current']=dict(rep=rep,rows=rows);BJ.jdump(result,str(O/(a.tag+'.json')))
   print(a.tag,rep,r.state_id,round(rows[r.state_id]['seconds'],2),checks,flush=True)
  result['passes'].append(dict(rows=rows,new_compiles=cc.n_compiled-cstart));result.pop('current',None);BJ.jdump(result,str(O/(a.tag+'.json')))
 result['compiles']=cc.summary()
if cache:result['cache_close']=cache_close(a.cache,cache['local_dir'])
result['completed']=True;result['passed']=all(all(row['checks'].values()) for pp in result['passes'] for row in pp['rows'].values())
BJ.jdump(result,str(O/(a.tag+'.json')))

if not result['passed']:
 raise SystemExit('M7 energy/determinism gate failed; see the JSON checks')
