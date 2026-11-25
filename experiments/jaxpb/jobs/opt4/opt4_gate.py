"""PB fingerprints, deterministic PB-only label shards, compile and RSS gates; Slurm only."""
import argparse, hashlib, json, os, time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID'), 'Coulson allocations only'
import numpy as np
import jax
import bench_jaxpb as BJ, prof2, prof_geomaps as PG, speed as SP
from opt2 import CPU_IDS, FIELDS, digest
from jaxpb import api as API, records as RC, maps as MP, geometry as G, sor as S
from jaxpb.compile_budget import CompileCounter
O=Path(BJ.TP)/'opt4'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--tag',required=True); p.add_argument('--selection',choices=('cpu','bench'),default='cpu'); p.add_argument('--passes',type=int,default=1); p.add_argument('--reference'); p.add_argument('--backend',default='xla'); p.add_argument('--resume'); p.add_argument('--repeat-cpu',action='store_true'); p.add_argument('--pin',type=int); a=p.parse_args()
 if a.pin:
  os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:a.pin]); os.environ['OMP_NUM_THREADS']=str(a.pin)
 ids=CPU_IDS if a.selection=='cpu' else BJ.STATES; recs=SP.triple_order(PG.load(BJ.INPUTS,ids),True)
 ref=json.load(open(a.reference)) if a.reference else None; ct=BJ.CompileTime(); inst=prof2.Inst(ct,jax.default_backend()=='cpu'); inst.install(); cap={}
 def capture(mod,name,kind,select=lambda x:x):
  f=getattr(mod,name)
  def w(*args,**kw):
   out=f(*args,**kw); cap.setdefault(kind,[]).append(jax.tree_util.tree_map(digest,select(out))); return out
  setattr(mod,name,w)
 capture(MP,'maps_batch','maps'); capture(G,'geometry_state','geometry',lambda x:(x.probes[:RC.NP_C[x.c]],x.flags)); capture(G,'geometry_sites','geometry_sites'); capture(S,'run_pool','phi'); capture(S,'run_pool_pallas','phi')
 # Flags are scalars in a pytree; digest includes shapes and dtype and exact bytes.
 solve=API.solve_pb
 def pb(*args,**kw):
  out=solve(*args,**kw); cap['pb']={k:digest(getattr(out[0],k)) for k in FIELDS}; return out
 API.solve_pb=pb
 result=dict(tag=a.tag,env=BJ.env_info(),passes=[],reference=a.reference,states=ids,frozen=os.getcwd(),label_mode='PB-only, MC omitted (untouched); all PB shard fields and NaN label placeholders hashed')
 energies={}
 resume=json.load(open(a.resume)) if a.resume else None
 with CompileCounter() as cc:
  for rep in range(a.passes):
   rows=dict(resume.get('current',{}).get('rows',{})) if resume and rep==0 else {}; gc={}; cstart=cc.n_compiled
   for r in recs:
    if rep and a.repeat_cpu and r.state_id not in CPU_IDS: continue
    if r.state_id in rows:
     cid,role=API.triple_role(r.state_id)
     if role=='AB' and any(API.triple_role(x.state_id)[0]==cid and API.triple_role(x.state_id)[1]!='AB' and x.state_id not in rows for x in recs):
      with G.x32(): gc[cid]=G.geometry_state(r.pos,r.rad,r.size_class,want_reuse=True).reuse
     continue
    cap.clear(); inst.begin(r); c0=cc.n_compiled; t=time.perf_counter(); tl=[]; e=API.energies([r],API.Config(backend=a.backend),timing=tl,geo_cache=gc)[0]; wall=time.perf_counter()-t; detail=inst.end()
    for k in ('g','W','occ','start','nst','pairs'): energies[r.state_id+'/'+k]=np.asarray(getattr(e,k))
    for k in ('pkint','e_back'): energies[r.state_id+'/'+k]=np.asarray(e.diag[k])
    for k in ('esolvM','esolvS','esolvM0','sweeps','maxchange','rho'): energies[r.state_id+'/'+k]=np.asarray(e.diag['pb'][k])
    np.savez_compressed(O/(a.tag+'-energies.npz'),**energies)
    fields=API._state_arrays(r.n_sites,e,None); shard=O/(a.tag+'-labels.npz'); API.write_shard(str(shard),[dict(state_id=r.state_id,alias=r.state_id,input='OPT4',status='ok',arrays=fields,diag={})],dict(task='OPT4'),{})
    with np.load(shard) as z: sh={k:digest(z[k]) for k in z.files if k not in ('provenance_json','timing_json')}
    hashes=json.loads(json.dumps(cap)); hashes['energy']={k:digest(getattr(e,k)) for k in ('g','W','occ','start','nst','pairs')}; hashes['labels']=sh
    checks={}
    if ref: checks['reference']=hashes==ref['passes'][0]['rows'][r.state_id]['hashes']
    if rep: checks['repeat']=hashes==result['passes'][0]['rows'][r.state_id]['hashes']
    rows[r.state_id]=dict(hashes=hashes,checks=checks,wall=wall,timing=tl[0],stages={k:max(0,v['t']-v['comp_s']) for k,v in detail['stages'].items()},detail=detail,compiles=cc.n_compiled-c0,host_hwm_kb=BJ.status('VmHWM'))
    result['current']=dict(rep=rep,rows=rows); BJ.jdump(result,str(O/(a.tag+'.json'))); print(a.tag,rep,r.state_id,round(wall,2),checks,flush=True)
   result['passes'].append(dict(rows=rows,new_compiles=cc.n_compiled-cstart)); result.pop('current',None); BJ.jdump(result,str(O/(a.tag+'.json')))
  result['compiles']=cc.summary()
 result['passed']=all(all(v['checks'].values()) for pp in result['passes'] for v in pp['rows'].values()); result['completed']=True; BJ.jdump(result,str(O/(a.tag+'.json')))
if __name__=='__main__': main()
