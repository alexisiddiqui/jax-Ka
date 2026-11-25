"""L3 solver effect with common RNG, two seeds and the independent PyPKA teacher."""
import os,json,sys,time,argparse
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import numpy as np,jax
from jaxpb import api as A,records as R
import bench_jaxpb as BJ,prof_geomaps as PG
O=Path(BJ.TP)/'m7'
p=argparse.ArgumentParser();p.add_argument('--converged-prefix',default='converged-final');args=p.parse_args()
plan=json.load(open(BJ.TP+'/a40/m6/m6/runA/plan.json'));ids=[it['state_id'] for sh in plan['shards'] for it in sh]
recs={r.state_id:r for path in BJ.INPUTS for r in R.load_states(path,only=[sid for sid in R.list_states(path) if sid in ids],with_ref=True)}
mfs=Path(BJ.TP).parent/'mf-surrogate-v1';sys.path.insert(0,str(mfs/'code'));import common as C

def stats(x):
 x=np.asarray(x,np.float64);x=x[np.isfinite(x)]
 return dict(n=len(x),mae=float(x.mean()) if len(x) else None,max=float(x.max()) if len(x) else None,p99=float(np.percentile(x,99)) if len(x) else None)
def err(a,b):
 aa=np.asarray(a);bb=np.asarray(b);valid=np.isfinite(aa)&np.isfinite(bb)
 return np.abs(aa[valid]-bb[valid]),int(np.sum(np.isfinite(aa)!=np.isfinite(bb))),int(np.sum((np.abs(aa-bb)>1e-12)&valid))
def compare(a,b):
 ds=[];mm=moved=0
 for i in ids:
  d,m,v=err(a[i],b[i]);ds.extend(d);mm+=m;moved+=v
 return dict(**stats(ds),midpoint_presence_mismatches=mm,midpoints_moved_gt_1e_12=moved)
E={};arrays={}
for mode,tag in [('delphi','candidate-gpu'),('converged',args.converged_prefix)]:
 arrays[mode]=np.load(O/(tag+'-energies.npz'));E[mode]=BJ.load_energies([str(O/(tag+'-energies.npz'))])
 for sid in ids:
  ar=arrays[mode];e=E[mode][sid];e.diag.update({k:ar[sid+'/'+k] for k in ('pkint','e_back')});e.diag['pb']={k:ar[sid+'/'+k] for k in ('esolvM','esolvS','esolvM0','sweeps','maxchange','rho')}
# Baseline independently re-executed for all104 energies, including stopping arrays.
z=np.load(O/'baseline-gpu-energies.npz');baseline_equal=all(z[k].dtype==arrays['delphi'][k].dtype and z[k].shape==arrays['delphi'][k].shape and z[k].tobytes()==arrays['delphi'][k].tobytes() for k in z.files)
assert baseline_equal
arms={};timing={};result=dict(env=BJ.env_info(),states=len(ids),sites=sum(recs[i].n_sites for i in ids),baseline_delphi_bitwise=baseline_equal,converged_energy_file=str(O/(args.converged_prefix+'-energies.npz')),comparisons={},titration_seconds=timing)
for mode in ('delphi','converged'):
 for seed in (1,2):
  cfg=A.Config(pb_mode=mode,seed=seed);t=time.perf_counter();tr=A.titrate([E[mode][i] for i in ids],cfg,chain_ids=[A.chain_id_of(i) for i in ids]);tag=f'{mode}_seed{seed}';timing[tag]=time.perf_counter()-t;arms[tag]={i:res['pka'] for i,res in zip(ids,tr)}
  if seed==1:
   D=O/('labels-'+mode);D.mkdir(exist_ok=True);rows=[]
   for sid,res in zip(ids,tr):rows.append(dict(state_id=sid,alias=sid,input='M7 frozen energy bundles',status='ok',arrays=A._state_arrays(recs[sid].n_sites,E[mode][sid],res),diag={}))
   A.write_shard(str(D/'labels_00000.npz'),rows,dict(cfg=A.cfg_dict(cfg),cfg_hash=A.cfg_hash(cfg),code_sha=A.code_sha()),{})
  print(tag,timing[tag],flush=True)
# Teacher alignment via its named site tokens; real StateRecords contain saved teacher JSON.
teacher={};flags={}
for sid in ids:
 r=recs[sid];data=json.loads(r.ref['mc_energies']);toks=data['all_sites'];rr={(x['chain'],x['resnum'],x['group']):x for x in json.loads(r.ref['result'])['rows']}
 teacher[sid]=np.asarray([np.nan if rr[C.token_key(t)]['pka'] is None else rr[C.token_key(t)]['pka'] for t in toks]);flags[sid]=C.load_state(r.cid,r.state)['rows']
 assert len(teacher[sid])==r.n_sites
arms['teacher']=teacher
for name in ('delphi_seed1','converged_seed1'):
 result['comparisons'][name+'_vs_teacher']=compare(arms[name],teacher)
result['comparisons']['converged_vs_delphi_crn']=compare(arms['converged_seed1'],arms['delphi_seed1'])
for mode in ('delphi','converged'):result['comparisons'][mode+'_seed_noise']=compare(arms[mode+'_seed1'],arms[mode+'_seed2'])
shifts={name:[] for name in arms};triples=sorted({recs[i].cid for i in ids if all(recs[i].cid+'_'+role in ids for role in ('AB','A','B'))})
for cid in triples:
 ab=cid+'_AB';lookup={C.sitekey(row):k for k,row in enumerate(flags[ab])}
 for role in ('A','B'):
  fr=cid+'_'+role
  for kf,row in enumerate(flags[fr]):
   ka=lookup.get(C.sitekey(row))
   if ka is None or not(flags[ab][ka]['interface'] and flags[ab][ka]['supervision_eligible'] and row['supervision_eligible']):continue
   for name in arms:shifts[name].append(arms[name][ab][ka]-arms[name][fr][kf])
result['shift_interface_sites']=len(shifts['teacher']);result['shift_comparisons']={}
for name,target in [('converged_seed1','teacher'),('delphi_seed1','teacher'),('converged_seed1','delphi_seed1'),('converged_seed1','converged_seed2'),('delphi_seed1','delphi_seed2')]:
 d,m,v=err(shifts[name],shifts[target]);result['shift_comparisons'][name+'_vs_'+target]=dict(**stats(d),midpoint_presence_mismatches=m,midpoints_moved_gt_1e_12=v)
np.savez_compressed(O/'pka-arms.npz',**{name+'/'+sid:pk for name,data in arms.items() for sid,pk in data.items()})
result['completed']=True;BJ.jdump(result,str(O/'labels.json'));print(json.dumps(result,indent=1),flush=True)
