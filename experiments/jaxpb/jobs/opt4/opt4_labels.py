"""Full label-shard gate from baseline-verified saved energy arrays; unchanged MC."""
import argparse,glob,json,os
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
import jax,numpy as np
import bench_jaxpb as BJ
from opt2 import CPU_IDS,digest
from jaxpb import api as A,records as R
p=argparse.ArgumentParser(); p.add_argument('--tag',required=True); p.add_argument('--candidate',required=True); p.add_argument('--baseline',required=True); a=p.parse_args()
O=Path(BJ.TP)/'opt4'; baseline=json.load(open(a.baseline)); fresh=json.load(open(a.candidate+'.json')); ids=CPU_IDS
assert fresh.get('passes') and len(fresh['passes'][0]['rows'])==20, 'complete first-pass PB capture required'
assert all(row['hashes']==baseline['passes'][0]['rows'][sid]['hashes'] for sid,row in fresh['passes'][0]['rows'].items()), 'first pass must match pristine baseline in every captured field'
assert baseline.get('completed') and baseline.get('passed'), 'baseline must be complete'
baseline_paths=[O/'baseline5-arrays-energies.npz']
array_capture=json.load(open(O/'baseline5-arrays.json'));assert array_capture.get('completed') and array_capture.get('passed'), 'independent baseline capture must pass reference check'
assert all(path.exists() for path in baseline_paths), 'independent baseline array files required'
old=BJ.load_energies([str(path) for path in baseline_paths]); new=BJ.load_energies([a.candidate+'-energies.npz'])
result=dict(tag=a.tag,job=os.environ['SLURM_JOB_ID'],states=ids,baseline_energy_files=[str(path) for path in baseline_paths],baseline_arrays_independent=True,candidate_first_pass_verified=True,candidate_input_snapshot=a.candidate,baseline_array_capture='baseline5-arrays.json',env=BJ.env_info(),baseline_verified={},shards={})
for sid in ids:
 h=baseline['passes'][0]['rows'][sid]['hashes']['energy']
 assert all(digest(getattr(new[sid],k))==v for k,v in h.items())
 assert fresh['passes'][0]['rows'][sid]['hashes']==baseline['passes'][0]['rows'][sid]['hashes']
 result['baseline_verified'][sid]=sid in old and all(digest(getattr(old[sid],k))==v for k,v in h.items()); assert result['baseline_verified'][sid]
for label,E,paths in [('baseline',old,baseline_paths),('candidate',new,[a.candidate+'-energies.npz'])]:
 for path in paths:
  with np.load(path) as arrays:
   for sid in ids:
    if sid+'/g' not in arrays:continue
    E[sid].diag.update(pkint=arrays[sid+'/pkint'],e_back=arrays[sid+'/e_back'],pb={k:arrays[sid+'/'+k] for k in ('esolvM','esolvS','esolvM0','sweeps','maxchange','rho')})
for label,E in [('baseline',old),('candidate',new),('repeat',new)]:
 states=[E[s] for s in ids]; cfg=A.Config(mc_sweeps=200000); tr=A.titrate(states,cfg,chain_ids=[A.chain_id_of(s) for s in ids]); rows=[]
 for i,sid in enumerate(ids):
  # Both arms use independently saved PB fields from each full source run.
  # Every baseline/candidate MC input is fingerprinted above.
  fields=A._state_arrays(len(E[sid].nst),E[sid],tr[i]); rows.append(dict(state_id=sid,alias=sid,input='OPT4',status='ok',arrays=fields,diag={}))
 path=O/(a.tag+'-'+label+'-labels.npz'); A.write_shard(str(path),rows,dict(task='OPT4',mc_sweeps=200000),{})
 with np.load(path) as z: result['shards'][label]={k:digest(z[k]) for k in z.files if k not in ('provenance_json','timing_json')}
 BJ.jdump(result,str(O/(a.tag+'.json')))
result['passed']=len({json.dumps(v,sort_keys=True) for v in result['shards'].values()})==1; result['completed']=True; BJ.jdump(result,str(O/(a.tag+'.json')))
