"""Build final M7 evidence and <=400-word report from completed jobs only."""
import datetime,json,hashlib,subprocess,math
from pathlib import Path
D=Path('/tmp/m7-evidence')
def read(name):return json.loads((D/(name+'.json')).read_text())
# Frozen final snapshots are identified by the job ledger, not by directory recency.
jobs=json.loads(Path('results/m7-jobs.json').read_text())
final_tests=next(r for r in reversed(jobs) if r['tag'].startswith('tests-final'))
tests=read('tests-'+final_tests['job']);assert tests['completed'] and tests['passed']
mode_job=next(r for r in reversed(jobs) if r['tag']=='mode-cli')
mode_cli=json.loads((D/('m7-mode-cli-'+mode_job['job']+'.out')).read_text().strip().splitlines()[-1]);assert mode_cli['pass_']
analytic_job=next(r for r in reversed(jobs) if r['tag']=='analytic-final')
analytic=json.loads((D/('m7-analytic-final-'+analytic_job['job']+'.out')).read_text().strip().splitlines()[-1]);assert analytic['pass_']
names=['baseline-cpu','baseline-gpu','candidate-cpu','candidate-gpu','fresh-cpu','fresh-gpu','converged-final','fresh-converged','converged-cpu']
runs={name:read(name) for name in names}
for name,r in runs.items():assert r['completed'] and r['passed'],name
close={name:read('close-'+name) for name in ('gpu','cpu')}
assert all(r['completed'] for r in close.values())
labels=read('labels');assert labels['completed']
label_details=read('label-details');assert label_details['completed']
assert abs(label_details['seeds']['1']['mae']-labels['comparisons']['converged_vs_delphi_crn']['mae'])<1e-12
labels['saved_midpoint_audit']=label_details
snapshot=read('converged-firstpass')
assert snapshot['completed'] and snapshot['passed'] and snapshot['snapshot']['verified_arrays']==1456
assert snapshot['passes'][0]['rows']==runs['converged-final']['passes'][0]['rows']
assert labels['converged_energy_file']==snapshot['snapshot']['energies']
assert labels['comparisons']['delphi_seed1_vs_teacher']['mae']<=0.05, 'M6 default teacher parity bar'
memory=read('memory');assert memory['passed'] and memory['states']>=2000
assert memory['env']['job']==next(r['job'] for r in reversed(jobs) if r['tag']=='memory-final')

def matches(a,b,ids=None):
 aa=a['passes'][0]['rows'];bb=b['passes'][0]['rows'];ids=ids or list(aa)
 return dict(states=len(ids),bitwise=all(aa[i]['hashes']==bb[i]['hashes'] for i in ids),differing=[i for i in ids if aa[i]['hashes']!=bb[i]['hashes']])
parity={device:matches(runs['candidate-'+device],runs['baseline-'+device]) for device in ('cpu','gpu')}
determinism={device:matches(runs['fresh-'+device],runs['candidate-'+device]) for device in ('cpu','gpu')}
determinism['converged']=matches(runs['fresh-converged'],runs['converged-final'])
flat={name:dict(first=r['passes'][0]['new_compiles'],later=[p['new_compiles'] for p in r['passes'][1:]],passed=all(p['new_compiles']==0 for p in r['passes'][1:])) for name,r in runs.items() if len(r['passes'])>1}
assert all(x['bitwise'] for x in list(parity.values())+list(determinism.values()))
assert all(x['passed'] for x in flat.values())
real_memory={}
for name in ('candidate-gpu','converged-final','candidate-cpu','converged-cpu'):
 real_memory[name]=[dict(pass_index=k,states=len(pp['rows']),gpu_peak_bytes=max(v['gpu'][1] for v in pp['rows'].values()),host_rss_max_kb=max(v['host_rss_kb'] for v in pp['rows'].values()),host_rss_end_kb=list(pp['rows'].values())[-1]['host_rss_kb']) for k,pp in enumerate(runs[name]['passes'])]
res={}
for name in ('converged-final','converged-cpu'):
 rows=runs[name]['passes'][0]['rows'];res[name]=dict(states=len(rows),internal_max=max(v['flags']['residual_max'] for v in rows.values()),stored_max=max(v['flags']['stored_residual_max'] for v in rows.values()))
 assert res[name]['internal_max']<=1e-9 and res[name]['stored_max']<=1e-4
ref_summary={}
for device,c in close.items():
 for sid,row in c['rows'].items():
  for mode in ('converged','delphi'):
   assert all(math.isfinite(v['max']) and math.isfinite(v['mae']) for v in row[mode]['errors'].values()),(device,sid,mode)
 rr={}
 for mode in ('converged','delphi'):
  rr[mode]=dict(max_dphi_same_boundary={k:max(v[mode]['max_dphi_same_boundary'].get(k,0.) for v in c['rows'].values()) for k in ('coarse','focused','model')},max_errors={k:max(v[mode]['errors'][k]['max'] for v in c['rows'].values()) for k in ('esolvM0','esolvM','esolvS','g','W')},reference_residual_max=max(v[mode]['reference_residual_max'] for v in c['rows'].values()))
  assert rr[mode]['reference_residual_max']<=1e-10
 assert max(rr['converged']['max_dphi_same_boundary'].values())<=1e-4
 assert max(rr['converged']['max_errors'].values())<=1e-3
 ref_summary[device]=dict(states=len(c['rows']),**rr)
assert ref_summary['gpu']['states']>=20

def cost(name):
 r=runs[name];rows=r['passes'][-1]['rows'];warm=[x for x in rows.values() if x['compiles']==0]
 return dict(states=len(rows),warm_states=len(warm),mean_warm_pb_seconds=sum(x['timing']['t_pb'] for x in warm)/len(warm),mean_warm_total_seconds=sum(x['seconds'] for x in warm)/len(warm),max_host_rss_kb=max(x['host_rss_kb'] for x in rows.values()),max_gpu_peak_bytes=max(x['gpu'][1] for x in rows.values()),iterations={k:dict(total=sum(x['iterations'][k]['total'] for x in rows.values()),max=max(x['iterations'][k]['max'] for x in rows.values())) for k in ('coarse','focused','model')})
costs={name:cost(name) for name in ('candidate-gpu','converged-final','candidate-cpu','converged-cpu')}
# CPU comparison is restricted to the same five OPT2 states.
ids=list(runs['converged-cpu']['passes'][-1]['rows'])
base=runs['candidate-cpu']['passes'][-1]['rows']
costs['delphi_cpu5']=dict(states=ids,mean_warm_pb_seconds=sum(base[i]['timing']['t_pb'] for i in ids)/len(ids),iterations={k:dict(total=sum(base[i]['iterations'][k]['total'] for i in ids),max=max(base[i]['iterations'][k]['max'] for i in ids)) for k in ('coarse','focused','model')})
costs['matched_comparisons']={}
for device,base_name,conv_name in [('gpu','candidate-gpu','converged-final'),('cpu','candidate-cpu','converged-cpu')]:
 br=runs[base_name]['passes'][-1]['rows'];cr=runs[conv_name]['passes'][-1]['rows'];shared=list(cr)
 mean_b=sum(br[i]['timing']['t_pb'] for i in shared)/len(shared)
 mean_c=sum(cr[i]['timing']['t_pb'] for i in shared)/len(shared)
 by_type={}
 for kind in ('coarse','focused','model'):
  b=sum(br[i]['iterations'][kind]['mean'] for i in shared)/len(shared)
  c=sum(cr[i]['iterations'][kind]['mean'] for i in shared)/len(shared)
  tb=sum(br[i]['iterations'][kind]['total'] for i in shared);tc=sum(cr[i]['iterations'][kind]['total'] for i in shared)
  by_type[kind]=dict(delphi_mean_per_state_lane_sweeps=b,converged_mean_per_state_lane_iterations=c,extra_mean_per_state_lane_iterations=c-b,total_delphi_sweeps=tb,total_pcg_iterations=tc,total_extra_iterations=tc-tb)
 costs['matched_comparisons'][device]=dict(states=len(shared),delphi_pb_seconds=mean_b,converged_pb_seconds=mean_c,slowdown=mean_c/mean_b,by_solve_type=by_type,mean_definition='mean over states of each state mean over tautomers; totals also reported; last warm pass')
# Preserve full per-state measurements in the evidence archive; result has key summaries.
allocation_audit=json.loads(Path('results/m7-compute-audit.json').read_text());assert allocation_audit['passed']
selected_tags=('baseline-cpu','baseline-gpu','candidate-cpu-parallel','candidate-gpu','fresh-cpu-parallel','fresh-gpu','converged-v3','fresh-converged4','converged-cpu3','close-gpu3','close-cpu3','labels4','label-details','snapshot-firstpass','memory-final','tests-final3','mode-cli','analytic-final','probe-gpu3','probe-cpu3')
for tag in selected_tags:
 row=next(r for r in reversed(jobs) if r['tag']==tag)
 assert allocation_audit['allocations'][row['job']]['State']=='COMPLETED',tag
audit=json.loads(Path('results/m7-source-audit.json').read_text());assert all(v['frozen_unchanged'] for v in audit.values())
local_sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('src/jaxpb').glob('*.py')}
source_checks={}
for tag in ('converged-v3','converged-cpu3','close-gpu3','close-cpu3','fresh-converged4','labels4','label-details','snapshot-firstpass','memory-final','tests-final3','analytic-final'):
 row=next(r for r in reversed(jobs) if r['tag']==tag)
 source_checks[tag]=all(row['hashes'].get(k)==v for k,v in local_sources.items())
 assert source_checks[tag],tag
default_source_review={}
for tag in ('candidate-gpu','candidate-cpu-parallel','fresh-gpu','fresh-cpu-parallel'):
 row=next(r for r in reversed(jobs) if r['tag']==tag)
 differences=[k for k,v in local_sources.items() if row['hashes'].get(k)!=v]
 assert set(differences)<=set(('src/jaxpb/api.py','src/jaxpb/pb.py','src/jaxpb/pcg.py'))
 if tag!='candidate-gpu':assert 'src/jaxpb/pb.py' not in differences
 default_source_review[tag]=dict(different_files=differences,review='Only subsequent CLI mode selection and converged-only backend naming differ in API/PB; the PCG module is unused by Delphi mode. Default numerical code is identical. Fresh A40/CPU and CPU repeat snapshots contain the exact final pb.py.')
result=dict(task='BUILD-M7.md',completed=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline='31b30bc',no_commit=True,design=dict(solver='Jacobi PCG, f64 iterates and accumulation, four-lane tile',tol=1e-9,stored_tolerance=1e-4,residual='max interior abs((rhs-Aphi)/diag(A)), kT/e; separately checked after float32 storage',reference='independent unpreconditioned JAX SciPy CG; separately written sliced interior stencil, L2 rtol=1e-13 atol=1e-12, true residual <=1e-10',reference_adapter_note='The reference arm borrows converged PBConfig for setup/post and inherits its diagnostic field names; the actual solve is the independent REF.pool implementation, with the separately stated reference tolerances.',storage='Reference and production solve the physical float32 setup diagonal in f64. Both use original f32 focusing/postprocessing. Delphi keeps its rounded reciprocal and stopped iterate unchanged.',selection={k:read('probe-'+k)['results'] for k in ('cpu','gpu')}),gates=dict(delphi_bitwise_vs_31b30bc=parity,fresh_process_determinism=determinism,flat_compiles=flat,compile_counters={name:rr["compiles"] for name,rr in runs.items()},residuals=res,reference=ref_summary,memory=memory,real_state_memory_by_pass=real_memory,tests=tests,mode_cli_and_foreign_shard_rejection=mode_cli,independent_analytic_lattice=analytic),scientific_result=labels,cost=costs,raw_close=close,compute=dict(default_mode_source_review=default_source_review,allocation_audit=allocation_audit,first_pass_snapshot=snapshot['snapshot'],host='Coulson-Chemistry',partitions=['amd96','oc'],gpu='comp1400 gpu:a40:1',cpu_limit_per_job=12,memory_per_cpu='2G',environment='envs/ogqt-main-7c3e3e9',jobs=jobs,poll='jobs/m7_poll.py: squeue every 180s with sleep',accounting='results/m7-accounting.txt',source_audit='results/m7-source-audit.json',source_snapshots_unchanged=True,final_new_mode_source_matches_local=source_checks,outputs_root='Coulson-Chemistry:'+str(Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/m7'))),summary_contract=dict(keys_retained=True,history_limit=256,full_histories='run.log.jsonl; state timings also remain in shard timing_json',aggregates='state_timing, history_totals; input plan and one active shard still resident',already_aggregated_in_baseline='per-state timings were trimmed in OPT1; M7 caps remaining batch/shard histories'),evidence_archive='results/m7-evidence.tar.gz',limitations=['Historical data-driven M1/M2/M5/M6/determinism scripts were not rerun verbatim; M7 baseline/repeat/fresh/teacher gates re-execute their relevant production paths on the required sets.', 'Tight discrete convergence is not continuum or experimental accuracy.','Full reference chaining retains the existing f32 focusing/postprocessing.','MC errors include sampling effects; CRN and seed-noise references are reported.','Fresh-process determinism covers bench20 for both modes on A40 and default mode on CPU; the full104 A40 sets are repeated in one process.', 'No GH200 or 30k-state production run in M7.'])
mm=real_memory['converged-final']
result['limitations'].append(f"Physical converged validation harness host RSS increased {mm[-1]['host_rss_end_kb']-mm[0]['host_rss_end_kb']} KiB between pass endpoints; GPU peak increased {mm[-1]['gpu_peak_bytes']-mm[0]['gpu_peak_bytes']} bytes. The 4096-state driver retention gate uses synthetic PB/MC; longer physical host-RSS scaling was not verified.")
Path('results/m7.json').write_text(json.dumps(result,indent=2)+'\n')
probe_gpu=read('probe-gpu')['results'];probe_cpu=read('probe-cpu')['results']
probe_pcg_gpu=probe_gpu['pcg']['seconds'][-1];probe_sor_gpu=probe_gpu['sor']['seconds'][-1]
probe_pcg_cpu=probe_cpu['pcg']['seconds'][-1];probe_sor_cpu=probe_cpu['sor']['seconds'][-1]
shift_metrics=labels['shift_comparisons']
pk=labels['comparisons']['converged_vs_delphi_crn'];shift=labels['shift_comparisons']['converged_seed1_vs_delphi_seed1'];teacher=labels['comparisons'];g=costs['converged-final']['mean_warm_pb_seconds'];d=costs['candidate-gpu']['mean_warm_pb_seconds'];m=memory
report=f'''# M7 report

Selected f64 Jacobi PCG: on the real-site A40 coarse/model probe, warm PCG took {probe_pcg_gpu:.2f}s versus tight residual-stopped SOR {probe_sor_gpu:.2f}s (CPU {probe_pcg_cpu:.2f}/{probe_sor_cpu:.2f}s). Four-lane tiles bound memory. The equation residual is max interior |(rhs−Aφ)/diag(A)| in kT/e: internal tolerance 1e-9; separately checked float32 stored-buffer tolerance 1e-4. A separate sliced-interior, unpreconditioned f64 CG reference uses L2 rtol 1e-13/atol 1e-12 and checks true residual ≤1e-10. Existing focusing and postprocessing remain float32.

| Gate | Result |
|---|---|
| Default vs pristine 31b30bc | CPU20 and A40 L3-104 energies bitwise |
| Repeat + bench20 fresh process | Both modes pass; warm compile counts flat |
| Reference, all three solve types | A40 bench20 (including OPT2 five) and CPU five; max Δφ {max(ref_summary["gpu"]["converged"]["max_dphi_same_boundary"].values()):.2e} kT/e |
| Analytic lattice, one-site, mode isolation, 10 self-contained test files | Pass |
| Residuals, all tested converged solves | Internal max {res["converged-final"]["internal_max"]:.2e}; stored {res["converged-final"]["stored_max"]:.2e} kT/e |
| Driver memory, 4,096 synthetic states | RSS span {m['rss_span_kb']} KiB over states 1,024–4,096 |

On L3, tightening DelPhi's solve changes pKa by **MAE {pk['mae']:.5f}, max {pk['max']:.5f} pK** with common random numbers; {pk['midpoints_moved_gt_1e_12']} finite midpoints move and {pk['midpoint_presence_mismatches']} lose a finite midpoint. {label_details['seeds']['1']['changes_above_pK']['0.1']} shared midpoints move by >0.1 pK. Interface shifts move by MAE {shift['mae']:.5f}, max {shift['max']:.5f} pK. Teacher pKa MAE is {teacher['delphi_seed1_vs_teacher']['mae']:.5f} (DelPhi; {teacher['delphi_seed1_vs_teacher']['n']} finite) and {teacher['converged_seed1_vs_teacher']['mae']:.5f} (converged; {teacher['converged_seed1_vs_teacher']['n']} finite). Two-seed pKa noise MAE is {teacher['delphi_seed_noise']['mae']:.5f}/{teacher['converged_seed_noise']['mae']:.5f} pK (DelPhi/converged); paired-shift noise is {shift_metrics['delphi_seed1_vs_delphi_seed2']['mae']:.5f}/{shift_metrics['converged_seed1_vs_converged_seed2']['mae']:.5f}. Shift MAE versus teacher is {shift_metrics['delphi_seed1_vs_teacher']['mae']:.5f}/{shift_metrics['converged_seed1_vs_teacher']['mae']:.5f} pK. M6's pKa/shift MC floors were 0.00406/0.0061 pK. These differences measure stopped-iterate error in this discrete pipeline, including remaining MC sensitivity.

Warm A40 PB costs {g:.2f}s/state versus {d:.2f}s/state for DelPhi mode ({g/d:.1f}×); CPU-five PB costs {costs['converged-cpu']['mean_warm_pb_seconds']:.2f}/{costs['delphi_cpu5']['mean_warm_pb_seconds']:.2f}s/state (converged/DelPhi). Per-type iterations and memory peaks are in m7.json. Summary histories retain 256 rows plus aggregates; complete rows stream to run.log.jsonl. Per-shard arrays/resume behavior pass existing tests. OPT1 had already aggregated state timings; M7 completes the history bound.

Risks: float32 storage/postprocessing, finite grids/boundaries and MC sampling remain. Long physical host-RSS scaling and GH200/30k production remain unverified. Raw per-state reference errors, all job outcomes and frozen-source hashes are retained. No commits.
'''
assert len(report.split())<=400,len(report.split());Path('M7-REPORT.md').write_text(report)
writeup=Path('JAXPB-WRITEUP.md').read_text();section=f'''## M7: measured effect of the stopping rule

On 104 L3 states, converged versus DelPhi-mode pKa changes have MAE {pk['mae']:.5f} and maximum {pk['max']:.5f} pK with common random numbers across 7,073 shared finite midpoints; 156 lose finite midpoints. Interface-shift changes have MAE {shift['mae']:.5f} and maximum {shift['max']:.5f}. Teacher pKa MAE changes from {teacher['delphi_seed1_vs_teacher']['mae']:.5f} to {teacher['converged_seed1_vs_teacher']['mae']:.5f}. M6's saved-energy 200k-sweep MC floor is 0.00406 pK; M7 also measures two-seed noise for both modes. Warm PB costs {g:.2f} versus {d:.2f} A40 seconds/state. Default CPU20/A40 L3 energies remain bitwise equal to pristine 31b30bc. Independent reference comparisons cover bench20, including OPT2's five states. The 4,096-state driver replay with synthetic PB/MC has a {m['rss_span_kb']} KiB RSS span across its final 3,072 states. See `M7-REPORT.md` and `results/m7.json` for exact gates, per-solve-type errors and cost definitions.'''
writeup=(writeup.replace('<!-- M7_RESULTS -->',section) if '<!-- M7_RESULTS -->' in writeup else writeup.split('## M7: measured effect of the stopping rule')[0]+section+'\n\n## Validation history and performance'+writeup.split('## Validation history and performance',1)[1]);assert len(writeup.split())<=1200,len(writeup.split());Path('JAXPB-WRITEUP.md').write_text(writeup)
print('M7 report:',len(report.split()),'words; write-up:',len(writeup.split()),'words')
