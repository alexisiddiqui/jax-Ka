"""Assemble final OPT4 evidence after the required completed cluster gates.

This script only reads collected JSON/accounting and repository metadata. It
never runs a scientific calculation or silently treats an incomplete gate as a pass.
"""
import argparse,datetime,hashlib,json,subprocess,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--decisions',default='results/opt4-decisions.json');a=p.parse_args()
with tarfile.open('results/opt4-evidence.tar.gz') as tf:
 evidence={m.name:json.load(tf.extractfile(m)) for m in tf.getmembers() if m.isfile() and '/' not in m.name and m.name.endswith('.json')}
D=json.loads(Path(a.decisions).read_text())
def complete(name,passed=False):
 d=evidence[name+'.json'];assert d.get('completed'),(name,'incomplete')
 if passed:assert d.get('passed'),(name,'failed')
 return d

def summarize_profile(d):
 rows={}
 for sid,r in d['rows'].items():
  stages=r['production']['stages'];stage=lambda k:max(0,stages[k]['t']-stages[k]['comp_s'])
  s=dict(maps=sum(r['production_maps_s']),geometry=stage('geometry'),geometry_sites=stage('site_probes'),sor=sum(x['seconds'] for x in r['sor']),setup=stage('setup1')+stage('setup2'),spectral_radius=stage('specrad'),post=stage('post'),assembly=stage('assembly'))
  rows[sid]=dict(production_seconds=s,production_stage_sum_seconds=sum(s.values()),isolated_maps_seconds=r['maps_ops'],geometry_ops_seconds=r['geometry_ops'],sor_pools=r['sor'],host_hwm_kb=r['host_hwm_kb'],maps_decomp_exact=r['maps_decomp_exact'],maps_decomp_differences=r.get('maps_decomp_differences'),geometry_parity=r['geometry_parity'])
 assert len(rows)==5
 means={k:sum(r['production_seconds'][k] for r in rows.values())/len(rows) for k in next(iter(rows.values()))['production_seconds']}
 maps_keys=set().union(*(r['isolated_maps_seconds'] for r in rows.values()))
 map_means={k:sum(r['isolated_maps_seconds'].get(k,0) for r in rows.values())/len(rows) for k in maps_keys}
 geo_means={name:sum(sum(r['geometry_ops_seconds'].get(k,0) for k in keys) for r in rows.values())/len(rows) for name,keys in dict(candidate_search=('vx_cand','arc_cand'),burial=('vx_burial','arc_burial'),generation=('vx_gen','arc_gen'),finish=('vx_finish','arc_finish')).items()}
 pools=[x for r in rows.values() for x in r['sor_pools']]
 return dict(env=d['env'],states=rows,mean_production_seconds=means,mean_pb_stage_sum_seconds=sum(means.values()),mean_isolated_maps_seconds=map_means,mean_from_scratch_geometry_op_seconds=geo_means,sor_seconds_per_useful_lane_sweep=sum(x['seconds'] for x in pools)/sum(x['lane_sweeps'] for x in pools))

def peak_host_kb(r):
 values=[r['host_hwm_kb']]
 for stage in r.get('detail',{}).get('mem',{}).values():
  values.extend(v for k,v in stage.items() if k.startswith('hwm_kb'))
 return max(values)

def accounting_peak_kb(job):
 values=[0]
 for line in Path('results/opt4-accounting.txt').read_text().splitlines():
  parts=line.split('|')
  if len(parts)>7 and parts[0].split('.')[0]==str(job) and parts[7]:
   raw=parts[7];unit=raw[-1];factor={'K':1,'M':1024,'G':1048576,'T':1073741824}.get(unit)
   values.append(float(raw[:-1])*factor if factor else float(raw)/1024)
 return max(values)

def summarize_gate(name):
 d=complete(name,True);passes=[]
 for pp in d['passes']:
  rows=pp['rows'];target={sid:r for sid,r in rows.items() if sid in profiles['cpu12']['states']};passes.append(dict(states=len(rows),target_cpu5_states=len(target),target_cpu5_mean_pb_stage_sum_seconds=sum(sum(r['stages'].values()) for r in target.values())/len(target),target_cpu5_mean_stages={k:sum(r['stages'].get(k,0) for r in target.values())/len(target) for k in set().union(*(r['stages'] for r in target.values()))},target_cpu5_mean_instrumented_pb_wall_seconds=sum(r['timing']['t_pb'] for r in target.values())/len(target),new_compiles=pp['new_compiles'],checks={sid:dict(r['checks'],**({'reference':True} if name in posthoc_tags else {})) for sid,r in rows.items()},mean_pb_stage_sum_seconds=sum(sum(r['stages'].values()) for r in rows.values())/len(rows),mean_stages={k:sum(r['stages'].get(k,0) for r in rows.values())/len(rows) for k in set().union(*(r['stages'] for r in rows.values()))},peak_host_hwm_kb=max(peak_host_kb(r) for r in rows.values())))
 return dict(passed=True,env=d['env'],accounting_peak_rss_kb=accounting_peak_kb(d['env']['job']),passes=passes,raw_evidence=name+'.json',reference_comparison=('posthoc strict hash equality, original capture retained' if name in posthoc_tags else 'inline'))
required=dict(cpu12=20,cpu12_fresh=20,cpu1=5,a40_pallas=20,a40_xla=20,a40_pallas_fresh=20,a40_xla_fresh=20)
assert set(D['gates'])==set(required), 'all required gates must be named'
assert set(D['profiles'])=={'cpu1','cpu12'}
posthoc_tags=set();posthoc_reference_evidence={}
for key,reference_tag in D.get('posthoc_references',{}).items():
 tag=D['gates'][key];d=complete(tag,True);reference=complete(reference_tag,True)
 checks=[{sid:r['hashes']==reference['passes'][0]['rows'][sid]['hashes'] for sid,r in pp['rows'].items()} for pp in d['passes']]
 assert all(all(row.values()) for row in checks),(tag,'posthoc reference mismatch')
 posthoc_tags.add(tag)
 if key=='cpu12_fresh':
  primary=complete(D['gates']['cpu12'],True);independent=[{sid:r['hashes']==primary['passes'][0]['rows'][sid]['hashes'] for sid,r in pp['rows'].items()} for pp in d['passes']]
  assert all(all(row.values()) for row in independent),'fresh process differs from primary'
 else:independent=None
 canonical=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 posthoc_reference_evidence[key]=dict(candidate=tag+'.json',baseline=reference_tag+'.json',checks=checks,independent_process_checks=independent,passed=True,candidate_canonical_json_sha256=canonical(d),baseline_canonical_json_sha256=canonical(reference),method='Strict equality of all maps/geometry/sites/phi/PB/energy/PB-shard fingerprints in every captured state and pass. Original capture evidence remains unmodified.')
for key,tag in D['gates'].items():
 d=complete(tag,True);assert len(d['passes'][0]['rows'])==required[key],key
 if key.startswith('cpu'):assert d['env']['affinity']==(1 if key=='cpu1' else 12),(key,'CPU affinity')
 else:assert d['env']['host']=='comp1400' and d['env']['backend']=='gpu',(key,'A40 allocation')
 assert tag in posthoc_tags or all(r['checks'].get('reference') is True for pp in d['passes'] for r in pp['rows'].values()),key
 if key in ('cpu12','a40_pallas','a40_xla'):
  assert len(d['passes'])==2 and d['passes'][1]['new_compiles']==0,(key,'warm compile growth')
  assert len(d['passes'][1]['rows'])==20,(key,'incomplete repeat')
  assert all(r['checks'].get('repeat') is True for r in d['passes'][1]['rows'].values()),key
array_format=json.loads(Path('results/opt4-baseline-array-format.json').read_text());assert array_format['passed'] and array_format['members']==70
array_capture=complete('baseline5-arrays',True);assert len(array_capture['passes'][0]['rows'])==5
assert all(row['checks'].get('reference') for row in array_capture['passes'][0]['rows'].values())
label_data=complete(D['label_gate'],True);assert label_data.get('baseline_arrays_independent'), 'independently saved baseline label inputs required'
label_snapshot=json.loads(Path('results/opt4-label-input-snapshot.json').read_text())
assert label_snapshot['source_full_pass_verified'] and label_snapshot['states']==20
assert label_data.get('candidate_first_pass_verified') and label_data['candidate_input_snapshot']+'-energies.npz'==label_snapshot['snapshot']
assert len(label_data['shards'])==3 and all(label_data['baseline_verified'].values())
complete(D['unit_gate'],True)
profiles={n:summarize_profile(complete(tag)) for n,tag in D['profiles'].items()}
assert profiles['cpu1']['env']['affinity']==1 and profiles['cpu12']['env']['affinity']==12
gates={key:summarize_gate(tag) for key,tag in D['gates'].items()}
for tag in D['additional_passed']:complete(tag,True)
matched=complete('matched12',True);assert len(matched['rows'])==5 and sum(len(rows) for rows in matched['rows'].values())==15
assert matched['env']['affinity']==12
D['matched_maps_cpu12']={ 'samples':15,'speedup':sum(row['baseline_s'] for rows in matched['rows'].values() for row in rows)/sum(row['selected_s'] for rows in matched['rows'].values() for row in rows),'bitwise':all(row['bitwise'] for rows in matched['rows'].values() for row in rows)}
for tag in D.get('additional_completed',[]):complete(tag)
complete(D['arm_evidence'])
fresh=complete(D['gates']['cpu12_fresh'],True);assert len(fresh['passes'])==2 and len(fresh['passes'][1]['rows'])==5 and fresh['passes'][1]['new_compiles']==0
assert all(r['checks'].get('repeat') for r in fresh['passes'][1]['rows'].values())
for tag,threads in (('native1c',1),('native12',12)):
 d=complete(tag);assert d['env']['affinity']==threads,(tag,'thread affinity')
 for mode in ('off','fast'):
  v=d['variants'][mode];assert v['compile_returncode']==0,(tag,mode,'compile')
  assert set(v['checks'])=={'1','10','100'} and 'converged' in v
  assert str(threads) in v['timing'] and v['timing'][str(threads)]['seconds_per_lane_sweep']>0
  if threads==12:assert set(v['checks_by_threads']['12'])=={'1','10','100'}
for tag in ('native-goff','native-gfast'):
 d=complete(tag);assert 'phi' in d['pools'][0]['comparisons'] and 'g' in d['energy']
source={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src/jaxpb').glob('*.py')}
manifest=json.loads(Path('results/opt4-freezes.json').read_text())+json.loads(Path('results/opt4-jobs.json').read_text())
selected=next(r for r in manifest if r['frozen']==D['selected_frozen'])
assert all(selected['hashes'].get(k)==v for k,v in source.items()), 'delivered source differs from the tested frozen source'
assert selected['hashes']['tests/test_opt4.py']==hashlib.sha256(Path('tests/test_opt4.py').read_bytes()).hexdigest()
base='bdffc91e57021ced00b4d8bcb9b4ff14216ece7e'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==base
baseline_paths=subprocess.check_output(['git','ls-tree','-r','--name-only',base,'src/jaxpb'],text=True).splitlines()
baseline_source={path:hashlib.sha256(subprocess.check_output(['git','show',base+':'+path])).hexdigest() for path in baseline_paths if path.endswith('.py')}
reference_source_audit={}
for suffix in ('base12','baseline-rest','control1','basegpu','basegx'):
 root='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline/opt4-'+suffix
 row=next(x for x in manifest if x['frozen']==root)
 assert all(row['hashes'].get(path)==sha for path,sha in baseline_source.items()),(root,'reference source differs from bdffc91')
 reference_source_audit[root]=dict(matches_baseline=True,source_sha256=baseline_source)

assert source['src/jaxpb/titrate.py']==hashlib.sha256(subprocess.check_output(['git','show',base+':src/jaxpb/titrate.py'])).hexdigest()
performance=dict(published_cpu12_pb_seconds=464.0,pypka_pb_seconds=8.8,pypka_total_seconds_range=[25,29],method='CPU5 compilation-subtracted stage sums; CPU12 uses the fresh-process warm CPU5 pass on comp0652, CPU1 its measured first pass. Instrumented PB wall is also retained in gate summaries.')
for key,rep in (('cpu1',0),('cpu12',1)):
 benchmark_key=D.get('benchmark_cpu12_gate','cpu12') if key=='cpu12' else key
 seconds=gates[benchmark_key]['passes'][rep]['target_cpu5_mean_pb_stage_sum_seconds']
 performance[key]=dict(benchmark_gate=benchmark_key,seconds_per_state=seconds,measured_sor_stage_floor_seconds=sum(gates[benchmark_key]['passes'][rep]['target_cpu5_mean_stages'].get(k,0) for k in ('sor1','sor2')),speedup_vs_464_seconds=464/seconds,ratio_to_pypka_pb=seconds/8.8,stages=gates[benchmark_key]['passes'][rep]['target_cpu5_mean_stages'])
for value in performance.values():
 if isinstance(value,dict) and 'seconds_per_state' in value:value['non_sor_stage_seconds']=value['seconds_per_state']-value['measured_sor_stage_floor_seconds']
for key,tag,threads in (('cpu1','native1c','1'),('cpu12','native12','12')):
 value=performance[key];speedup=evidence[tag+'.json']['variants']['off']['timing'][threads]['speedup']
 value['conditional_native_off_estimate_seconds_per_state']=value['non_sor_stage_seconds']+value['measured_sor_stage_floor_seconds']/speedup
 value['conditional_native_estimate_method']='Optimistic substitution of fixed-100-sweep real-lane native-off speedup into measured SOR stage; assumes exact rounding can be matched, unchanged stops and non-SOR work, excludes callback overhead. Not measured full PB and not eligible for production.'

topology=json.loads(Path('results/opt4-cpu-topology.json').read_text())
fresh_topology=topology[str(gates['cpu12_fresh']['env']['job'])]
assert len(fresh_topology['cpus'])==12 and len({(row['socket'],row['core']) for row in fresh_topology['topology']})==6
assert len({row['L3'] for row in fresh_topology['topology']})==1, 'CPU12 benchmark requires baseline-matched single L3 group'
job_manifest=json.loads(Path('results/opt4-jobs.json').read_text())
assert all(j['cpus']<=12 and '--mem-per-cpu=2G' in j['command'] for j in job_manifest)
for line in Path('results/opt4-accounting.txt').read_text().splitlines():
 parts=line.split('|')
 if len(parts)>3 and parts[3]:assert int(parts[3])<=12,parts
final_poll=json.loads(Path('results/opt4-squeue.jsonl').read_text().splitlines()[-1])
assert final_poll['returncode']==0 and not final_poll['output'].strip(), 'wait for all own cluster jobs before finalizing'
result=dict(task='BUILD-OPT4.md',completed=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_commit=base,no_commit=True,reference_source_audit=reference_source_audit,posthoc_reference_evidence=posthoc_reference_evidence,performance=performance,step0=profiles,step0_methodology=dict(isolated_maps='Warm individually jitted operation copies; compilation subtracted. Not additive estimates of the fused executable. Only untouched production maps are returned to physics.',production='Maps measured inside the wrapper before diagnostic copies; SOR timed around original run_pool; other original stages from prof2 instrumentation with compilation subtracted. PB stage sum excludes isolated diagnostic work.',geometry='From-scratch f64 operation decomposition, including chain states whose production geometry uses AB reuse. Reported separately from production geometry.',cells='Cell-build measurement is a subset of stage0; do not add both.'),decisions=D,independent_arms=evidence[D['arm_evidence']+'.json'],independent_arm_parts={key:evidence.get(key+'.json') for key in ('ab5-maps','ab5-sites','tune5','site-tune5')},native={n:evidence[n+'.json'] for n in ('native1c','native12','native-goff','native-gfast')},gates=gates,baseline_array_format=array_format,label_input_snapshot=label_snapshot,label_gate=evidence[D['label_gate']+'.json'],unit_gate=evidence[D['unit_gate']+'.json'],additional_evidence={n:evidence[n+'.json'] for n in D['additional_passed']+D.get('additional_completed',[])},source_sha256=source,source_diff=subprocess.check_output(['git','diff',base,'--','src','tests'],text=True),new_source_files={str(p):p.read_text() for p in [Path('src/jaxpb/_cpu_cells.py'),Path('tests/test_opt4.py')] if p.exists()},compute=dict(final_squeue_poll=final_poll,resource_samples=[json.loads(line) for line in Path('results/opt4-resource-samples.jsonl').read_text().splitlines()] if Path('results/opt4-resource-samples.jsonl').exists() else [],reference_cleanup=json.loads(Path('results/opt4-reference-cleanup.json').read_text()),cpu_topology=json.loads(Path('results/opt4-cpu-topology.json').read_text()) if Path('results/opt4-cpu-topology.json').exists() else None,job_manifest=json.loads(Path('results/opt4-jobs.json').read_text()),frozen_snapshots=json.loads(Path('results/opt4-freezes.json').read_text()),allocation_control=evidence.get('aux-control.json'),pipeline=evidence.get('aux-cpu-queue.json.status.json'),accounting=Path('results/opt4-accounting.txt').read_text(),scheduling_updates=json.loads(Path('results/opt4-scheduling-updates.json').read_text()) if Path('results/opt4-scheduling-updates.json').exists() else [],arm_checkpoint=json.loads(Path('results/opt4-ab-restart-checkpoint.json').read_text()) if Path('results/opt4-ab-restart-checkpoint.json').exists() else None,polling='results/opt4-squeue.jsonl: squeue every 180 seconds with sleep',raw_evidence_archive='results/opt4-evidence.tar.gz'))
Path('results/opt4.json').write_text(json.dumps(result,indent=2)+'\n');print('Wrote completed results/opt4.json')
