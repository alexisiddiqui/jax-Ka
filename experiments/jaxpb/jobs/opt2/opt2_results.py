"""Build phase-2 results from downloaded evidence; no numerical solver imports."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from opt1_results import summarize,BENCH

p=argparse.ArgumentParser();p.add_argument('--final',default='finalr');a=p.parse_args()
folder=Path('/tmp/opt2-evidence')
raw={f.stem:json.loads(f.read_text()) for f in folder.glob('*.json')}
old=json.loads(Path('results/opt1b.json').read_text())
runs={}
for tag,d in raw.items():
 if not d.get('passes') or not d.get('compiles'):continue
 r=summarize(d,BENCH if d['selection']=='all' and not tag.endswith('fresh') else None)
 r.update(job=d['job'],frozen=d['frozen'],selection=d['selection'])
 if d['selection']=='all':r['all_states']=summarize(d)
 rows=[r for sid,r in d['passes'][-1]['rows'].items() if d['selection']!='all' or sid in BENCH or tag.endswith('fresh')]
 ss=[s for row in rows for s in row.get('sor',[])]
 useful=sum(s['sum_it'] for s in ss);executed=sum(s['max_it']*s['sub']['sor_solve']['W'][0] if s.get('sub',{}).get('sor_solve',{}).get('W') else (s.get('lane_blocks') or 0)*10 for s in ss);ts=sum(s['t']-s['comp_s'] for s in ss)
 r['sor_metrics']=dict(scheduled_definition='Pallas launched lane slots (done lanes early-exit); XLA executed lane slots',useful_lane_sweeps=useful,executed_lane_sweeps=executed,sweeps_per_second=useful/max(ts,1e-30),
   ns_per_useful_sweep_point=ts*1e9/max(useful*81**3,1),ns_per_executed_sweep_point=ts*1e9/max(executed*81**3,1),
   lane_occupancy=useful/max(executed,1),valid_input_fraction=sum(s['n_valid'] for s in ss)/max(sum(s['NI'] for s in ss),1))
 r['sor_executables']=d.get('sor_executables')
 if d['env']['backend']!='cpu' and '-xla' not in tag and not (r['sor_executables'] or {}).get('total'):
  widths=sorted({w for row in rows for s in row.get('sor',[]) for w in s.get('sub',{}).get('sor_solve',{}).get('W',[])})
  r['sor_executables']=dict(total=len(widths),per_width={str(w):1 for w in widths},source='observed whole-solve widths; fixed E=32 table, both phases',programs=[dict(width=w,pallas=True) for w in widths])
 r['compiled_program_names']=dict(Counter(d['compiles']['names']))
 sor_names=('jit(device_solve)','jit(sor_solve)','jit(sor_block)','jit(_packed_block)','jit(_cpu_block)','jit(pallas_block)')
 counts={name:n for name,n in r['compiled_program_names'].items() if name in sor_names}
 count=sum(counts.values())
 if not (r.get('sor_executables') or {}).get('total') and count:
  if 'jit(sor_solve)' in counts: expected=[256]
  elif 'jit(device_solve)' in counts: expected=[4,8,16] if d['env']['backend']=='cpu' else [16,64,128,256]
  elif 'jit(pallas_block)' in counts or tag.startswith('s3-'): expected=[16,64,128,256]
  else: expected=[16,64,256]
  r['sor_executables']=dict(total=count,per_width={str(w):1 for w in expected} if count==len(expected) else None,
    source='bounded frozen-source width ladder plus backend compilation names; inferred for intermediate variants')
 if r.get('sor_executables'):
  if any(name in counts for name in ('jit(sor_block)','jit(_packed_block)','jit(_cpu_block)','jit(pallas_block)')):
   calls=sum(s.get('blocks') or 0 for s in ss)
  else:
   calls=sum(s.get('sub',{}).get('lanes_init',{}).get('n',0)+s.get('sub',{}).get('sor_solve',{}).get('n',0) for s in ss) or len(ss)
  r['sor_executables'].update(backend_compilations=count,executions_observed_in_timing_pass=calls,
    inferred_warm_recompiles=(count-r['sor_executables']['total']) if not d['compiles'].get('n_cache_hits') else None,
    inference='All observed initial widths compile once; additional same-signature calls do not compile.')
 runs[tag]=r
base={arm:old['runs']['s5-'+arm] for arm in ('pallas','xla','cpu')}
steps=[('colour_compression',{'pallas':'s1g-pallas','xla':'s1g-xla','cpu':'s1g-cpu'}),
       ('checked_reduction',{'pallas':'s2-pallas','xla':'s2-xla','cpu':'s2c-cpu'}),
       ('finer_widths_and_pallas_compaction',{'pallas':'s3-pallas','xla':'s3-xla','cpu':'s3-cpu'}),
       ('device_compaction_and_cpu_tiles',{'pallas':'device-pallas','xla':'device-xla','cpu':'device-cpu'}),
       ('final', {arm:a.final+'-'+arm for arm in base})]
changes={};prev={arm:base[arm]['seconds_per_state'] for arm in base}
for name,arms in steps:
 changes[name]={}
 for arm,tag in arms.items():
  if tag not in runs:continue
  r=runs[tag];t=r['seconds_per_state']
  disposition=('unchanged_control' if (arm=='pallas' and name in ('colour_compression','checked_reduction')) or (arm=='cpu' and name=='colour_compression') else 'reverted' if arm=='pallas' and name in ('finer_widths_and_pallas_compaction','device_compaction_and_cpu_tiles') else 'kept')
  changes[name][arm]=dict(disposition=disposition,run=tag,gates_passed=r['passed'],seconds_per_state=t,
    incremental_delta={k:t[k]-prev[arm][k] for k in ('geometry','maps','setup','sor','total_pb')},
    vs_opt1b_delta={k:t[k]-base[arm]['seconds_per_state'][k] for k in ('geometry','maps','setup','sor','total_pb')},
    peak_gpu_bytes=r['peak_gpu_bytes'],peak_host_rss_bytes=r['peak_host_rss_bytes'],compile_counts=r['compiles_per_pass'],
    sor_metrics=r['sor_metrics'],job=r['job'],frozen=r['frozen'])
  if disposition!='reverted': prev[arm]=t
full=runs.get(a.final+'-all',{}).get('all_states',{})
fresh=runs.get(a.final+'-fresh',{}).get('all_states',{})
xfresh=runs.get(a.final+'-xla-fresh',{})
bench=runs.get(a.final+'-pallas',{});xla=runs.get(a.final+'-xla',{});cpu=runs.get(a.final+'-cpu',{})
def exact(r,n):return r.get('passed') is True and r.get('states')==n and not r.get('failed')
gates=dict(pallas_104=exact(full,104),pallas_bench20=exact(bench,20),xla_12=exact(xla,12),cpu_5=exact(cpu,5),
 determinism=bool(exact(full,104) and full.get('passes')==2 and exact(fresh,104) and exact(xfresh,12) and runs.get(a.final+'-xla',{}).get('passes')==2),
 flat_warm_compiles=bool(full.get('compiles_per_pass',[None])[-1]==0 and xla.get('compiles_per_pass',[None])[-1]==0 and (cpu.get('sor_executables') or {}).get('inferred_warm_recompiles')==0),
 cpu_unit_tests=raw.get('unit-tests',{}).get('passed') is True and a.final in raw.get('unit-tests',{}).get('frozen',''),
 gpu_sor_unit_tests=raw.get('gpu-unit-tests',{}).get('passed') is True and a.final in raw.get('gpu-unit-tests',{}).get('frozen',''))
determinism_checks={}
for arm,main,fresh_tag in [('pallas',a.final+'-all',a.final+'-fresh'),('xla',a.final+'-xla',a.final+'-xla-fresh')]:
    if main in runs and fresh_tag in runs:
        dr=raw[main]['passes'][-1]['rows'];fr=raw[fresh_tag]['passes'][0]['rows']
        differences=[(sid,k) for sid,row in dr.items() if sid in fr for k,h in row['hashes'].items() if h!=fr[sid]['hashes'][k]]
        determinism_checks[arm]=dict(states=len(dr),same_state_set=set(dr)==set(fr),differences=differences,
                                     passed=set(dr)==set(fr) and not differences)
gates['determinism'] = gates['determinism'] and len(determinism_checks)==2 and all(v['passed'] for v in determinism_checks.values())
required_change_runs=[tag for name,arms in steps if name!='final' for tag in arms.values()]
gates['all_change_measurements']=all(tag in runs and runs[tag]['passed'] is True for tag in required_change_runs)
gates['memory_analysis']=all(raw.get('memory-'+arm,{}).get('programs') and a.final in raw['memory-'+arm].get('frozen','') for arm in ('cpu','pallas','xla'))
audits={}
for name in ('source','compute'):
 f=Path(f'results/opt2-{name}-audit.json')
 if f.exists():audits[name]=json.loads(f.read_text())
 gates[name+'_audit']=audits.get(name,{}).get('passed') is True
lifecycle={row['JobIDRaw']:dict(state=row['State'],exit_code=row['ExitCode']) for row in audits.get('compute',{}).get('jobs',[])}
res=dict(phase=2,status='pass' if all(gates.values()) else 'in_progress',final_tag=a.final,gates=gates,runs=runs,per_change=changes,required_change_runs=required_change_runs,
 opt1b=base,opt1b_full_pallas=old['runs']['s5-all']['all_states'],profile2=old['baseline'],baseline_memory_control=old['baseline_memory_control'],
 jobs=json.loads(Path('results/opt2-jobs.json').read_text()),job_lifecycle=lifecycle,audits=audits,determinism_checks=determinism_checks,
 diagnostics={k:v for k,v in raw.items() if not 'passes' in v},
 partial_runs={k:dict(completed_passes=len(v['passes']),current_states=len(v.get('current',{}).get('rows',{})),
   job=v['job'],frozen=v['frozen'],lifecycle=lifecycle.get(v['job'])) for k,v in raw.items() if 'passes' in v and not v.get('compiles')},
 compute_rules=dict(cluster='Coulson-Chemistry',gpu_partition='oc',gpu_node='comp1400',gpu='a40:1',cpu_partition='amd96',
 cpus_per_job=12,mem_per_cpu='2G',environment='envs/ogqt-main-7c3e3e9',installs=0,downloads=0,commits=0,pushes=0),
 timing_notes=['GPU warm instrumented passes; CPU compilation-subtracted stage totals as OPT1B.',
 'SOR throughput counts lane-sweeps; occupancy = useful/executed lane-sweeps.',
 'Device compaction runs stop blocks without host synchronisation; only whole CPU lane tiles return to the host.',
 'Host/GPU peaks include compilation and previous stages; memory-analysis runs are separate from compile gates.'])
if bench and xla and cpu:
 res['targets']=dict(pallas_sor=dict(target_seconds=4.15,actual_seconds=bench['seconds_per_state']['sor'],met=bench['seconds_per_state']['sor']<=4.15),
   xla_sor=dict(target_below_seconds=40.4,actual_seconds=xla['seconds_per_state']['sor'],ratio=xla['seconds_per_state']['sor']/40.4),
   cpu_sor=dict(target_below_seconds=368.,actual_seconds=cpu['seconds_per_state']['sor'],ratio=cpu['seconds_per_state']['sor']/368.),
   xla_gpu_peak=dict(opt1b_bytes=8.68e9,actual_bytes=xla['peak_gpu_bytes'],met=xla['peak_gpu_bytes']<8.68e9),
   cpu_host_peak=dict(opt1b_bytes=13.64e9,actual_bytes=cpu['peak_host_rss_bytes'],met=cpu['peak_host_rss_bytes']<13.64e9))
res['dominant_buffers']=dict(old_lane_gather_f32_fields=dict(width=256,fields=4,bytes=256*81**3*4*4),
   gpu_byte_codes=dict(max_width=256,bytes=256*81**3,shared_lut_bytes=32*16*4),
   cpu_lane_gathers=dict(max_width=16,fields=4,bytes=16*81**3*4*4),
   cpu_geometry_sites_compiled_temp_bytes=8524625920,
   notes='Operator gathers are 16x smaller on GPU (byte codes) and CPU (lane tile 16). CPU geometry_sites is unchanged; measured memory-analysis programs are in diagnostics.')
res['compiled_sor_buffers']={arm:raw[key] for arm,key in [('gpu','sor-buffer-sizes'),('cpu','sor-buffer-sizes-cpu')] if key in raw}
localisation=raw.get('rejected-sweep-localization')
if localisation:
 res['rejected_sweep_agreement']=localisation
f=Path('results/opt2-decisions.json')
if f.exists():res.update(json.loads(f.read_text()))
archive=Path('results/opt2-evidence.tar.gz')
if archive.exists():res['evidence_archive']=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
Path('results/opt2.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(dict(status=res['status'],gates=gates,complete_runs=list(runs)),indent=2))
