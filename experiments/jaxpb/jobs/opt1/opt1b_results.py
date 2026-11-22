"""Build the reviewable phase report from downloaded cluster evidence, without JAX."""
import argparse
import hashlib
import json
from pathlib import Path

from opt1_results import BENCH, summarize

p = argparse.ArgumentParser()
p.add_argument('--final', default='s5')
p.add_argument('--steps', nargs='+', default=['s1', 's2', 's3r', 's3g8', 's4', 's5'])
a = p.parse_args()
folder = Path('/tmp/opt1b-evidence')
phase1 = json.loads(Path('results/opt1.json').read_text())
raw = {f.stem: json.loads(f.read_text()) for f in folder.glob('*.json')}
runs = {k: summarize(v, BENCH if v['selection']=='all' and not k.endswith('-fresh') else None) for k,v in raw.items()
        if v.get('passes') and v.get('compiles')}
for tag, row in runs.items():
    row.update(job=raw[tag]['job'], frozen=raw[tag]['frozen'], selection=raw[tag]['selection'],
               timing_scope='bench20 subset' if raw[tag]['selection']=='all' and not tag.endswith('-fresh') else raw[tag]['selection'])
    if raw[tag]['selection'] == 'all': row['all_states'] = summarize(raw[tag])
base = {arm:phase1['comparisons'][arm]['profile2_seconds_per_state'] for arm in ('pallas','xla','cpu')}
previous = {arm:phase1['runs'][{'pallas':'C9-pallas','xla':'C9-xla','cpu':'C8-cpu'}[arm]]['seconds_per_state']
            for arm in base}
changes = {}
for number, tag in enumerate(a.steps, 1):
    changes[tag] = dict(step={'s1':1,'s2':2,'s3r':3,'s3g8':3,'s4':4,'s5':5}.get(tag,number), arms={})
    for arm in base:
        run_tag = 's3r-cpu' if tag=='s3g8' and arm=='cpu' else f'{tag}-{arm}'
        run = runs.get(run_tag)
        if run:
            timing = run['seconds_per_state']
            changes[tag]['arms'][arm] = dict(run=run_tag, reused_cpu_measurement=(run_tag!=tag+'-'+arm), passed=run['passed'], failed=run['failed'],
                cumulative_seconds_per_state=timing,
                incremental_seconds_per_state={k:timing[k]-previous[arm][k] for k in base[arm]},
                vs_baseline_seconds_per_state={k:timing[k]-base[arm][k] for k in base[arm]},
                peak_gpu_bytes=run['peak_gpu_bytes'], peak_host_rss_bytes=run['peak_host_rss_bytes'],
                compile_counts=run['compiles_per_pass'], job=run['job'], frozen=run['frozen'])
            previous[arm] = timing
full = runs.get(a.final+'-all', {}).get('all_states', {})
fresh = runs.get(a.final+'-fresh', {}).get('all_states', {})
xla = runs.get(a.final+'-xla', {})
cpu = runs.get(a.final+'-cpu', {})
bench = runs.get(a.final+'-pallas', {})

def exact(run, states):
    return bool(run.get('passed') and run.get('states') == states and not run.get('failed'))

gates = dict(pallas_104=exact(full,104), pallas_bench20=exact(bench,20), xla_12=exact(xla,12), cpu_5=exact(cpu,5),
             determinism=bool(exact(full,104) and full.get('passes') == 2 and exact(fresh,104)),
             flat_warm_compiles=bool(full.get('compiles_per_pass', [None])[-1]==0 and xla.get('compiles_per_pass',[None])[-1]==0),
             cpu_unit_tests=raw.get('unit-tests',{}).get('passed') is True,
             maps_gpu=raw.get('maps-gpu',{}).get('passed') is True,
             maps_cpu=raw.get('maps-cpu',{}).get('passed') is True)
gates['compile_counts_matched']=bool(full.get('compiles')==35 and xla.get('compiles')==36 and cpu.get('compiles')==34)
for name in ('source','compute'):
    path=Path(f'results/opt1b-{name}-audit.json')
    gates[name+'_audit']=path.exists() and json.loads(path.read_text()).get('passed') is True
required=[f'{tag}-{arm}' for tag in a.steps for arm in base if not (tag=='s3g8' and arm=='cpu')]
gates['all_change_measurements']=all(tag in runs for tag in required)
res = dict(phase='OPT1B', status='pass' if all(gates.values()) else 'in_progress', final_tag=a.final,
           gates=gates, runs=runs, per_change=changes, jobs=json.loads(Path('results/opt1b-jobs.json').read_text()),
           baseline=base, baseline_memory_control={arm:{k:phase1['runs']['baseline-'+arm][k] for k in ('peak_gpu_bytes','peak_host_rss_bytes')} for arm in base}, phase1={arm:phase1['runs'][tag] for arm,tag in [('pallas','C9-pallas'),('xla','C9-xla'),('cpu','C8-cpu')]},
           diagnostics={k:v for k,v in raw.items() if not v.get('passes') and 'current' not in v},
           audits={name:json.loads(Path(f'results/opt1b-{name}-audit.json').read_text()) for name in ('source','compute') if Path(f'results/opt1b-{name}-audit.json').exists()},
           required_runs=required, pending_runs=[tag for tag in required if tag not in runs],
           compute_rules=dict(cluster='Coulson-Chemistry', gpu_partition='oc',gpu_node='comp1400',gpu='a40:1',
                              cpu_partition='amd96',cpus_per_job=12,mem_per_cpu='2G',installs=0,downloads=0,commits=0),
           timing_notes=['GPU instrumented warm passes; CPU compilation-subtracted stages.',
                         'CPU PB sums measured stages excluding assembly, as in phase 1.',
                         'Whole-process allocator and host peaks include compilation; baseline-map checks reported separately.'])
if bench and xla and cpu:
    res['targets'] = dict(pallas_maps=dict(target=3.40, actual=bench['seconds_per_state']['maps'],met=bench['seconds_per_state']['maps']<=3.40),
        xla_maps=dict(target=3.12,actual=xla['seconds_per_state']['maps'],met=xla['seconds_per_state']['maps']<=3.12),
        cpu_maps=dict(target=370.,actual=cpu['seconds_per_state']['maps'],met=cpu['seconds_per_state']['maps']<370.),
        cpu_host_peak=dict(target_bytes=14.1e9,actual_bytes=cpu['peak_host_rss_bytes'],met=cpu['peak_host_rss_bytes']<=14.1e9),
        pallas_gpu_peak=dict(target_bytes=5.5e9,actual_bytes=max(bench['peak_gpu_bytes'],full.get('peak_gpu_bytes',0),fresh.get('peak_gpu_bytes',0))),
        xla_gpu_peak=dict(target_bytes=5.5e9,actual_bytes=xla['peak_gpu_bytes'],met=xla['peak_gpu_bytes']<=5.5e9))
    res['targets']['pallas_gpu_peak']['met']=res['targets']['pallas_gpu_peak']['actual_bytes']<=5.5e9
notes = Path('results/opt1b-decisions.json')
if notes.exists(): res.update(json.loads(notes.read_text()))
Path('results/opt1b.json').write_text(json.dumps(res, indent=2)+'\n')
print(json.dumps(dict(status=res['status'],gates=gates),indent=2))
if not all(gates.values()): raise SystemExit(0)
lines = ['# OPT1B', '',
         'GPU tiles: `_q_chunk`, `_brute_chunk`, `reach_mask_b`. '
         '`build_cells` uses budgeted dense tables and one bounds search; coarse probes remain CSR. '
         '`maps_batch`/`_maps_group`/`build_maps_b`/`_dyn_rows` preserve group width and clean padding; `pb._site_layout` supplies validity. '
         '`_q_one` uses 128-slot slabs and int32 ranks.', '',
         '| Gate | Result |','|---|---|',
         '| Pallas | Exact: 104 L3 + 20 bench |', '| XLA / CPU | Exact: 12 / 5 |',
         '| Determinism | Two passes + fresh process exact |',
         '| Direct maps | Bytewise: five capacities, GPU/CPU |',
         '| Tests / GPU warm compiles | CPU modes + 11 edge tests pass / zero |', '',
         'A40 Pallas/XLA; CPU: 12 cores. s/state: geometry/maps/setup/SOR/PB; baseline PROFILE2; CPU compiler time excluded.', '',
         '| Arm | Baseline | Phase 1 | OPT1B |', '|---|---|---|---|']
for arm in base:
    def fmt(row): return '/'.join(f'{row[k]:.2f}' for k in ('geometry','maps','setup','sor','total_pb'))
    final = runs[a.final+'-'+arm]
    lines.append(f'| {arm} | {fmt(base[arm])} | {fmt(res["phase1"][arm]["seconds_per_state"])} | {fmt(final["seconds_per_state"])} |')
lines += ['', 'Peak GPU/host GB (matched baseline controls).', '',
          '| Arm | Baseline | Phase 1 | OPT1B |', '|---|---|---|---|']
for arm in base:
    final = full if arm=='pallas' else runs[a.final+'-'+arm]
    def mem(row):
        gpu='—' if arm=='cpu' else f"{row['peak_gpu_bytes']/1e9:.2f}"
        return gpu+f"/{row['peak_host_rss_bytes']/1e9:.2f}"
    lines.append(f"| {arm} | {mem(res['baseline_memory_control'][arm])} | {mem(res['phase1'][arm])} | {mem(final)} |")
lines += ['', 'Compile counts (phase 1: 35/36/34): ' + '/'.join(str((full if arm=='pallas' else runs[a.final+'-'+arm])['compiles']) for arm in base)+'.', '',
          res.get('report_decisions',''), '',
          'Per-change evidence: `results/opt1b.json`. No commits; stopped after OPT1B.']
report='\n'.join(lines)+'\n'
assert len(report.split()) <=350, len(report.split())
Path('OPT1B-REPORT.md').write_text(report)
