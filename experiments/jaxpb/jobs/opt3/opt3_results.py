"""Local evidence transport, verification, and phase-3 report generation; no MC/PB."""
import ast
import datetime
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

T='/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline'
rows=json.load(open('results/opt3-jobs.json'))
archive=Path('results/opt3-evidence.tar.gz')
with archive.open('wb') as fo:
    subprocess.run(['ssh','Coulson-Chemistry',f'tar czf - -C {T}/opt3 .'],stdout=fo,check=True)
with tarfile.open(archive) as tf:
    data={Path(m.name).name:json.load(tf.extractfile(m)) for m in tf.getmembers() if m.isfile() and m.name.endswith('.json')}
# Keep the rejected initial scalar runs distinct from the final gated kernel.
prototype={n:data[n+'.json'] for n in ('cpu12','cpu1','timing12','timing1')}
selection={'cpu12':'network12','cpu1':'network1','timing12':'network-timing12',
           'timing1':'network-timing1','units':'network-units'}
for logical,physical in selection.items(): data[logical+'.json']=data[physical+'.json']
required=['baseline12','cpu12','cpu1','gpu','units','timing12','timing1','final','trials']
assert all(data[x+'.json'].get('passed',True) for x in required), 'unfinished or failed gate'
b,c,f,g=(data[x+'.json'] for x in ('baseline12','cpu12','cpu1','gpu'))
def run(d,label): return next(x for x in d['runs'] if x['label']==label)
def compare(a,b,fields=('outputs','raw_counts','shard_fields')):
    return {k:a[k]==b[k] for k in fields}
gate1={label:compare(run(b,label),run(c,label)) for label in ('50k','200k')}
prototype_checks={label:compare(run(b,label),run(prototype['cpu12'],label)) for label in ('50k','200k')}
assert all(data[n+'.json']['energy_hashes']==b['energy_hashes'] for n in ('cpu12','cpu1','timing12','timing1','gpu')), 'saved input energy drift'
fresh={label:compare(run(c,label),run(f,label)) for label in ('50k','200k')}
fresh_all200k=compare(run(data['timing1.json'],'200k-all'),run(data['timing12.json'],'200k-all'))
flat=all(all(x['compiles']==(1 if r['label']=='warm' else 0) for r in data[n+'.json']['runs'] for x in r['buckets']) and data[n+'.json']['compiles']['n_compiled']==5 for n in ('cpu12','cpu1','timing12','timing1'))
final=data['final.json']
gates=dict(cpu_bitwise=dict(passed=all(all(x.values()) for x in gate1.values()),checks=gate1,
                           states_50k=len(run(c,'50k')['states']),states_200k=len(run(c,'200k')['states'])),
           gpu_unchanged=dict(passed=g['pallas_unchanged'] and g['ref_unchanged'],states=20,sweeps=50000,
                              pallas=g['pallas_unchanged'],ref=g['ref_unchanged']),
           determinism=dict(passed=c['repeat_exact'] and c['repeat_200k_exact'] and all(all(x.values()) for x in fresh.values()) and final['passed'] and all(fresh_all200k.values()),
                            fresh_200k_20_across_cpu_allocations=fresh_all200k,two_runs_50k=c['repeat_exact'],two_runs_200k=c['repeat_200k_exact'],fresh_process=fresh,final_auto_api=final['passed']),
           flat_compiles=dict(passed=flat,cpu=c['compiles'],warm_bucket_count=5,warm_after_first=0),
           unit_tests=dict(passed=data['units.json']['passed'] and final['new_tests']['passed'] and final['existing_tests']['passed'] and final['api_tests'],
                           existing='tests/test_titrate.py local',new='tests/test_opt3.py',api='tests/test_api.py'))
assert all(x['passed'] for x in gates.values()),gates

# Source audit: only the requested MC/API files may differ; the GPU functions are byte-identical.
source={}
for p in Path('src/jaxpb').glob('*.py'):
    old=subprocess.check_output(['git','show','a5d5c1d:'+str(p)])
    now=p.read_bytes()
    source[str(p)]=dict(baseline=hashlib.sha256(old).hexdigest(),final=hashlib.sha256(now).hexdigest(),unchanged=old==now)
assert all(v['unchanged'] for k,v in source.items() if k not in ('src/jaxpb/titrate.py','src/jaxpb/api.py'))
old=subprocess.check_output(['git','show','a5d5c1d:src/jaxpb/titrate.py'],text=True)
now=Path('src/jaxpb/titrate.py').read_text()
def funcs(s): return {f.name:ast.get_source_segment(s,f) for f in ast.parse(s).body if isinstance(f,ast.FunctionDef)}
oldf,newf=funcs(old),funcs(now)
preserved={k:oldf[k]==newf[k] for k in oldf if k != 'titrate'}
assert all(preserved.values())
# Check each immutable frozen copy against the manifest recorded before submission.
frozen_checks={}
for root in sorted({r['frozen'] for r in rows}):
    expected=next(r['hashes'] for r in rows if r['frozen']==root)
    import shlex
    got=subprocess.check_output(['ssh','Coulson-Chemistry',f'cd {root} && sha256sum '+shlex.join(expected)],text=True)
    frozen_checks[root]={line.split()[1]:line.split()[0] for line in got.splitlines()}==expected
assert all(frozen_checks.values())
source_audit=dict(files=source,preserved_functions=preserved,immutable_copies=frozen_checks,
                  final_validated_source={k:source[k]['final'] for k in ('src/jaxpb/titrate.py','src/jaxpb/api.py')})
final_job=next(x for x in rows if x['job']==final['job'])
assert all(final_job['hashes'][k]==v for k,v in source_audit['final_validated_source'].items())
Path('results/opt3-source-audit.json').write_text(json.dumps(source_audit,indent=2)+'\n')

ids=','.join(r['job'] for r in rows)
fields=['JobIDRaw','JobName','Partition','ReqCPUS','AllocCPUS','ReqMem','NodeList','State','ExitCode','Elapsed','AllocTRES']
cmd=['ssh','Coulson-Chemistry',f'sacct -X -n -P -j {ids} -o JobIDRaw,JobName%40,Partition,ReqCPUS,AllocCPUS,ReqMem,NodeList,State,ExitCode,Elapsed,AllocTRES%250']
accounting=subprocess.check_output(cmd,text=True)
acct=[dict(zip(fields,line.split('|'))) for line in accounting.splitlines() if line.strip()]
assert {r['JobIDRaw'] for r in acct}=={r['job'] for r in rows}
resource_checks=[]
for ar in acct:
    submitted=next(x for x in rows if x['job']==ar['JobIDRaw'])
    import re
    def mem_bytes(tres):
        value=dict(x.split('=',1) for x in tres.split(','))['mem']
        m=re.fullmatch(r'(\d+(?:\.\d+)?)([KMGTPE]?)',value)
        assert m,value
        return int(float(m[1]) * 1024 ** ('KMGTPE'.index(m[2])+1 if m[2] else 0))
    alloc_mem=mem_bytes(ar['AllocTRES'])
    ok=(int(ar['AllocCPUS'])<=12 and int(ar['ReqCPUS'])==submitted['cpus']
        and '--mem-per-cpu=2G' in shlex.split(submitted['command'])
        and alloc_mem==int(ar['AllocCPUS'])*2*1024**3 and
        ar['Partition']==('oc' if submitted['gpu'] else 'amd96') and
        (not submitted['gpu'] or (ar['NodeList']=='comp1400' and 'gres/gpu=1' in ar['AllocTRES'].split(','))))
    resource_checks.append(ok)
    assert ar['State'] not in ('RUNNING','PENDING','COMPLETING')
assert all(resource_checks)
maxrss=subprocess.check_output(['ssh','Coulson-Chemistry',f'sacct -n -P -j {ids} -o JobIDRaw,MaxRSS,MaxVMSize'],text=True)
audit=dict(passed=True,allocation_note='amd96 rounds one requested task CPU to two allocated CPUs (one task CPU/affinity=1). Allocated memory is 2 GiB per allocated CPU; each command explicitly requests --mem-per-cpu=2G.',accounting=acct,step_memory=maxrss,resource_checks=resource_checks,
           transport='ssh Coulson-Chemistry only',environment='envs/ogqt-main-7c3e3e9; existing jobs/a40site on A40',
           software_changes='none; numba checked absent, no installs/downloads/env creation',
           scientific_compute='Slurm amd96 / oc comp1400 only',unrelated_jobs_touched=False)
Path('results/opt3-compute-audit.json').write_text(json.dumps(audit,indent=2)+'\n')

timings={}
for name in ('cpu1','cpu12','baseline12','timing1','timing12'):
    d=data[name+'.json']
    timings[name]=dict(job=d['job'],frozen=d['frozen'],cpus=d['env']['affinity'],runs=[{k:r[k] for k in ('label','sweeps','states','S','wall_s','s_per_state','buckets')} for r in d['runs']],
                       compiles=d['compiles'],compiled_memory=d.get('compiled_memory'),host_hwm_kb=d.get('host_hwm_kb'))
timings['a40']={x['label']:{k:x[k] for k in ('wall_s','s_per_state','buckets')} for x in g['runs']}
t1,t12=(run(data[x+'.json'],'200k-all')['s_per_state'] for x in ('timing1','timing12'))
peakcpu=max(data[x+'.json']['host_hwm_kb'] for x in ('cpu1','cpu12','timing1','timing12'))*1024/1e9
benchmarks=dict(cpu_200k_20_measured=dict(cpu1=t1,cpu12=t12,speedup_1_to_12=t1/t12),
                pypka_single_core_mc=222.,pypka_source='BUILD-OPT3: bench.json 538 total - 179 PB - 137 assembly',
                speedup_vs_pypka_cpu1=222/t1,speedup_vs_pypka_cpu12=222/t12,
                old_jaxpb_cpu12_200k_20_profile2_projection=588.2,
                speedup_vs_profile2_projection=588.2/t12,
                matched_5_cpu12_old=run(b,'200k')['s_per_state'],matched_5_cpu12_new=run(c,'200k')['s_per_state'])
trial=data['trials.json']
trial['method']='2k counted sweeps + 1k equilibrium, seed=1/counter chain keys, fixed bucket shapes. Shared device views and first-chain compile precede timing; two warm runs per variant, second reported. CPU ref S=1 avoids padded baseline lanes.'
assert all(len(trial['candidates'][n])==5 for n in ('branch','index','site','pair'))
assert all(v['equal'] for v in trial['candidates']['site'].values())
reverts=[dict(change='scalar jnp.sum and current-index-only/unchanged scalar chains',reason='200k raw counts differ in two states; one curve differs by one count at pH=6.75, site=105, 70c9c3eaa126c736_AB. Batched and scalar CPU reduction lowering differ.',evidence=data['exact.json']),
         dict(change='unrestricted pruned tree',reason='Larger buckets and >8 microstates/site do not share the validated reduction tree; dispatch these to ref.',evidence=data['physical-tree.json']),
         dict(change='generic lax.sort',reason='Fixed integer sorting network is bitwise equivalent and faster.',evidence=data['sort.json']),
         dict(change='four-index pair energy difference',reason='f32 reassociation differs by one ULP; retained original masked reduction',evidence=data['pair-rounding.json']),
         dict(change='reduction rank padding / repeated reduction lanes',reason='Rank padding alone does not restore the batched reduction; repeated lanes add work. Pruning chosen in the measured supported domain.',evidence=data['tile.json']),
         dict(change='Numba option',reason='not importable in required environment; not installed')]
failures=[dict(job=x['JobIDRaw'],state=x['State'],exit_code=x['ExitCode']) for x in acct if x['State']!='COMPLETED']
result=dict(milestone='OPT3',date=datetime.date.today().isoformat(),status='complete through phase 3',baseline='a5d5c1d',committed=False,
            design='Scalar XLA CPU chains, <=12 threads sharing W; direct current/site indices; real n/P; pruned reference pair tree with a five-comparison integer sorting network; E/X/count accumulation unchanged. Fast path only Linux AVX2/JAX 0.11.1, Mp64..1024, <=8 microstates/site and canonical 73 pH grid. Other inputs use ref.',
            default='auto: Pallas on supported GPU; cpu on validated Linux AVX2/JAX0.11.1 CPU; ref otherwise. CPU dispatcher retains ref for unvalidated shapes/cardinalities. Explicit API xla remains ref MC.',
            gate_method='Same saved bench energy pack and seed=1/alias chain_keys on each arm. Byte hashes of raw counts (real microstates), prot/tot/curves/pKa, actual label-writer ragged fields, status and titration diagnostics. MC-only label shards leave energy payloads empty; provenance/timing metadata is excluded. PB is not rerun.',
            inputs=dict(energy_paths=b['energy_paths'],energy_hashes=b['energy_hashes'],seed=1,key_scheme='chain_keys(seed, chain_id_of(alias)*73+pH_index)'),
            gates=gates,timings=timings,benchmarks=benchmarks,per_change_measurements=trial,reverts=reverts,
            memory_peak_cpu_mc_GB=peakcpu,
            a40_timing_note='50k comparison only. Old baseline is warm; new module calls include one first compile per bucket. GPU kernel sources are byte-identical; these calls test output identity, not a speed change.',
            memory={x:dict(host_hwm_kb=data[x+'.json'].get('host_hwm_kb'),compiled=data[x+'.json'].get('compiled_memory')) for x in ('cpu1','cpu12','timing1','timing12')},
            jobs=[{k:v for k,v in x.items() if k!='hashes'} for x in rows],non_completed_jobs=failures,prototype_evaluation=dict(status='rejected; timing figures are not the final backend',checks_against_reference=prototype_checks),prototype_runs=prototype,retirement_decisions=json.load(open('results/opt3-decisions.json')),
            failure_explanation='Initialization: two old-module attribute errors, three incomplete CPU energy pack assertions, one diagnostic syntax error. Generic-domain sparse/rank checks intentionally failed and those formulations were rejected. Three own superseded lax.sort measurements were cancelled. Every final gate is from the selected network source; PB was never rerun.',
            frozen_paths=sorted({x['frozen'] for x in rows}),source_audit='results/opt3-source-audit.json',compute_audit='results/opt3-compute-audit.json',
            evidence=str(archive),polls='results/opt3-squeue.jsonl',
            limitations=['Validated on JAX 0.11.1 / EPYC 9224, buckets 64..1024; larger buckets and other CPU/XLA versions require regression gates.',
                         'Fast pair differences use a pruned tree under a bounded domain; unvalidated cases call ref. Accepted E/X/count vector updates remain bucket-sized.',
                         'Timings are non-exclusive Slurm allocations; profile2 old 20-state 200k figure is a full-bucket projection, unlike the measured current 20-state runs.'],
            phase4='CPU geometry and remaining maps cost remain unchanged; OPT2 CPU mean geometry 44.35 and maps 248.18 s/state. Preserve numerical ordering and bucket bounds.')
Path('results/opt3.json').write_text(json.dumps(result,indent=2)+'\n')
temp=data['timing12.json']['compiled_memory']['1024']['temp_size_in_bytes']
report=f'''# OPT3 — phase 3

Added `cpu`: scalar XLA chains on ≤12 host threads sharing W, indexed current/site access, real n/P loops and rejection skipping. A fixed sorting network prunes the reference pair-sum tree. E/X/count order, RNG, BLOCK structure and acceptance are preserved. Fast path: validated Linux AVX2/JAX 0.11.1, Mp64–1024, ≤8 microstates/site. Other inputs retain ref. `auto` uses this gated CPU dispatcher; GPU selection is unchanged.

| Gate | Result |
|---|---|
| CPU vs a5d5c1d ref | Bitwise: 20/50k + 5/200k; raw counts, curves, pKa, titration shard fields |
| A40 Pallas / ref | Both bitwise: 20/50k |
| Determinism | Two runs, fresh 1-CPU process, final public auto/API |
| Compiles | Five bucket executables; zero later compiles |
| Tests | Existing titration, new pair/odd/single-site/brute/fallback, API shard/resume: pass |

Measured 200k MC seconds/state, all 20 states: **{t1:.2f} (1 CPU), {t12:.2f} (12 CPUs)**; scaling **{t1/t12:.2f}×**. PyPKA single-core: 222 s/state (speedup ratios {222/t1:.2f}× / {222/t12:.2f}×). Old PROFILE2 CPU12 projection: 588.2 s/state ({588.2/t12:.2f}×). Matched five-state CPU12: {run(b,'200k')['s_per_state']:.2f}→{run(c,'200k')['s_per_state']:.2f}. Allocations were non-exclusive. One task CPU has affinity 1; Slurm reserves two CPUs.

Reverted ordinary scalar reductions: raw counts differed in two states at 200k; one curve differed by one count at pH 6.75/site 105 of 70c9c3eaa126c736_AB. Naive four-term sums also change rounding. Unsupported domains fall back; generic sorting was slower. Numba was absent; nothing installed.

Peak CPU MC RSS: {peakcpu:.2f} GB; Mp1024 compiled temporary: {temp:,} bytes/chain. Runtime/compiler changes require new gates.

Phase 4: unchanged CPU geometry 44.35 s/state and maps 248.18 dominate; preserve numerical ordering. Per-bucket measurements, jobs, frozen paths and reverts: `results/opt3.json` and its evidence archive. No commits; stopped after phase 3.
'''
assert len(report.split())<=300,len(report.split())
Path('OPT3-REPORT.md').write_text(report)
print(json.dumps(dict(gates={k:v['passed'] for k,v in gates.items()},benchmarks=benchmarks,report_words=len(report.split())),indent=2))
