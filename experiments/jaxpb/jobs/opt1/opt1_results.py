"""Summarize downloaded opt1 JSON evidence without importing JAX. Local report-only work."""
import argparse
import json
from pathlib import Path
from statistics import mean

BENCH = [f'{c}_{s}' for c in ['4066edfcdb3ea8e3', '70c9c3eaa126c736', '2db42a83bc0b799f',
                             '29d8e3cd7c0c56e3', 'd0d5c619bd23b13d'] for s in ['AB', 'A', 'B']] + [
    'c84a01ff31e41ea4_AB', '0dbad2ec4e61686e_AB', '4de4ae106157408f_A', '24d02660df3ac695_A', 'd6004cf2ee4008d2_AB']


def summarize(data, ids=None):
    pp = data['passes'][-1]
    rows = [r for sid, r in pp['rows'].items() if ids is None or sid in ids]
    allrows = [r for p in data['passes'] for r in p['rows'].values()]
    stages = {k: mean(r['stages'].get(k, 0.) for r in rows) for k in set().union(*(r['stages'] for r in rows))}
    hostpeak = max([r['host_hwm_kb'] for r in allrows] + [v for r in allrows for m in r['memory'].values()
                   for k, v in m.items() if k.startswith('hwm_kb')]) * 1024
    failed = [(sid, check, field) for p in data['passes'] for sid, row in p['rows'].items()
              for check, fields in row['checks'].items() for field, ok in fields.items() if not ok]
    cpu = data['env']['backend'] == 'cpu'
    warm_rss = max([0]+[v for r in allrows for m in r['memory'].values() for k,v in m.items() if k=='hwm_kb_warm'])*1024
    cold_rss = max([0]+[v for r in allrows for m in r['memory'].values() for k,v in m.items() if k=='hwm_kb_compiling'])*1024
    return dict(states=len(rows), passes=len(data['passes']), passed=data.get('passed'), failed=failed,
                exact_comparisons=sum(len(v) for r in allrows for v in r['checks'].values()),
                legacy_reference_states=sum(r['legacy_energy'] is not None for r in rows),
                legacy_reference_passed=all(r['legacy_energy'] is not False for r in rows),
                seconds_per_state=dict(geometry=stages.get('geometry', 0.), maps=stages.get('maps', 0.),
                    setup=stages.get('setup1', 0.)+stages.get('setup2', 0.),
                    sor=stages.get('sor1', 0.)+stages.get('sor2', 0.),
                    total_pb=sum(v for k, v in stages.items() if k != 'assembly') if cpu else mean(r['timing']['t_pb'] for r in rows),
                    stages=stages),
                peak_host_rss_bytes=hostpeak, peak_host_rss_warm_bytes=warm_rss, peak_host_rss_compiling_bytes=cold_rss, peak_gpu_bytes=max(r['gpu'][1] for r in allrows),
                compiles=data['compiles']['n_compiled'], compiles_per_pass=[p['new_compiles'] for p in data['passes']],
                env=data['env'], code_sha=data['code_sha'])



def write_report(res, path):
    runs = res['runs']
    lines = [
        '# OPT1 — phase 1', '',
        '`api.MemTrack` initializes lazily; `_run_shard` keeps bounded timing aggregates. `pb._solve_state` bounds promotion, '
        'uses host `spec` and releases buffers. `maps.build_cells`, `_q_one`, `maps_batch`, `reach_mask_b` use CSR ranges, '
        'ranked candidate slabs, byte budgets, fixed CPU groups and padding skips. `coulomb_bc`/`setup1`/`setup2` evaluate '
        '38,402 faces only and select backend operators. Timing readers stream shard records.', '',
        '| Gate | Result |', '|---|---|',
        '| Pallas energies, potentials, sweeps | Exact: 104 L3 + 20 bench, A/B/cumulative |',
        '| XLA / CPU | Exact: 12 / 5 final states; A-only CPU: 4 exact, fifth OOM |',
        '| Determinism | Exact: two 104-state passes + fresh process |',
        '| Compiles / CPU tests | Zero warm compiles; four existing modes + seven edge tests pass |',
        '| Final CPU source | Identical StableHLO at all 10 normal/XL variants |', '',
        'Seconds/state: geometry/maps/setup/SOR/PB. GPU: instrumented warm passes; CPU: compiler-subtracted stage totals. '
        'Baselines: PROFILE2 fine timings; XLA memory: matched control. GB are decimal.', '',
        '| Arm/change | g / m / setup / SOR / PB | GPU / host GB |', '|---|---|---|']
    tables = res['profile2_baseline']['tables']
    for arm, key, basekey in [('Pallas20','pallas','A40_pallas_20_states'),
                             ('XLA12','xla','A40_xla_fine_12_states'),
                             ('CPU5/12cores','cpu','cpu12_xla_fine_5_states')]:
        b = tables[basekey]
        sor = b['SOR (phase1 coarse+model)']['s'] + b['SOR (phase2 focused)']['s']
        total = sum(v['s'] for k,v in b.items() if isinstance(v,dict) and k not in
                    ('prep_load','assembly','mc_200k') and (key!='cpu' or k!='pb_host_other (layout, dispatch)'))
        vals = [b['geometry']['s'], b['maps_batch']['s'], b['solve_setup (setup1+setup2)']['s'], sor, total]
        mem = ('16.15/—' if key == 'pallas' else '—/21.2–25.1' if key == 'cpu' else
               f"{runs['baseline-xla']['peak_gpu_bytes']/1e9:.2f}/{runs['baseline-xla']['peak_host_rss_bytes']/1e9:.2f}")
        lines.append('| '+arm+' baseline | '+'/'.join(f'{v:.2f}' for v in vals)+' | '+mem+' |')
        for stage,prefix in [('A','A'),('A+B','B9'),('A+B+C','C9')]:
            tag = (('B8' if prefix=='B9' else 'C8' if prefix=='C9' else prefix)+'-cpu') if key=='cpu' else prefix+'-'+key
            r = runs[tag]; t = r['seconds_per_state']
            vals = [t[k] for k in ('geometry','maps','setup','sor','total_pb')]
            mr = runs[prefix+'-all'] if key=='pallas' and prefix in ('A','B9') else r
            mem = ('—' if key=='cpu' else f"{mr['peak_gpu_bytes']/1e9:.2f}")+f"/{mr['peak_host_rss_bytes']/1e9:.2f}"
            label = 'CPU4+OOM' if key=='cpu' and prefix=='A' else arm
            lines.append('| '+label+' '+stage+' | '+'/'.join(f'{v:.2f}' for v in vals)+' | '+mem+' |')
    p = runs['C9-pallas']; x = runs['C9-xla']
    lines += ['',
        f"Compiles: Pallas {p['compiles']}/104 ({runs['B9-pallas']['compiles']}/bench); XLA {x['compiles']}/12. "
        f"CPU {runs['C8-cpu']['compiles']}/5. BUILD’s 40/33 covers different sets; matched controls: 35/36/34.", '',
        f"GPU PB change: {100*(runs['C9-pallas']['seconds_per_state']['total_pb']/sum(v['s'] for k,v in tables['A40_pallas_20_states'].items() if isinstance(v,dict) and k not in ('prep_load','assembly','mc_200k'))-1):+.1f}%. "
        'Reverted CPU single-site tail maps after model-energy bit differences; query/reach skips retain group-of-four '
        f"projection arithmetic. A-only CPU OOM was resolved by B/C. CPU peak {runs['C8-cpu']['peak_host_rss_bytes']/1e9:.2f} GB misses the 12 GB target; SOR retains the peak. Phase 2: colour-compressed XLA SOR, preserving arithmetic and ten-sweep checks.", '',
        'Evidence, job IDs, frozen paths and per-change deltas: `results/opt1.json` and `results/opt1-evidence.tar.gz`. '
        'Stopped after phase 1; no commits.', '']
    report='\n'.join(lines)
    assert len(report.split()) <= 400, len(report.split())
    Path(path).write_text(report)


def main():
    p = argparse.ArgumentParser(); p.add_argument('evidence'); p.add_argument('output'); a = p.parse_args()
    evidence = Path(a.evidence)
    res = dict(phase=1, status='in_progress', compute_rules=dict(cluster='Coulson-Chemistry', gpu_node='comp1400',
        gpu='a40:1', gpu_partition='oc', cpu_partition='amd96', cpus_per_job=12,
        memory_per_cpu='2G', baseline_cpu_memory_per_cpu='4G', installs=0, commits=0),
        timing_notes=['GPU: instrumented warm pass, production PB total.',
                      'CPU: one instrumented pass, compilation subtracted per stage; PB is sum of stages, excluding host overhead.',
                      'GPU allocator peaks are process-wide; host peaks include compilation and all measured stages.'],
        runs={})
    raw = {}
    for f in sorted(evidence.glob('*.json')):
        data = json.load(open(f))
        if f.stem == 'A-cpu' and not data.get('passes') and (evidence/'A-cpu-oom.json').exists():
            rows=data['current']['rows']
            count=max(r['compiles_total'] for r in rows.values())
            data['passes']=[dict(rows=rows, new_compiles=count, compiles_total=count)]
            data['compiles']=dict(n_compiled=count)
            data['passed']=False
            data['failure']=json.load(open(evidence/'A-cpu-oom.json'))
        if data.get('passes') and 'compiles' in data:
            raw[f.stem] = data
            res['runs'][f.stem] = summarize(data, BENCH if data['selection'] == 'all' else None)
            if data.get('failure'):
                res['runs'][f.stem]['failure']=data['failure']
                res['runs'][f.stem]['compiles_lower_bound']=res['runs'][f.stem]['compiles']
                res['runs'][f.stem]['compiles']=None
                res['runs'][f.stem]['compiles_per_pass']=[None]
                res['runs'][f.stem]['peak_host_rss_bytes']=max(res['runs'][f.stem]['peak_host_rss_bytes'], data['failure']['slurm_maxrss_kb']*1024)
            if data['selection'] == 'all':
                res['runs'][f.stem]['all_states'] = summarize(data)
        elif 'passes' in data:
            rows = data.get('current', {}).get('rows', {})
            res.setdefault('partial_or_superseded', {})[f.stem] = dict(env=data.get('env'),
                completed_passes=len(data['passes']), current_states=len(rows),
                current_checks_passed=all(all(all(v.values()) for v in r['checks'].values()) and r.get('legacy_energy') is not False for r in rows.values()))
        else:
            res[f.stem] = data
    # Independent CPU runs save complete PB/assembly fingerprints before the baseline finishes.
    # Compare those fingerprints here; no numerical work or JAX import is performed locally.
    if 'baseline-cpu' in raw:
        baseline = raw['baseline-cpu']['passes'][0]['rows']
        for tag in ('A-cpu', 'B8-cpu', 'C8-cpu'):
            if tag in raw:
                diffs = [(sid, field) for sid, row in raw[tag]['passes'][0]['rows'].items()
                         for field, h in row['hashes'].items() if h != baseline[sid]['hashes'][field]]
                res['runs'][tag]['posthoc_baseline_check'] = dict(passed=not diffs and len(raw[tag]['passes'][0]['rows']) == len(baseline),
                    bitwise_passed=not diffs, states=len(raw[tag]['passes'][0]['rows']), expected_states=len(baseline),
                    fields=sorted(next(iter(baseline.values()))['hashes']), differences=diffs)
    required = ('baseline-pallas', 'baseline-xla', 'baseline-cpu', 'A-pallas', 'A-xla', 'A-cpu', 'A-all',
                'B9-pallas', 'B9-xla', 'B8-cpu', 'B9-all', 'C9-pallas', 'C9-xla', 'C8-cpu', 'C9-fresh')
    res['required_runs'] = list(required)
    res['pending_runs'] = [k for k in required if k not in res['runs']]
    res['historical_compile_reference'] = dict(build_opt1=dict(xla=40, pallas=33),
        profile2=dict(xla=44, pallas=33), note='Coverage differs: current matched XLA gate uses 12 states; Pallas full gate uses 104.')
    res['gates'] = {}
    for stage, prefix in (('A', 'A'), ('B', 'B9'), ('cumulative', 'C9')):
        fulltag = prefix+'-pallas' if stage == 'cumulative' else prefix+'-all'
        def exact(tag):
            r = res['runs'].get(tag, {})
            return bool(r.get('passed') and not r.get('failed')) if r else None
        cr = res['runs'].get(('B8' if prefix=='B9' else 'C8' if prefix=='C9' else prefix)+'-cpu', {})
        res['gates'][stage] = dict(pallas_104=exact(fulltag), pallas_bench20=exact(prefix+'-pallas'),
            xla_12=exact(prefix+'-xla'), cpu_5=cr.get('posthoc_baseline_check', {}).get('passed'),
            warm_compile_flat=all(res['runs'][prefix+'-'+be]['compiles_per_pass'][-1] == 0 for be in ('pallas', 'xla'))
                if all(prefix+'-'+be in res['runs'] for be in ('pallas', 'xla')) else None)
    if 'C9-pallas' in raw and 'C9-fresh' in raw:
        res['gates']['determinism'] = bool(len(raw['C9-pallas']['passes']) == 2 and raw['C9-pallas']['passed']
            and raw['C9-fresh']['passed'])
    res['gates']['cpu_unit_tests'] = res.get('unit-tests', {}).get('passed')
    res['gates']['edge_tests'] = res.get('edge-tests', {}).get('passed')
    res['gates']['final_source_consistent'] = res.get('cpu-final-source-equivalence', {}).get('passed')
    res['gates']['compute_policy'] = res.get('compute-audit', {}).get('passed')
    res['gates']['source_audit'] = res.get('source-audit', {}).get('passed')
    res['gates']['summary_reader_tests'] = res.get('summary-reader-tests', {}).get('passed')
    res['final_source'] = dict(frozen='opt1-C9-20261121', cpu_runs=['B8-cpu','C8-cpu'],
        gpu_runs='C9/B9: direct numerical gates on the final source', cpu_proof='C8/C9 identical StableHLO for all normal/XL classes; all other solver sources byte-identical')
    if not res['pending_runs']:
        # Hard gates apply to the final cumulative source. Intermediate A's known baseline-memory OOM
        # remains explicit; B/C must resolve it under the same 2G/CPU limit.
        checks = [v for k,g in res['gates'].items() for n,v in (g.items() if isinstance(g,dict) else [('result',g)])
                  if not (k=='A' and n=='cpu_5' and 'A-cpu-oom' in res)]
        good = all(v is True for v in checks)
        res['status'] = 'pass' if good else 'gate_failure'
        res['intermediate_A_cpu_oom_resolved'] = bool(good and res['gates']['cumulative']['cpu_5'])
    res['changes'] = {
        'A': dict(functions=['pb._solve_state', 'api.MemTrack', 'api._run_shard'],
                  changes=['bounded class promotion', 'streamed downstream timing readers', 'host spectral radii', 'timing-only barriers',
                           'lazy memory tracker', 'bounded state timing aggregates'], runs=['A-pallas','A-xla','A-cpu','A-all']),
        'B': dict(functions=['maps.build_cells','maps._q_one','maps.maps_batch','maps.reach_mask_b','pb._site_layout'],
                  changes=['site-specific CSR ranges', 'occupied 32-slot slabs with original candidate ranks',
                           'byte budgets including site groups', 'fixed CPU group of 4', 'empty padding and skipped cell setup'],
                  runs=['B9-pallas','B9-xla','B8-cpu','B9-all']),
        'cumulative': dict(functions=['pb.coulomb_bc','pb.face_points','pb.setup1','pb.setup2','pb._solve_state'],
                  changes=['38402 face points', 'point-only Coulomb tiles; charge reduction unchanged',
                           'face-only focused interpolation', 'finite backend setup variants', 'release consumed buffers'],
                  runs=['C9-pallas','C9-xla','C8-cpu','C9-fresh'])}
    res['tile_budgets_bytes'] = dict(cpu_query=64<<20, cpu_reach=32<<20, gpu_query=512<<20, gpu_reach=256<<20,
                                    query_estimate_per_candidate=48, reach_estimate_per_candidate=64,
                                    max_query_rows=dict(atom=4096, probe=1024, brute=65536))
    profile = json.load(open(Path(a.output).parent / 'profile2.json'))
    res['profile2_baseline'] = dict(tables=profile['tables_s_per_state'], memory=profile['memory'])
    res['comparisons'] = {}
    tables = profile['tables_s_per_state']
    for arm, bk in [('pallas','A40_pallas_20_states'),('xla','A40_xla_fine_12_states'),('cpu','cpu12_xla_fine_5_states')]:
        b = tables[bk]
        before = dict(geometry=b['geometry']['s'], maps=b['maps_batch']['s'],
            setup=b['solve_setup (setup1+setup2)']['s'],
            sor=b['SOR (phase1 coarse+model)']['s']+b['SOR (phase2 focused)']['s'])
        before['total_pb'] = sum(v['s'] for k,v in b.items() if isinstance(v,dict) and k not in
                ('prep_load','assembly','mc_200k') and (arm!='cpu' or k!='pb_host_other (layout, dispatch)'))
        baseline = dict(before)
        expected_states={'pallas':20,'xla':12,'cpu':5}[arm]
        previous_states=expected_states
        rows = {}
        for stage, prefix in [('A','A'),('B','B9'),('C','C9')]:
            tag = (('B8' if prefix=='B9' else 'C8' if prefix=='C9' else prefix)+'-cpu') if arm=='cpu' else prefix+'-'+arm
            if tag not in res['runs']:
                continue
            after=res['runs'][tag]['seconds_per_state']
            rows[stage] = dict(run=tag, states=res['runs'][tag]['states'], cumulative_seconds_per_state={k:after[k] for k in baseline},
                incremental_delta_seconds={k:after[k]-before[k] for k in baseline} if res['runs'][tag]['states']==previous_states else None,
                delta_vs_profile2_seconds={k:after[k]-baseline[k] for k in baseline} if res['runs'][tag]['states']==expected_states else None,
                ratio_vs_profile2={k:after[k]/baseline[k] for k in baseline} if res['runs'][tag]['states']==expected_states else None)
            before=after
            previous_states=res['runs'][tag]['states']
        res['comparisons'][arm] = dict(profile2_seconds_per_state=baseline, changes=rows)
        if arm=='pallas' and 'C9-pallas' in res['runs']:
            prod=tables['A40_pallas_production_split_20_states']['t_pb']
            res['comparisons'][arm]['production_reference']=dict(total_pb=prod, ratio=res['runs']['C9-pallas']['seconds_per_state']['total_pb']/prod)
    res['reverted'] = ['CPU one-site tail map changed esolvS; retain B=4 arithmetic and skip padding inside query/reach loops.']
    res['superseded_tuning'] = ['Conservative GPU query/reach tiles; replaced by fixed 512/256 MiB budgets.',
        'Unstreamed CSR probe slots; replaced by occupied 32-slot slabs with original candidate ranks.']
    if 'C8-cpu' in res['runs']:
        c=res['runs']['C8-cpu']
        res['cpu_memory_target'] = dict(target_bytes=12000000000, peak=c['peak_host_rss_bytes'], warm_peak=c['peak_host_rss_warm_bytes'],
            full_peak_under_target=c['peak_host_rss_bytes']<12000000000, warm_peak_under_target=c['peak_host_rss_warm_bytes']<12000000000)
    res['phase2_first'] = 'Colour-compressed XLA SOR; preserve the floating-point addition order and ten-sweep stopping checks.'

    res['per_change_gates']={k:res['gates'].pop(k) for k in ('A','B','cumulative')}
    res['gates']={**res['per_change_gates']['cumulative'], **res['gates']}
    if res['status']=='pass':
        from datetime import datetime, timezone
        import hashlib
        res['completed_utc']=datetime.now(timezone.utc).isoformat()
        archive=Path(a.output).parent/'opt1-evidence.tar.gz'
        if archive.exists():
            res['evidence_archive']=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    json.dump(res, open(a.output, 'w'), indent=2)
    if res['status'] == 'pass':
        write_report(res, Path(a.output).parent.parent / 'OPT1-REPORT.md')


if __name__ == '__main__':
    main()
