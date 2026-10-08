"""JAX-Ka production-1024-v2: JAX-only rerun on the frozen production-1024-v1 inputs.

v1 deferred JAX-Ka after workers were OOM-killed above ~460 residues. v2 uses
the memory-reduced execution path (production_worker_v2) with unchanged
equations, solver settings, grid and validity rules. Release requires every
state of all 50 accepted smoke complexes to reproduce the accepted 1024-step
predictions (status exact; curves 1e-6; midpoints 1e-4). Scoring against the
other methods is a separate, explicit step; this module produces receipts and
predictions only.
"""
import fcntl
import json
import os
import subprocess
import sys
import time
import shutil
from collections import Counter
from pathlib import Path
from .runtime import require_compute, atomic_json, digest, config_hash
from .schema import read_table, write_table, key

VERSION = 'production-1024-v2'
FILES = ('structures.parquet','sites.parquet','assignments.parquet','site_masks.parquet','input-files.json','pilot.json')
CURVE_ATOL, PKA_ATOL = 1e-6, 1e-4


def runtime():
    return Path(os.environ['PKABENCH_RUNTIME'])


def source_campaign():
    return runtime()/'campaigns/production-1024-v1'


def smoke_campaign():
    return runtime()/'campaigns/frozen-smoke-jax1024-v1'


def code_hashes():
    base = Path(__file__).parent
    names = ['production_jax_v2.py','production_worker_v2.py','production.py','adapters/base.py','prep.py','schema.py','runtime.py']
    result = {name: digest(base/name) for name in names}
    result.update({'jaxpropka/'+p.name: digest(p) for p in sorted((base.parent/'jaxpropka').glob('*.py'))})
    return result


def initialise(out):
    require_compute()
    src = source_campaign(); m1 = json.loads((src/'manifest.json').read_text())
    assert digest(src/'site_masks.parquet') == m1['site_masks_sha256']
    assert digest(src/'assignments.parquet') == m1['assignments_sha256']
    assert digest(src/'pilot.json') == m1['pilot_sha256']
    smoke = smoke_campaign()
    assert json.loads((smoke/'verification.json').read_text())['passed']
    out = Path(out); out.mkdir(exist_ok=False); (out/'structures').mkdir()
    for name in FILES:
        shutil.copyfile(src/name, out/name)
    pinned = json.loads((out/'input-files.json').read_text())
    structures = read_table(out/'structures.parquet')
    for s in structures:
        cid = s['complex_id']; target = (src/'structures'/cid).resolve()
        for state, sha in pinned[cid]['state_sha256'].items():
            assert digest(target/f'{state}.cif') == sha
        (out/'structures'/cid).symlink_to(target, target_is_directory=True)
    pilot = set(json.loads((out/'pilot.json').read_text())['complex_ids'])
    tasks = [{'complex_id': s['complex_id'], 'method': 'jaxka', 'priority': 0 if s['complex_id'] in pilot else 1,
              'n_residues': s['n_residues'], 'split': s['split']} for s in structures]
    tasks.sort(key=lambda t: (t['priority'], config_hash(['v2-task-order', t['complex_id']])))
    from .production_worker_v2 import settings
    from jaxpropka.parameters import ModelConfig
    manifest = {'version': VERSION, 'method': 'jaxka', 'source_campaign': str(src),
        'source_manifest_sha256': digest(src/'manifest.json'),
        'files_sha256': {name: digest(out/name) for name in FILES},
        'site_masks_sha256': digest(out/'site_masks.parquet'),
        'code_sha256': code_hashes(), 'worker_settings': settings(ModelConfig(steps=1024)),
        'jax_steps': 1024, 'timeout_seconds': 5400, 'structures': len(structures),
        'split_counts': dict(Counter(s['split'] for s in structures)),
        'smoke_reference': str(smoke), 'smoke_manifest_sha256': digest(smoke/'manifest.json'),
        'equivalence_tolerances': {'status': 'exact', 'curve_atol': CURVE_ATOL, 'pka_atol': PKA_ATOL},
        'environment_sha256': {'runner.requirements.lock': digest(runtime()/'manifests/runner.requirements.lock')},
        'authorization': 'User approved memory-reduced v2 JAX-Ka path and equivalence gate (2026-10-05); '
                         'pool launch requires separate confirmation. 400-core cap, 2 GB/core.'}
    atomic_json(out/'manifest.json', manifest)
    atomic_json(out/'queue.json', {'next': 0, 'tasks': tasks}); atomic_json(out/'claims.json', {})
    (out/'jobs').mkdir(); (out/'validation').mkdir(); (out/'source').mkdir()
    for name in ('production_jax_v2.py', 'production_worker_v2.py'):
        shutil.copyfile(Path(__file__).with_name(name), out/'source'/name)
    print(json.dumps({'complexes': len(structures), 'tasks': len(tasks), 'pilot_first': len(pilot)}), flush=True)


def check(campaign):
    m = json.loads((campaign/'manifest.json').read_text())
    assert m['version'] == VERSION
    assert m['code_sha256'] == code_hashes(), 'v2 implementation changed'
    for name, sha in m['files_sha256'].items():
        assert digest(campaign/name) == sha
    for name, sha in m['environment_sha256'].items():
        assert digest(runtime()/'manifests'/name) == sha
    return m


def _run_states(structure_campaign, cid, work, timeout):
    """Run the v2 worker for AB/A/B in fresh processes. Returns rows, errors, timings, receipts."""
    from .adapters.base import execute
    from .production import failed_rows
    rows, errors, timings, receipts = [], {}, {}, {}
    started = time.monotonic()
    for state in ('AB', 'A', 'B'):
        log = work/f'{state}-process'; log.mkdir(parents=True); dest = work/state
        try:
            remaining = timeout-(time.monotonic()-started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(['jaxka'], timeout)
            timings[state] = execute([sys.executable, '-m', 'pkabench.production_worker_v2',
                                      str(structure_campaign), cid, state, str(dest)], log, remaining)
            receipt = json.loads((dest/'receipt.json').read_text())
            assert receipt['predictions_sha256'] == digest(dest/'predictions.parquet')
            rows.extend(read_table(dest/'predictions.parquet')); receipts[state] = receipt
        except Exception as exc:
            errors[state] = {'type': type(exc).__name__, 'detail': str(exc)}
            rows.extend(failed_rows(structure_campaign, cid, 'jaxka', state))
    return rows, errors, timings, receipts, time.monotonic()-started


def compare(rows, reference):
    """Status/curve/midpoint differences of v2 rows against reference rows (same keys)."""
    import numpy as np
    old = {(key(r), r['state']): r for r in reference}
    status_mismatch, missing, curve_max, pka_max = [], [], 0., 0.
    for r in rows:
        o = old.get((key(r), r['state']))
        if o is None:
            missing.append([*key(r), r['state']]); continue
        if r['status'] != o['status']:
            status_mismatch.append({'site': [*key(r), r['state']], 'v2': r['status'], 'reference': o['status']})
            continue
        if r['pka'] is not None and o['pka'] is not None:
            pka_max = max(pka_max, abs(r['pka']-o['pka']))
        if r['curve'] is not None and o['curve'] is not None:
            curve_max = max(curve_max, float(np.max(np.abs(np.asarray(r['curve'])-np.asarray(o['curve'])))))
    return {'rows': len(rows), 'status_mismatches': status_mismatch, 'missing_reference': missing,
            'curve_max_abs': curve_max, 'pka_max_abs': pka_max,
            'passed': not status_mismatch and not missing and curve_max <= CURVE_ATOL and pka_max <= PKA_ATOL}


def validate(campaign, cid):
    require_compute(); campaign = Path(campaign).resolve(); check(campaign)
    smoke = smoke_campaign()
    assert cid in {s['complex_id'] for s in read_table(smoke/'structures.parquet')}
    work = campaign/'validation'/cid; work.mkdir(parents=True, exist_ok=False)
    rows, errors, timings, receipts, elapsed = _run_states(smoke, cid, work, 5400)
    reference = read_table(smoke/'jobs/jaxka'/f'{cid}.parquet')
    result = compare(rows, reference) | {'complex_id': cid, 'errors': errors, 'state_seconds': timings,
        'wall_seconds': elapsed, 'peak_rss_mib': {s: r['peak_rss_mib'] for s, r in receipts.items()},
        'reference_sha256': digest(smoke/'jobs/jaxka'/f'{cid}.parquet'), 'job': os.environ['SLURM_JOB_ID']}
    result['passed'] = result['passed'] and not errors
    atomic_json(work/'result.json', result)
    print(json.dumps({k: result[k] for k in ('complex_id', 'passed', 'curve_max_abs', 'pka_max_abs', 'wall_seconds')}
                     | {'status_mismatches': len(result['status_mismatches'])}), flush=True)


def gate(campaign):
    require_compute(); campaign = Path(campaign).resolve(); check(campaign)
    smoke = smoke_campaign(); ids = sorted(s['complex_id'] for s in read_table(smoke/'structures.parquet'))
    results, absent = [], []
    for cid in ids:
        path = campaign/'validation'/cid/'result.json'
        if path.exists(): results.append(json.loads(path.read_text()))
        else: absent.append(cid)
    failed = [r['complex_id'] for r in results if not r['passed']]
    report = {'passed': not absent and not failed and len(results) == 50, 'validated': len(results),
              'absent': absent, 'failed': failed,
              'status_mismatches': sum(len(r['status_mismatches']) for r in results),
              'curve_max_abs': max([r['curve_max_abs'] for r in results] or [None]),
              'pka_max_abs': max([r['pka_max_abs'] for r in results] or [None]),
              'max_peak_rss_mib': max([max(r['peak_rss_mib'].values() or [0]) for r in results] or [None]),
              'max_wall_seconds': max([r['wall_seconds'] for r in results] or [None]),
              'tolerances': {'status': 'exact', 'curve_atol': CURVE_ATOL, 'pka_atol': PKA_ATOL},
              'manifest_sha256': digest(campaign/'manifest.json'), 'job': os.environ['SLURM_JOB_ID']}
    atomic_json(campaign/'release_gate.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('absent',)}, indent=1), flush=True)
    if not report['passed']:
        raise SystemExit('v2 release gate FAILED; pool must not be launched')


def pool(campaign):
    require_compute(); campaign = Path(campaign).resolve(); manifest = check(campaign)
    g = json.loads((campaign/'release_gate.json').read_text())
    assert g['passed'] and g['manifest_sha256'] == digest(campaign/'manifest.json')
    pinned = json.loads((campaign/'input-files.json').read_text()); started = time.monotonic()
    base = campaign/'jobs'/'jaxka'; base.mkdir(parents=True, exist_ok=True)
    while time.monotonic()-started < 21*3600:
        with (campaign/'queue.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            q = json.loads((campaign/'queue.json').read_text()); i = q['next']
            if i == len(q['tasks']): return
            task = q['tasks'][i]; q['next'] = i+1
            claims = json.loads((campaign/'claims.json').read_text())
            claims[str(i)] = {'job': os.environ['SLURM_JOB_ID'], 'task': task}
            atomic_json(campaign/'claims.json', claims); atomic_json(campaign/'queue.json', q)
        cid = task['complex_id']
        for state, sha in pinned[cid]['state_sha256'].items():
            assert digest(campaign/'structures'/cid/f'{state}.cif') == sha
        work = base/cid/f"attempt-{os.environ['SLURM_JOB_ID']}"
        rows, errors, timings, receipts, elapsed = _run_states(campaign, cid, work, manifest['timeout_seconds'])
        expected = {(key(s), st) for s in read_table(campaign/'structures'/cid/'sites.parquet') for st in ('AB', s['partner'])}
        assert {(key(r), r['state']) for r in rows} == expected and len(rows) == len(expected)
        output = base/f'{cid}.parquet'; write_table(output, 'predictions', rows)
        atomic_json(base/f'{cid}.json', {'status': 'failed' if errors else 'complete', 'errors': errors,
            'state_seconds': timings, 'extra': receipts, 'wall_seconds': elapsed, 'workdir': str(work),
            'node': os.environ['SLURMD_NODENAME'], 'job': os.environ['SLURM_JOB_ID'],
            'output_sha256': digest(output), 'manifest_sha256': digest(campaign/'manifest.json'),
            'task_index': i, 'input_state_sha256': pinned[cid]['state_sha256']})
        print(json.dumps({'task': i, 'complex_id': cid, 'status': 'failed' if errors else 'complete',
                          'seconds': elapsed}), flush=True)
    raise RuntimeError('Worker lifetime reached; remaining queue requires another bounded pool wave')


def status(campaign):
    require_compute(); campaign = Path(campaign)
    q = json.loads((campaign/'queue.json').read_text()); counts = Counter()
    for p in (campaign/'jobs/jaxka').glob('*.json'):
        counts[json.loads(p.read_text())['status']] += 1
    report = {'total_tasks': len(q['tasks']), 'claimed': q['next'], 'receipts': sum(counts.values()), 'counts': dict(counts)}
    atomic_json(campaign/'status.json', report); print(json.dumps(report, indent=2), flush=True)


def collect(campaign):
    require_compute(); campaign = Path(campaign).resolve(); check(campaign); status(campaign)
    q = json.loads((campaign/'queue.json').read_text()); rows, jobs, missing = [], [], []
    for t in q['tasks']:
        path = campaign/'jobs/jaxka'/f"{t['complex_id']}.json"
        if not path.exists(): missing.append(t); continue
        receipt = json.loads(path.read_text())
        assert receipt['manifest_sha256'] == digest(campaign/'manifest.json')
        assert digest(path.with_suffix('.parquet')) == receipt['output_sha256']
        rows.extend(read_table(path.with_suffix('.parquet'))); jobs.append(t | {'status': receipt['status'], 'errors': receipt['errors']})
    atomic_json(campaign/'merge_report.json', {'jobs': jobs, 'missing': missing})
    if missing:
        raise RuntimeError(f'{len(missing)} tasks lack receipts; recover explicitly before collection')
    write_table(campaign/'predictions.parquet', 'predictions', rows)
    # Extra evidence: agreement with complexes v1 production completed before deferral.
    v1 = source_campaign()/'jobs/jaxka'; comparisons = []
    for path in sorted(v1.glob('*.json')):
        r1 = json.loads(path.read_text())
        if r1['status'] != 'complete': continue
        cid = path.stem
        mine = [r for r in rows if r['complex_id'] == cid]
        comparisons.append(compare(mine, read_table(path.with_suffix('.parquet'))) | {'complex_id': cid})
    atomic_json(campaign/'v1_comparison.json', {'complexes': len(comparisons),
        'passed': sum(c['passed'] for c in comparisons),
        'status_mismatches': sum(len(c['status_mismatches']) for c in comparisons),
        'curve_max_abs': max([c['curve_max_abs'] for c in comparisons] or [None]),
        'pka_max_abs': max([c['pka_max_abs'] for c in comparisons] or [None]), 'details': comparisons})
    atomic_json(campaign/'completion.json', {'complete': True, 'tasks': len(jobs),
        'failed_jobs': [r for r in jobs if r['status'] != 'complete'],
        'note': 'JAX-Ka v2 predictions assembled. Combined scoring with production-nojax-v1 is a separate step.'})
    print('v2 collection completed', flush=True)
