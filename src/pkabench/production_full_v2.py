"""Full six-method score report: production-nojax-v1 plus JAX-Ka production-1024-v2.

No predictions are recomputed. Non-JAX receipts come from the verified
production-nojax-v1 report; JAX-Ka receipts from the gated, collected v2
campaign. Frozen sites, assignments and masks are taken from the non-JAX report
and checked row-for-row against the v2 campaign. Scoring uses the unchanged
frozen scorer; adding JAX-Ka shrinks all-method common support, so the
non-JAX report remains the reference for the five-method common-support table.
"""
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from .runtime import require_compute, atomic_json, digest
from .schema import read_table, write_table, key

METHODS = ['pypka','propka','jaxka','pkai','pkai_plus','null']


def _audit(out):
    """Independently recompute every reported group macro from per-complex rows."""
    with (out/'scores_per_complex.csv').open() as f:
        complexes = list(csv.DictReader(f))
    buckets = defaultdict(list)
    for r in complexes:
        buckets[tuple(r[k] for k in ('method','split','scope','subset'))].append(r)
    checks = 0
    for r in json.loads((out/'scores_set1.json').read_text()):
        rr = buckets[tuple(r[k] for k in ('method','split','scope','subset'))]
        assert sum(int(x['n']) for x in rr) == r['sites']
        for metric in ('mae','rmse','skill','spearman','sign_accuracy','error_cancellation'):
            groups = defaultdict(list)
            for x in rr:
                if x[metric] != '': groups[x['component_id']].append(float(x[metric]))
            means = [sum(v)/len(v) for v in groups.values()]
            if means: assert math.isclose(sum(means)/len(means), r[metric], rel_tol=1e-10, abs_tol=1e-10)
            else: assert r[metric] is None
            checks += 1
    return checks


def run(campaigns):
    require_compute(); campaigns = Path(campaigns).resolve()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .frozen_score import score
    from .frozen_score_secondary import run as secondary
    nojax, jax = campaigns/'production-nojax-v1', campaigns/'production-1024-v2'
    v = json.loads((nojax/'verification.json').read_text())
    assert v['passed'] and v['scope'] == 'full'
    assert digest(nojax/'predictions.parquet') == v['prediction_sha256']
    assert digest(nojax/'site_masks.parquet') == v['mask_sha256']
    assert json.loads((jax/'release_gate.json').read_text())['passed']
    jc = json.loads((jax/'completion.json').read_text()); assert jc['complete']
    jm = json.loads((jax/'manifest.json').read_text())
    # Frozen inputs must agree row-for-row between the two sources.
    for name in ('sites','site_masks','assignments'):
        a = read_table(nojax/f'{name}.parquet'); b = read_table(jax/f'{name}.parquet')
        ka = (lambda r: key(r)) if name != 'assignments' else (lambda r: r['complex_id'])
        assert {ka(r): r for r in a} == {ka(r): r for r in b}, f'{name} differs between sources'
    out = campaigns/'production-full-v2'; out.mkdir(exist_ok=False)
    for name in ('structures.parquet','sites.parquet','assignments.parquet','site_masks.parquet'):
        shutil.copyfile(nojax/name, out/name)
    (out/'structures').symlink_to((nojax/'structures').resolve(), target_is_directory=True)
    ids = [s['complex_id'] for s in read_table(out/'structures.parquet')]; assert len(ids) == 1452
    base = json.loads((nojax/'manifest.json').read_text())
    manifest = base | {'version': 'production-full-v2', 'methods': METHODS,
        'site_masks_sha256': digest(out/'site_masks.parquet'), 'assignments_sha256': digest(out/'assignments.parquet'),
        'sources': {'nojax': str(nojax), 'nojax_manifest_sha256': digest(nojax/'manifest.json'),
                    'nojax_verification_sha256': digest(nojax/'verification.json'),
                    'jaxka': str(jax), 'jaxka_manifest_sha256': digest(jax/'manifest.json'),
                    'jaxka_release_gate_sha256': digest(jax/'release_gate.json'),
                    'jaxka_version': jm['version']},
        'jaxka': 'production-1024-v2: memory-reduced execution of the unchanged 1024-step model; '
                 '50-complex equivalence gate passed against the accepted smoke.',
        'scope': 'full'}
    atomic_json(out/'manifest.json', manifest)
    rows, statuses, lineage = [], [], []
    for method in METHODS:
        src = (jax if method == 'jaxka' else nojax)/'jobs'/method
        dest = out/'jobs'/method; dest.mkdir(parents=True)
        for cid in ids:
            path = src/f'{cid}.json'; receipt = json.loads(path.read_text())
            assert digest(path.with_suffix('.parquet')) == receipt['output_sha256']
            rr = read_table(path.with_suffix('.parquet'))
            assert all(r['method'] == method and r['complex_id'] == cid for r in rr)
            for ext in ('.json','.parquet'): shutil.copyfile(path.with_suffix(ext), dest/f'{cid}{ext}')
            rows.extend(rr)
            statuses.append({'complex_id': cid, 'method': method, 'status': receipt['status'], 'errors': receipt['errors']})
            lineage.append({'receipt': str(path), 'sha256': digest(path)})
    write_table(out/'predictions.parquet', 'predictions', rows)
    atomic_json(out/'merge_report.json', {'missing': [], 'jobs': statuses})
    atomic_json(out/'derivation.json', {'sources': manifest['sources'], 'source_receipts': lineage,
                                        'report_code_sha256': digest(Path(__file__))})
    score(out); secondary(out)
    checks = _audit(out)
    audit = {'passed': True, 'receipts_checked': len(statuses), 'group_metrics_checked': checks, 'scope': 'full',
             'jax_excluded': False, 'prediction_sha256': digest(out/'predictions.parquet'),
             'mask_sha256': digest(out/'site_masks.parquet')}
    atomic_json(out/'verification.json', audit)
    atomic_json(out/'completion.json', {'complete': True, 'failed_jobs': [r for r in statuses if r['status'] != 'complete'],
        'process_status_counts': dict(Counter(r['method']+':'+r['status'] for r in statuses)),
        'note': 'Six-method full report. Failed/unreported sites remain excluded; JAX-Ka nonconverged states are failed sites.'})
    shutil.copyfile(Path(__file__), out/'production_full_v2.py')
    print(json.dumps({'report': str(out), 'verification': audit,
                      'coverage': json.loads((out/'coverage_gate.json').read_text())}, indent=2), flush=True)
