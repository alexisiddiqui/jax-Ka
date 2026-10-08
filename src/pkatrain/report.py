"""Common-support comparisons using the existing benchmark aggregation."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from pkabench.runtime import atomic_json,digest
from pkabench.frozen_score import measures,aggregate,METRICS
from .records import read,KEY


def key(row): return tuple(row[k] for k in KEY)

def score(rows):
    complexes=defaultdict(list)
    for r in rows: complexes[r['complex_id']].append(r)
    items=[]
    for cid,rr in complexes.items():
        items.append(dict(complex_id=cid,component_id=rr[0]['component_id'],n=len(rr),
            **measures([r['target'] for r in rr],[r['prediction'] for r in rr])))
    return aggregate(items),items


def collect(out):
    m=read(out/'manifest.json'); methods={}; provenance={}; coverage=[]
    for name,folder in [('untrained_shared_solver',out/'baseline')]+[(f'trained-{s}',out/f'seed-{s}/epoch-20') for s in (17,29,43)]:
        gate=read(folder/'verification.json'); path=folder/'predictions.parquet'
        assert gate['passed'] and digest(path)==gate['predictions_sha256']
        records=pq.read_table(path).to_pylist(); methods[name]={key(r):r for r in records if r['valid']}
        coverage.append({'method':name,'eligible_sites':len(records),'valid_sites':len(methods[name]),
            'valid_complexes':len({r['complex_id'] for r in methods[name].values()}),
            'valid_groups':len({r['component_id'] for r in methods[name].values()})})
        provenance[str(path)]=digest(path)
    common=set.intersection(*(set(v) for v in methods.values()))
    assert common
    results=[]; per_complex={}
    for name,records in methods.items():
        (stats,groups),items=score([records[k] for k in sorted(common)])
        results.append(dict(method=name,**stats)); per_complex[name]={r['complex_id']:r for r in items}
    paired=[]
    for seed in (17,29,43):
        rows=[]
        for cid,r in per_complex[f'trained-{seed}'].items():
            base=per_complex['untrained_shared_solver'][cid]
            rows.append(dict(complex_id=cid,component_id=r['component_id'],n=r['n'],
                **{field:(r[field]-base[field] if r[field] is not None and base[field] is not None else None) for field in METRICS}))
        stats,_=aggregate(rows); paired.append(dict(seed=seed,mae_change=stats['mae'],ci95=stats['mae_ci95']))
    # Independent predictors and completed training baselines: read validation only.
    root=out.parents[1]; prod=root/'campaigns/production-nojax-v1'
    supplemental={}; assignments={cid:read(out/'records'/f'{cid}.json') for cid in m['val']}
    targets=methods['untrained_shared_solver']
    for method in ('pkai','propka','jaxka'):
        dest=root/'campaigns/production-1024-v2' if method=='jaxka' else prod
        mapping={}
        for cid in m['val']:
            path=dest/'jobs'/method/f'{cid}.parquet'; side=read(path.with_suffix('.json'))
            assert digest(path)==side['output_sha256']; provenance[str(path)]=digest(path)
            rows=pq.read_table(path).to_pylist(); aa={key(r):r for r in rows if r['state']=='AB' and r['status']=='ok'}
            ff={key(r):r for r in rows if r['state']!='AB' and r['status']=='ok'}
            for k in aa.keys()&ff.keys()&targets.keys():
                if aa[k]['pka'] is not None and ff[k]['pka'] is not None:
                    mapping[k]=dict(targets[k],prediction=aa[k]['pka']-ff[k]['pka'])
        supplemental[method]=mapping
    base=root/'finetune/diagnostic-v1'
    for arm in ('last','all','catboost'):
        for seed in (17,29,43):
            path=base/f'{arm}-{seed}/predictions.parquet'; receipt=read(path.with_name('receipt.json'))
            assert digest(path)==receipt['prediction_sha256']; provenance[str(path)]=digest(path)
            rows=pq.read_table(path,filters=[('split','=','val')]).to_pylist()
            mapping={}
            for r in rows:
                k=key(r)
                if k in targets:
                    assert abs(targets[k]['target']-r['target_delta_pka'])<1e-8
                    mapping[k]=dict(targets[k],prediction=r['prediction'])
            supplemental[f'{arm}-{seed}']=mapping
    hybrid=root/'tierB/hybrid-mc-v1/full/paired_errors.parquet'
    if hybrid.exists():
        provenance[str(hybrid)]=digest(hybrid)
        for r in pq.read_table(hybrid).to_pylist():
            k=key(r)
            if k in targets and r['prediction'] is not None and np.isfinite(r['prediction']):
                assert abs(targets[k]['target']-r['target'])<1e-8
                supplemental.setdefault('hybrid-'+r['method'],{})[k]=dict(targets[k],prediction=r['prediction'])
    all_methods=methods|supplemental
    all_common=set.intersection(*(set(v) for v in all_methods.values()))
    additional=[]
    for name,records in all_methods.items():
        stats,_=score([records[k] for k in sorted(all_common)])[0]
        additional.append(dict(method=name,**stats))
    dest=out/'report'; dest.mkdir(exist_ok=True)
    for name,rows in [('primary',results),('all_methods',additional),('coverage',coverage)]:
        with (dest/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Shared JAX-Ka parameter-training pilot','',
        'Final-epoch checkpoints; no validation checkpoint selection. Three seeds vary sampling order, not initialization.',
        'Agreement is measured against current PypKa labels, not experimental truth.', '',
        f'Primary common support: {len(common)} interface sites.', '', '| Method | Shift MAE | 95% group CI |','|---|---:|---|']
    for r in results: lines.append(f"| {r['method']} | {r['mae']:.4f} | {r['mae_ci95']} |")
    lines+=['','| Method | Eligible sites | Valid sites | Complexes | Groups |','|---|---:|---:|---:|---:|']
    for r in coverage: lines.append(f"| {r['method']} | {r['eligible_sites']} | {r['valid_sites']} | {r['valid_complexes']} | {r['valid_groups']} |")
    lines+=['','| Seed | Paired MAE change vs untrained | 95% group CI |','|---|---:|---|']
    for r in paired: lines.append(f"| {r['seed']} | {r['mae_change']:.4f} | {r['ci95']} |")
    lines+=['',f'All-method intersection: {len(all_common)} sites. Fine-tuned pKAI checkpoints were selected on validation; hybrid models retain PypKa interactions.', '', '| Method | Shift MAE | 95% group CI |','|---|---:|---|']
    for r in sorted(additional,key=lambda r:r['mae']): lines.append(f"| {r['method']} | {r['mae']:.4f} | {r['mae_ci95']} |")
    lines+=['','Training and validation coverage, curves, parameter histories and branch diagnostics remain in the per-epoch and seed artifacts. A coverage change must accompany every performance claim.',
        'Hydrogen-bond scale includes local H bonds, carboxylate reorganization and protonation-state-dependent pair terms.']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(dest/'verification.json',{'passed':True,'manifest_sha256':digest(out/'manifest.json'),'inputs_sha256':provenance,
        'primary_sites':len(common),'all_method_sites':len(all_common),'paired_changes':paired,'test_data_included':False})
