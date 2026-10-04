"""User-requested first-round reports excluding deferred JAX-Ka."""
import csv
import json
import math
import shutil
import time
from collections import Counter,defaultdict
from pathlib import Path
from .runtime import require_compute,atomic_json,digest
from .schema import read_table,write_table,key

METHODS=['pypka','propka','pkai','pkai_plus','null']


def run(campaign,scope):
    require_compute(); campaign=Path(campaign).resolve()
    from .production import check
    manifest=check(campaign)
    pilot=json.loads((campaign/'pilot.json').read_text())['complex_ids']
    ids=pilot if scope=='pilot' else [s['complex_id'] for s in read_table(campaign/'structures.parquet')]
    # The pilot can be delivered without waiting for the full production pool.
    deadline=time.monotonic()+90*60
    while True:
        missing=[(cid,m) for cid in ids for m in METHODS if not (campaign/'jobs'/m/f'{cid}.json').exists()]
        if not missing: break
        atomic_json(campaign/f'{scope}-nojax-waiting.json',{'expected':len(ids)*len(METHODS),'missing':len(missing),'missing_tasks':missing})
        if scope!='pilot' or time.monotonic()>deadline: raise RuntimeError(f'{len(missing)} non-JAX receipts missing; no incomplete full report published')
        time.sleep(30)
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .frozen_score import score
    from .frozen_score_secondary import run as secondary
    out=campaign.parent/('training-pilot-nojax-v1' if scope=='pilot' else 'production-nojax-v1')
    out.mkdir(exist_ok=False); selected=set(ids)
    for name in ('structures','sites'):
        write_table(out/f'{name}.parquet',name,[r for r in read_table(campaign/f'{name}.parquet') if r['complex_id'] in selected])
    for name in ('assignments','site_masks'):
        pq.write_table(pa.Table.from_pylist([r for r in read_table(campaign/f'{name}.parquet') if r['complex_id'] in selected]),out/f'{name}.parquet')
    if scope=='pilot':
        assert len(ids)==500 and all(s['split']=='train' for s in read_table(out/'assignments.parquet'))
        assert all(not r['evaluation_eligible'] for r in read_table(out/'site_masks.parquet'))
    (out/'structures').symlink_to(campaign/'structures',target_is_directory=True)
    derived=manifest|{'version':out.name,'methods':METHODS,'structures':len(ids),'parent_manifest_sha256':digest(campaign/'manifest.json'),
        'site_masks_sha256':digest(out/'site_masks.parquet'),'assignments_sha256':digest(out/'assignments.parquet'),
        'split_counts':dict(Counter(s['split'] for s in read_table(out/'assignments.parquet'))),
        'jaxka':'Deferred at user request; original successes and failures preserved in source campaign. Excluded from scoring and common support.',
        'scope':scope}
    atomic_json(out/'manifest.json',derived)
    rows=[]; statuses=[]; lineage=[]
    for method in METHODS:
        dest=out/'jobs'/method; dest.mkdir(parents=True)
        for cid in ids:
            path=campaign/'jobs'/method/f'{cid}.json'; receipt=json.loads(path.read_text())
            assert receipt['manifest_sha256']==derived['parent_manifest_sha256']
            assert digest(path.with_suffix('.parquet'))==receipt['output_sha256']
            rr=read_table(path.with_suffix('.parquet'))
            assert all(r['method']==method and r['complex_id']==cid for r in rr)
            expected={(key(s),st) for s in read_table(campaign/'structures'/cid/'sites.parquet') for st in ('AB',s['partner'])}
            assert len(rr)==len(expected) and {(key(r),r['state']) for r in rr}==expected
            for ext in ('.json','.parquet'): shutil.copyfile(path.with_suffix(ext),dest/f'{cid}{ext}')
            rows.extend(rr); statuses.append({'complex_id':cid,'method':method,'status':receipt['status'],'errors':receipt['errors']})
            lineage.append({'receipt':str(path),'sha256':digest(path)})
    write_table(out/'predictions.parquet','predictions',rows)
    atomic_json(out/'merge_report.json',{'missing':[],'jobs':statuses})
    atomic_json(out/'derivation.json',{'source':str(campaign),'source_receipts':lineage,'report_code_sha256':digest(Path(__file__))})
    score(out); secondary(out)
    # Independently recompute every reported group macro from per-complex CSV.
    with (out/'scores_per_complex.csv').open() as f: complexes=list(csv.DictReader(f))
    buckets=defaultdict(list)
    for r in complexes: buckets[tuple(r[k] for k in ('method','split','scope','subset'))].append(r)
    checks=0
    for r in json.loads((out/'scores_set1.json').read_text()):
        rr=buckets[tuple(r[k] for k in ('method','split','scope','subset'))]
        assert sum(int(x['n']) for x in rr)==r['sites']
        for metric in ('mae','rmse','skill','spearman','sign_accuracy','error_cancellation'):
            groups=defaultdict(list)
            for x in rr:
                if x[metric]!='': groups[x['component_id']].append(float(x[metric]))
            means=[sum(v)/len(v) for v in groups.values()]
            if means: assert math.isclose(sum(means)/len(means),r[metric],rel_tol=1e-10,abs_tol=1e-10)
            else: assert r[metric] is None
            checks+=1
    audit={'passed':True,'receipts_checked':len(statuses),'group_metrics_checked':checks,'scope':scope,'jax_excluded':True,
        'prediction_sha256':digest(out/'predictions.parquet'),'mask_sha256':digest(out/'site_masks.parquet')}
    atomic_json(out/'verification.json',audit)
    atomic_json(out/'completion.json',{'complete':True,'failed_jobs':[r for r in statuses if r['status']!='complete'],
        'process_status_counts':dict(Counter(r['method']+':'+r['status'] for r in statuses)),
        'note':'JAX excluded by scope. Failed/unreported teacher sites remain excluded; process completion does not imply usable training coverage.'})
    shutil.copyfile(Path(__file__),out/'production_nojax.py')
    print(json.dumps({'report':str(out),'verification':audit,'coverage':json.loads((out/'coverage_gate.json').read_text())},indent=2),flush=True)
