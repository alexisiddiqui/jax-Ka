"""State-specific pilot teacher retries; completed states are never recomputed."""
import json
import os
from pathlib import Path
from .runtime import require_compute,atomic_json,digest
from .schema import read_table,write_table,key


def initialise(out):
    require_compute(); out=Path(out); out.mkdir(exist_ok=False)
    runtime=Path(os.environ['PKABENCH_RUNTIME']); source=runtime/'campaigns/production-1024-v1'
    pilot=json.loads((source/'pilot.json').read_text())['complex_ids']; tasks=[]; other=[]
    for cid in pilot:
        p=source/'jobs/pypka'/f'{cid}.json'; side=json.loads(p.read_text())
        for state,error in side['errors'].items():
            r={'complex_id':cid,'state':state,'original_error':error,'source_receipt':str(p),'source_receipt_sha256':digest(p)}
            if error['type']=='TimeoutExpired': tasks.append(r)
            else: other.append(r)
    atomic_json(out/'manifest.json',{'source':str(source),'source_manifest_sha256':digest(source/'manifest.json'),
        'pilot_sha256':digest(source/'pilot.json'),'tasks':tasks,'preparation_failures':other,
        'state_timeout_seconds':10800,'policy':'Only timed-out states rerun, with 3 hours per state. Reuse completed state predictions unchanged. Preparation failures remain separate.','code_sha256':digest(Path(__file__))})
    print(json.dumps({'timeout_states':len(tasks),'timeout_complexes':len({r['complex_id'] for r in tasks}),'other_failed_states':len(other)}),flush=True)


def run(out,index):
    require_compute(); out=Path(out).resolve(); m=json.loads((out/'manifest.json').read_text()); task=m['tasks'][index]
    assert digest(Path(__file__))==m['code_sha256']
    from .production import check
    from .adapters.base import Adapter
    source=Path(m['source']); check(source)
    assert digest(source/'manifest.json')==m['source_manifest_sha256']
    assert digest(Path(task['source_receipt']))==task['source_receipt_sha256']
    cid=task['complex_id']; state=task['state']; work=out/'states'/cid/state; work.mkdir(parents=True,exist_ok=False)
    adapter=Adapter('pypka',read_table(source/'structures'/cid/'sites.parquet'),os.environ['PKABENCH_RUNTIME'],m['state_timeout_seconds'])
    rows=adapter.run({state:source/'structures'/cid/f'{state}.cif'},work/'raw')
    write_table(work/'predictions.parquet','predictions',rows)
    atomic_json(work/'receipt.json',{'complex_id':cid,'state':state,'errors':adapter.errors,'timings':adapter.timings,'extra':adapter.extras,
        'status':'failed' if adapter.errors else 'complete','output_sha256':digest(work/'predictions.parquet'),
        'input_sha256':digest(source/'structures'/cid/f'{state}.cif'),'job':os.environ['SLURM_JOB_ID'],'manifest_sha256':digest(out/'manifest.json')})
    print(json.dumps({'complex_id':cid,'state':state,'errors':adapter.errors,'timings':adapter.timings}),flush=True)


def collect(out):
    require_compute(); out=Path(out); m=json.loads((out/'manifest.json').read_text()); source=Path(m['source'])
    records=[]; replacements={}
    for t in m['tasks']:
        base=out/'states'/t['complex_id']/t['state']; receipt=base/'receipt.json'
        if not receipt.exists(): records.append(t|{'status':'missing_receipt'}); continue
        r=json.loads(receipt.read_text()); assert digest(base/'predictions.parquet')==r['output_sha256']
        records.append(t|{'status':r['status'],'errors':r['errors']})
        if r['status']=='complete': replacements[t['complex_id'],t['state']]=read_table(base/'predictions.parquet')
    rows=[]
    for cid in json.loads((source/'pilot.json').read_text())['complex_ids']:
        original=read_table(source/'jobs/pypka'/f'{cid}.parquet')
        rows.extend(r for r in original if (cid,r['state']) not in replacements)
        for state in ('AB','A','B'): rows.extend(replacements.get((cid,state),[]))
    write_table(out/'teacher-pilot-v2.parquet','predictions',rows)
    atomic_json(out/'recovery_report.json',{'records':records,'replaced_states':len(replacements),
        'prediction_sha256':digest(out/'teacher-pilot-v2.parquet'),'preparation_failures':m['preparation_failures'],
        'note':'Versioned teacher overlay for the same 500 training complexes; original labels and completed states retained. Handoff v1 remains immutable.'})
    print(json.dumps({'replaced_states':len(replacements),'missing':sum(r['status']=='missing_receipt' for r in records),'failed':sum(r['status']=='failed' for r in records)}),flush=True)
