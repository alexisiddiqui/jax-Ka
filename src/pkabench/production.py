"""Frozen-input production queue with a group-diverse 500-complex training pilot."""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter,defaultdict
from .runtime import require_compute,atomic_json,digest,config_hash
from .schema import read_table,write_table,KEY,key

METHODS=['pypka','propka','jaxka','pkai','pkai_plus','null']


def code_hashes():
    base=Path(__file__).parent
    names=['production.py','production_worker.py','adapters/base.py','adapters/worker.py','prep.py','schema.py','runtime.py','frozen_score.py','frozen_score_secondary.py']
    result={name:digest(base/name) for name in names}
    result.update({'jaxpropka/'+p.name:digest(p) for p in (base.parent/'jaxpropka').glob('*.py')})
    return result


def initialise(out):
    require_compute()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .adapters.base import TEACHER
    runtime=Path(os.environ['PKABENCH_RUNTIME']); freeze=runtime/'universe/structural-freeze-v1'
    smoke=runtime/'campaigns/frozen-smoke-jax1024-v1'
    assert json.loads((smoke/'verification.json').read_text())['passed']
    assert json.loads((smoke/'solver_validation.json').read_text())['candidate_nonconverged_states']==0
    fm=json.loads((freeze/'manifest.json').read_text())
    for name,sha in fm['artifacts_sha256'].items(): assert digest(freeze/name)==sha
    assignments={r['complex_id']:r for r in read_table(freeze/'assignments.parquet')}
    inputs={r['complex_id']:r for r in json.loads((freeze/'input-files.json').read_text())}
    structures=[]
    for s in read_table(freeze/'structures.parquet'):
        a=assignments[s['complex_id']]; field='train_interface_sites' if a['split']=='train' else 'eval_interface_sites'
        if a['benchmark_eligible'] and a[field]>0: structures.append(s)
    assert Counter(s['split'] for s in structures)=={'train':778,'val':151,'test':523}
    # Deterministic group round robin. No runtime, residue-count or label selection.
    groups=defaultdict(list)
    for s in structures:
        if s['split']=='train': groups[s['component_id']].append(s['complex_id'])
    for ids in groups.values(): ids.sort(key=lambda c:config_hash(['pilot-20261004',c]))
    order=sorted(groups,key=lambda g:config_hash(['pilot-20261004',g])); pilot=[]
    while len(pilot)<500:
        for g in order:
            if groups[g] and len(pilot)<500: pilot.append(groups[g].pop(0))
    out=Path(out); out.mkdir(exist_ok=False); (out/'structures').mkdir()
    sites=[]; selected={s['complex_id'] for s in structures}
    for s in structures:
        cid=s['complex_id']; inp=inputs[cid]; folder=Path(inp['structure_root'])
        for state,sha in inp['state_sha256'].items(): assert digest(folder/f'{state}.cif')==sha
        for name,sha in inp['files_sha256'].items(): assert digest(folder/name)==sha
        (out/'structures'/cid).symlink_to(folder.resolve(),target_is_directory=True)
        sites.extend(read_table(folder/'sites.parquet'))
    write_table(out/'structures.parquet','structures',structures); write_table(out/'sites.parquet','sites',sites)
    pq.write_table(pa.Table.from_pylist([assignments[c] for c in sorted(selected)]),out/'assignments.parquet')
    pq.write_table(pa.Table.from_pylist([r for r in read_table(freeze/'site_masks.parquet') if r['complex_id'] in selected]),out/'site_masks.parquet')
    atomic_json(out/'input-files.json',{c:inputs[c] for c in sorted(selected)})
    atomic_json(out/'pilot.json',{'complex_ids':pilot,'count':500,'split':'train','selection':'Seeded SHA-256 sequence-group round robin; no size, label or method-success selection. Pilot membership fixed before production.'})
    manifest={'version':'production-1024-v1','freeze':str(freeze),'freeze_manifest_sha256':digest(freeze/'manifest.json'),
        'methods':METHODS,'teacher':TEACHER,'site_masks_sha256':digest(out/'site_masks.parquet'),
        'assignments_sha256':digest(out/'assignments.parquet'),'pilot_sha256':digest(out/'pilot.json'),
        'code_sha256':code_hashes(),'scorer_sha256':digest(Path(__file__).with_name('frozen_score.py')),
        'structures':len(structures),'split_counts':dict(Counter(s['split'] for s in structures)),
        'jax_steps':1024,'source_smoke':str(smoke),'source_smoke_manifest_sha256':digest(smoke/'manifest.json'),
        'timeout_seconds':{'pypka':5400,'jaxka':5400,'propka':600,'pkai':600,'pkai_plus':600,'null':600},
        'production_authorization':'User authorized using available CPUs to proceed from validated smoke to production; 400-core cap and 2GB/core retained.',
        'environment_sha256':{p.name:digest(p) for p in (runtime/'manifests').glob('*.requirements.lock')},
        'weights_sha256':{str(p):digest(p) for p in (runtime/'envs/pkai/lib/python3.11/site-packages/pkai/models').glob('*.pt')},
        'native_manifest_sha256':digest(runtime/'manifests/fortran-explicit.txt')}
    atomic_json(out/'manifest.json',manifest)
    tasks=[]; pilotset=set(pilot)
    for s in structures:
        cid=s['complex_id']
        for method in METHODS:
            priority=(0 if method in ('pypka','propka') else 1) if cid in pilotset else (2 if method in ('pypka','propka') else 3)
            tasks.append({'complex_id':cid,'method':method,'priority':priority,'n_residues':s['n_residues'],'split':s['split']})
    tasks.sort(key=lambda t:(t['priority'],config_hash(['task-order',t['complex_id']]),METHODS.index(t['method'])))
    atomic_json(out/'queue.json',{'next':0,'tasks':tasks}); atomic_json(out/'claims.json',{})
    (out/'jobs').mkdir(); (out/'source').mkdir()
    for name in ('production.py','production_worker.py'): shutil.copyfile(Path(__file__).with_name(name),out/'source'/name)
    print(json.dumps({'complexes':len(structures),'pilot':500,'pilot_groups':len({assignments[c]['component_id'] for c in pilot}),'tasks':len(tasks),'splits':manifest['split_counts']}),flush=True)


def check(campaign):
    m=json.loads((campaign/'manifest.json').read_text())
    assert m['code_sha256']==code_hashes(),'production implementation changed'
    assert digest(campaign/'site_masks.parquet')==m['site_masks_sha256']
    runtime=Path(os.environ['PKABENCH_RUNTIME'])
    for name,sha in m['environment_sha256'].items(): assert digest(runtime/'manifests'/name)==sha
    for path,sha in m['weights_sha256'].items(): assert digest(Path(path))==sha
    assert digest(runtime/'manifests/fortran-explicit.txt')==m['native_manifest_sha256']
    return m


def failed_rows(campaign,cid,method,state=None):
    for s in read_table(campaign/'structures'/cid/'sites.parquet'):
        for st in ('AB',s['partner']):
            if state is not None and st!=state: continue
            yield {k:s[k] for k in KEY}|{'state':st,'method':method,'method_version':'unavailable','config_sha256':'',
                'status':'failed','pka':None,'curve':None,'curve_source':None,'intrinsic_pka':None}


def calculate(campaign,cid,method,work,timeout):
    from .adapters.base import Adapter,execute
    started=time.monotonic()
    if method!='jaxka':
        adapter=Adapter(method,read_table(campaign/'structures'/cid/'sites.parquet'),os.environ['PKABENCH_RUNTIME'],timeout)
        rows=adapter.run({s:campaign/'structures'/cid/f'{s}.cif' for s in ('AB','A','B')},work)
        return rows,adapter.errors,adapter.timings,adapter.extras,time.monotonic()-started
    rows=[]; errors={}; timings={}; extras={}
    for state in ('AB','A','B'):
        log=work/f'{state}-process'; log.mkdir(); dest=work/state
        try:
            remaining=timeout-(time.monotonic()-started)
            if remaining<=0: raise subprocess.TimeoutExpired(['jaxka'],timeout)
            timings[state]=execute([sys.executable,'-m','pkabench.production_worker',str(campaign),cid,state,str(dest)],log,remaining)
            receipt=json.loads((dest/'receipt.json').read_text())
            assert receipt['predictions_sha256']==digest(dest/'predictions.parquet')
            rows.extend(read_table(dest/'predictions.parquet')); extras[state]=receipt
        except Exception as exc:
            errors[state]={'type':type(exc).__name__,'detail':str(exc)}
            rows.extend(failed_rows(campaign,cid,method,state))
    return rows,errors,timings,extras,time.monotonic()-started


def gate(campaign):
    require_compute(); campaign=Path(campaign).resolve(); manifest=check(campaign)
    import numpy as np
    source=Path(manifest['source_smoke'])
    cid=min(read_table(source/'structures.parquet'),key=lambda s:s['n_residues'])['complex_id']
    work=campaign/'gate'; work.mkdir(exist_ok=False)
    rows,errors,timings,extras,elapsed=calculate(campaign,cid,'jaxka',work,1800)
    assert not errors,errors
    old={(key(r),r['state']):r for r in read_table(source/'jobs/jaxka'/f'{cid}.parquet')}
    for r in rows:
        o=old[key(r),r['state']]; assert r['status']==o['status']
        if r['pka'] is not None: assert abs(r['pka']-o['pka'])<=1e-4
        if r['curve'] is not None: assert np.allclose(r['curve'],o['curve'],atol=1e-6,rtol=0)
    atomic_json(campaign/'release_gate.json',{'passed':True,'complex_id':cid,'rows_verified':len(rows),
        'checks':'Versioned production worker matches validated 1024 smoke; unchanged frozen inputs, configurations and runtime locks.',
        'manifest_sha256':digest(campaign/'manifest.json'),'wall_seconds':elapsed,'job':os.environ['SLURM_JOB_ID']})
    print('Production wrapper gate passed',flush=True)


def reuse(campaign,cid,method,manifest):
    source=Path(manifest['source_smoke']); path=source/'jobs'/method/f'{cid}.json'
    if not path.exists(): return None
    side=json.loads(path.read_text()); assert side['status']=='complete' and not side['errors']
    assert digest(path.with_suffix('.parquet'))==side['output_sha256']
    for state in ('AB','A','B'):
        assert digest(source/'structures'/cid/f'{state}.cif')==digest(campaign/'structures'/cid/f'{state}.cif')
    rows=read_table(path.with_suffix('.parquet'))
    return rows,side,{'source_receipt':str(path),'source_receipt_sha256':digest(path)}


def pool(campaign):
    require_compute(); campaign=Path(campaign).resolve(); manifest=check(campaign)
    gate=json.loads((campaign/'release_gate.json').read_text()); assert gate['passed'] and gate['manifest_sha256']==digest(campaign/'manifest.json')
    pinned=json.loads((campaign/'input-files.json').read_text()); started=time.monotonic()
    while time.monotonic()-started<21*3600:
        with (campaign/'queue.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            q=json.loads((campaign/'queue.json').read_text()); i=q['next']
            if i==len(q['tasks']): return
            task=q['tasks'][i]; q['next']=i+1
            claims=json.loads((campaign/'claims.json').read_text()); claims[str(i)]={'job':os.environ['SLURM_JOB_ID'],'task':task}
            atomic_json(campaign/'claims.json',claims); atomic_json(campaign/'queue.json',q)
        cid=task['complex_id']; method=task['method']; base=campaign/'jobs'/method; base.mkdir(exist_ok=True)
        for state,sha in pinned[cid]['state_sha256'].items(): assert digest(campaign/'structures'/cid/f'{state}.cif')==sha
        work=base/cid/f"attempt-{os.environ['SLURM_JOB_ID']}"; work.mkdir(parents=True,exist_ok=False)
        reused=reuse(campaign,cid,method,manifest)
        if reused:
            rows,old,origin=reused; errors=old['errors']; timings=old['state_seconds']; extras=old.get('extra',{}); elapsed=old['wall_seconds']; workdir=old.get('workdir',str(work))
        else:
            rows,errors,timings,extras,elapsed=calculate(campaign,cid,method,work,manifest['timeout_seconds'][method]); origin=None; workdir=str(work)
        expected={(key(s),st) for s in read_table(campaign/'structures'/cid/'sites.parquet') for st in ('AB',s['partner'])}
        assert {(key(r),r['state']) for r in rows}==expected and len(rows)==len(expected)
        output=base/f'{cid}.parquet'; write_table(output,'predictions',rows)
        atomic_json(base/f'{cid}.json',{'status':'failed' if errors else 'complete','errors':errors,'state_seconds':timings,
            'extra':extras,'wall_seconds':elapsed,'workdir':workdir,'reused':origin,'node':os.environ['SLURMD_NODENAME'],
            'job':os.environ['SLURM_JOB_ID'],'output_sha256':digest(output),'manifest_sha256':digest(campaign/'manifest.json'),
            'task_index':i,'input_state_sha256':pinned[cid]['state_sha256']})
        print(json.dumps({'task':i,'complex_id':cid,'method':method,'status':'failed' if errors else 'complete','reused':bool(reused),'seconds':elapsed}),flush=True)
    raise RuntimeError('Worker lifetime reached; remaining queue requires another bounded pool wave')


def status(campaign):
    require_compute(); campaign=Path(campaign); q=json.loads((campaign/'queue.json').read_text()); counts=Counter(); done=0
    pilot=set(json.loads((campaign/'pilot.json').read_text())['complex_ids']); completed=defaultdict(set)
    for p in (campaign/'jobs').glob('*/*.json'):
        r=json.loads(p.read_text()); counts[p.parent.name+':'+r['status']]+=1; done+=1
        if r['status']=='complete': completed[p.parent.name].add(p.stem)
    report={'total_tasks':len(q['tasks']),'claimed':q['next'],'receipts':done,'counts':dict(counts),
        'pilot_teacher_and_propka_process_complete':len(pilot&completed['pypka']&completed['propka']),
        'note':'Process completion is not usable-label coverage; failed/unreported sites remain excluded.'}
    atomic_json(campaign/'status.json',report); print(json.dumps(report,indent=2),flush=True)


def collect(campaign):
    require_compute(); campaign=Path(campaign).resolve(); manifest=check(campaign)
    from .frozen_score import score
    from .frozen_score_secondary import run as secondary
    status(campaign)
    q=json.loads((campaign/'queue.json').read_text()); missing=[]; rows=[]; jobs=[]
    for t in q['tasks']:
        cid=t['complex_id']; method=t['method']; path=campaign/'jobs'/method/f'{cid}.json'
        if not path.exists(): missing.append(t); continue
        receipt=json.loads(path.read_text())
        assert receipt['manifest_sha256']==digest(campaign/'manifest.json')
        assert digest(path.with_suffix('.parquet'))==receipt['output_sha256']
        rr=read_table(path.with_suffix('.parquet'))
        assert all(r['complex_id']==cid and r['method']==method for r in rr)
        rows.extend(rr); jobs.append(t|{'status':receipt['status'],'errors':receipt['errors']})
    atomic_json(campaign/'merge_report.json',{'jobs':jobs,'missing':missing})
    if missing: raise RuntimeError(f'{len(missing)} tasks lack receipts; inspect Slurm and recover explicitly before full scoring')
    write_table(campaign/'predictions.parquet','predictions',rows)
    score(campaign); secondary(campaign)
    subprocess.run([str(Path(os.environ['PKABENCH_RUNTIME'])/'envs/radial-plots/bin/python'),'-m','pkabench.frozen_smoke_plots',str(campaign)],check=True)
    atomic_json(campaign/'completion.json',{'complete':True,'tasks':len(jobs),'failed_jobs':[r for r in jobs if r['status']!='complete'],
        'note':'Predictions and scores assembled; label coverage, runtime failures and scientific review remain explicit before training.'})
    print('Production collection completed',flush=True)
