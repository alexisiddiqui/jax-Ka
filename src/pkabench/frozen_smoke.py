"""Versioned reproducible-preparation gate and frozen-dataset engineering smoke."""
import json
import os
import shutil
import subprocess
from pathlib import Path
from collections import Counter
from .runtime import require_compute, atomic_json, digest, config_hash
from .schema import read_table, write_table

METHODS=['propka','jaxka','pkai','pkai_plus','null','pypka']


def initialise(freeze,out):
    require_compute()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .adapters.base import TEACHER
    freeze=Path(freeze).resolve(); out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    fm=json.loads((freeze/'manifest.json').read_text()); verify=json.loads((freeze.parent/(freeze.name+'-verification.json')).read_text())
    assert verify['passed'] and verify['manifest_sha256']==digest(freeze/'manifest.json')
    for name,sha in fm['artifacts_sha256'].items(): assert digest(freeze/name)==sha
    assignments={r['complex_id']:r for r in read_table(freeze/'assignments.parquet')}; inputs={r['complex_id']:r for r in json.loads((freeze/'input-files.json').read_text())}
    candidates=[]
    for s in read_table(freeze/'structures.parquet'):
        a=assignments[s['complex_id']]; role='train' if a['split']=='train' else 'eval'
        if not a['benchmark_eligible'] or a[role+'_interface_sites']==0 or s['n_residues']>1000: continue
        removals=json.loads((Path(inputs[s['complex_id']]['structure_root'])/'removed_components.json').read_text())['components']
        tags={r['policy_class'] for r in removals}; tags.add('antibody' if a['role']=='antibody_antigen' else 'general')
        candidates.append((s,tags))
    candidates.sort(key=lambda x:(x[0]['n_residues'],x[0]['complex_id']))
    chosen=[]; ids=set(); groups=set()
    # Size-bounded engineering sample; ensure preparation/role coverage, then
    # round-robin frozen splits with distinct groups before allowing repeats.
    def add(s,tags):
        chosen.append((s,tags)); ids.add(s['complex_id']); groups.add(s['component_id'])
    for tag,target in [('glycan',8),('buffer',10),('antibody',12)]:
        for s,tags in candidates:
            if sum(tag in t for _,t in chosen)>=target: break
            if tag in tags and s['complex_id'] not in ids and s['component_id'] not in groups: add(s,tags)
    for distinct in (True,False):
        while len(chosen)<50:
            progress=False
            for split in ('train','val','test'):
                options=[(s,t) for s,t in candidates if s['split']==split and s['complex_id'] not in ids and (not distinct or s['component_id'] not in groups)]
                if options and len(chosen)<50: add(*options[0]); progress=True
            if not progress: break
    assert len(chosen)==50
    (out/'structures').mkdir(); sites=[]
    for s,tags in chosen:
        cid=s['complex_id']; folder=Path(inputs[cid]['structure_root'])
        for state,sha in inputs[cid]['state_sha256'].items(): assert digest(folder/f'{state}.cif')==sha
        for name,sha in inputs[cid]['files_sha256'].items(): assert digest(folder/name)==sha
        (out/'structures'/cid).symlink_to(folder,target_is_directory=True); sites.extend(read_table(folder/'sites.parquet'))
    write_table(out/'structures.parquet','structures',[s for s,_ in chosen]); write_table(out/'sites.parquet','sites',sites)
    pq.write_table(pa.Table.from_pylist([assignments[c] for c in sorted(ids)]),out/'assignments.parquet')
    masks=[r for r in read_table(freeze/'site_masks.parquet') if r['complex_id'] in ids]
    pq.write_table(pa.Table.from_pylist(masks),out/'site_masks.parquet')
    manifest={'version':'frozen-smoke-v1','freeze':str(freeze),'freeze_manifest_sha256':digest(freeze/'manifest.json'),'methods':METHODS,'teacher':TEACHER,'teacher_config_sha256':config_hash(TEACHER),
        'candidates':[{'complex_id':s['complex_id'],'n_residues':s['n_residues'],'split':s['split'],'tags':sorted(tags)} for s,tags in chosen],
        'site_masks_sha256':digest(out/'site_masks.parquet'),'assignments_sha256':digest(out/'assignments.parquet'),'scorer_sha256':digest(Path(__file__).with_name('frozen_score.py')),
        'selection':'50 small-first complexes <=1000 residues, targeted glycan/buffer/antibody coverage, then split round robin, distinct sequence groups preferred. Engineering smoke, not a representative scientific sample.',
        'historical_pKPDB_equivalence':'unresolved; explicitly not the gate for current generated teacher labels','production_allowed':False}
    atomic_json(out/'manifest.json',manifest)
    atomic_json(out/'scoring_contract.json',{'primary':'Per-complex metrics averaged within sequence group, then equal group means. Skill per complex = 1 - MSE(delta)/mean(reference_delta^2); zero-denominator cases undefined. Spearman and sign accuracy averaged over defined complex/group metrics.',
        'bootstrap':'2000 group resamples for primary interface/shell scores; 400 for strata; require >=5 valid groups for percentile 95% CI. Seed 20261004.',
        'support':'Pairwise method/teacher support and all-method common support; coverage against frozen eligible sites. No missing-prediction imputation.',
        'strata':'Frozen split, interface and 0-20 A shells, residue, shift magnitude, dSASA and antibody/general role.',
        'linkage':'Complete charge coverage only; current masks, never partial masked sums. Native and HH curves labelled.',
        'inference':'Engineering smoke diagnostic only; PB agreement is not independent experimental accuracy.'})
    print(json.dumps({'selected':50,'residues':[min(s['n_residues'] for s,_ in chosen),max(s['n_residues'] for s,_ in chosen)],'splits':dict(Counter(s['split'] for s,_ in chosen)),'tags':dict(Counter(t for _,tags in chosen for t in tags))}),flush=True)


def preparation_gate(campaign):
    require_compute()
    from .adapters.base import TEACHER
    from .jobs import inputs,run_job
    from .preflight import preflight
    campaign=Path(campaign); manifest=json.loads((campaign/'manifest.json').read_text()); freeze=Path(manifest['freeze'])
    assert digest(freeze/'manifest.json')==manifest['freeze_manifest_sha256']
    assert digest(campaign/'site_masks.parquet')==manifest['site_masks_sha256']
    assert digest(campaign/'assignments.parquet')==manifest['assignments_sha256']
    assert config_hash(TEACHER)==manifest['teacher_config_sha256']
    target=min(read_table(campaign/'structures.parquet'),key=lambda s:s['n_residues']); cid=target['complex_id']
    run_job(campaign,cid,'pypka',1800)
    side=json.loads((campaign/'jobs/pypka'/f'{cid}.json').read_text()); rows=read_table(campaign/'jobs/pypka'/f'{cid}.parquet')
    native=all(any(r['state']==state and r['curve_source']=='native' and r['curve'] is not None for r in rows) for state in ('AB','A','B'))
    g1=side['status']=='complete' and native
    if not g1:
        atomic_json(campaign/'release_gates.json',{'G1':{'status':'fail','evidence':str(campaign/'jobs/pypka'/f'{cid}.json')},'smoke_allowed':False,'production_allowed':False})
        raise ValueError('Teacher G1 failed; no fallback teacher')
    assert side['inputs']==inputs(campaign,cid,'pypka')
    preflight(campaign,['propka','jaxka','pkai','pkai_plus','kaml'])
    controls=json.loads((campaign/'preflight.json').read_text())
    allowed=all(controls['G3'][m]['status']=='pass' for m in METHODS if m not in ('pypka','null'))
    gate={'version':'current-preparation-v1','G1':{'status':'pass','complex_id':cid,'evidence':str(campaign/'jobs/pypka'/f'{cid}.json')},
        'G1b':{'status':'pass' if len(side['extra'])==3 and all(x.get('intermediates') and x.get('intrinsic_tautomers_available') for x in side['extra'].values()) else 'partial'},
        'G2_current':{'status':'pass','teacher':TEACHER,'teacher_config_sha256':config_hash(TEACHER),'freeze_manifest_sha256':manifest['freeze_manifest_sha256'],
            'checks':'Frozen prepared structure hashes, split-aware masks, exact current teacher config, environment/implementation/weights hashes and completed native teacher state curves. All job receipts carry input/config hashes.'},
        'G2_historical':{'status':'unresolved','blocking_current_labels':False,'reason':'Exact historical pKPDB preparation/version provenance is not established. Current labels explicitly use our recorded configuration; no claim of historical reproduction.'},
        **controls,'smoke_allowed':allowed,'production_allowed':False,'scope':'Current frozen <=50-pair engineering smoke only; no 50% acceptance gate applied to rejected universe. Frozen accepted structures already passed chemical preparation.'}
    atomic_json(campaign/'release_gates.json',gate); print(json.dumps(gate,indent=2),flush=True)
    if not allowed: raise ValueError('Fresh joint-system method gate did not pass')


def dispatch(campaign):
    require_compute()
    import fcntl
    campaign=Path(campaign).resolve(); manifest=json.loads((campaign/'manifest.json').read_text()); gate=json.loads((campaign/'release_gates.json').read_text())
    assert len(manifest['candidates'])==50 and gate['smoke_allowed'] and gate['G2_current']['status']=='pass'
    assert digest(campaign/'site_masks.parquet')==manifest['site_masks_sha256']
    runtime=Path(os.environ['PKABENCH_RUNTIME']); ledger_path=campaign/'submission.json'
    with (runtime/'submission.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX); ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        for c in manifest['candidates']:
            cid=c['complex_id']
            if all(f'{cid}/{m}' in ledger for m in METHODS): continue
            used=sum(int(x) for x in subprocess.check_output(['squeue','-r','-h','-u',os.environ['USER'],'-o','%C'],text=True).split())
            if used+2>400: break
            status=subprocess.check_output(['sinfo','-N','-h','-p','generalaccess,amd96','-o','%N %t'],text=True)
            excluded={'comp1400'}|{r.split()[0] for r in status.splitlines() if r.split()[1] not in ('idle','mix','alloc')}
            command=['sbatch','--parsable','--exclude='+','.join(sorted(excluded)),f'--output={runtime}/logs/%x-%j.out','/home/coulson/oc/lina4225/_HPC/submission/jax-Ka/pkabench/work.sbatch','frozen-smoke','run','--campaign',str(campaign),'--complex-id',cid]
            job=subprocess.check_output(command,text=True).strip()
            for m in METHODS: ledger[f'{cid}/{m}']={'job':job,'requested_cpus':2,'requested_memory_gib':4,'queued_cores_before':used}
            atomic_json(ledger_path,ledger)
    remaining=[c['complex_id'] for c in manifest['candidates'] if f"{c['complex_id']}/pypka" not in ledger]
    atomic_json(campaign/'unsent.json',remaining)
    print(json.dumps({'submitted_complexes':len(ledger)//len(METHODS),'unsent':remaining,'jobs':sorted({v['job'] for v in ledger.values()})}),flush=True)


def run(campaign,cid):
    require_compute()
    from .jobs import run_job
    campaign=Path(campaign); manifest=json.loads((campaign/'manifest.json').read_text())
    assert cid in {r['complex_id'] for r in manifest['candidates']}
    assert json.loads((campaign/'release_gates.json').read_text())['smoke_allowed']
    for method in METHODS: run_job(campaign,cid,method,1800 if method=='pypka' else 600)


def collect(campaign):
    require_compute()
    from .jobs import merge
    from .frozen_score import score
    campaign=Path(campaign); manifest=json.loads((campaign/'manifest.json').read_text())
    assert digest(Path(__file__).with_name('frozen_score.py'))==manifest['scorer_sha256']
    snapshot=campaign/'scoring_sources'; snapshot.mkdir(exist_ok=True)
    for name in ('frozen_score.py','frozen_score_secondary.py','frozen_smoke_plots.py','score.py','linkage.py','frozen_smoke.py'):
        shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    atomic_json(snapshot/'hashes.json',{p.name:digest(p) for p in snapshot.glob('*.py')})
    merge(campaign,METHODS); score(campaign)
    from .frozen_score_secondary import run as secondary
    secondary(campaign)
    subprocess.run([str(Path(os.environ["PKABENCH_RUNTIME"])/"envs/radial-plots/bin/python"),"-m","pkabench.frozen_smoke_plots",str(campaign)],check=True)
    jobs=json.loads((campaign/'merge_report.json').read_text())['jobs']; scoring=json.loads((campaign/'scoring_report.json').read_text())
    atomic_json(campaign/'completion.json',{'complete':True,'job_statuses':dict(Counter(f"{r['method']}:{r['status']}" for r in jobs)),'failed_jobs':[r for r in jobs if r['status']!='complete'],'scoring':scoring,'production_allowed':False,
        'initial_jax_timeouts':len(list((campaign/'timeout-history').glob('*/initial.json'))),'retry_policy':'One 1800-second JAX retry after original 600-second budget, first attempts retained','note':'Completed engineering diagnostics; failures/coverage require review before production dispatch.'})
    print(json.dumps({'job_statuses':dict(Counter(f"{r['method']}:{r['status']}" for r in jobs)),'scoring':scoring},indent=2),flush=True)


def verify(campaign):
    require_compute()
    import math
    import csv
    from collections import defaultdict
    campaign=Path(campaign); manifest=json.loads((campaign/'manifest.json').read_text())
    complete=json.loads((campaign/'completion.json').read_text()); assert complete['complete']
    assert digest(campaign/'site_masks.parquet')==manifest['site_masks_sha256']
    assert digest(Path(manifest['freeze'])/'manifest.json')==manifest['freeze_manifest_sha256']
    jobs=json.loads((campaign/'merge_report.json').read_text()); assert not jobs['missing'] and len(jobs['jobs'])==300
    with (campaign/'scores_per_complex.csv').open() as f: rows=list(csv.DictReader(f))
    summary=json.loads((campaign/'scores_set1.json').read_text()); buckets=defaultdict(list)
    for r in rows: buckets[tuple(r[k] for k in ('method','split','scope','subset'))].append(r)
    checks=0
    for r in summary:
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
    mask={tuple(r[k] for k in ('complex_id','chain','resnum','icode','group')):r for r in read_table(campaign/'site_masks.parquet')}
    assert all(not (r['training_eligible'] and r['evaluation_eligible']) for r in mask.values())
    curves=json.loads((campaign/'linkage.json').read_text())
    for r in curves:
        if r['status']=='ok': assert all(s['training_eligible'] or s['evaluation_eligible'] for s in mask.values() if s['complex_id']==r['complex_id'])
    # Audit the effective defaults saved by PypKa, rather than assuming that
    # the small user-supplied configuration includes every solver setting.
    import ast
    from .gates import compare_deposit
    resolved=[]
    for sidepath in (campaign/'jobs/pypka').glob('*.json'):
        side=json.loads(sidepath.read_text())
        if side['status']!='complete': continue
        for state in ('AB','A','B'):
            path=Path(side['workdir'])/state/'resolved-config.txt'
            params=ast.literal_eval(path.read_text())
            expected={k:v for k,v in manifest['teacher'].items() if k!='temp'}
            assert compare_deposit(params,expected)['status']=='pass'
            mc=params['mc_params']; assert mc['seed']==1234567 and mc['mcsteps']==200000 and mc['eqsteps']==1000
            resolved.append({'complex_id':sidepath.stem,'state':state,'resolved_config_sha256':digest(path),'seed':mc['seed'],'mcsteps':mc['mcsteps'],'eqsteps':mc['eqsteps']})
    atomic_json(campaign/'resolved_teacher_audit.json',{'states_checked':len(resolved),'records':resolved,'temperature':'298.15 K pinned in worker request/config hash; PypKa get_clean_params intentionally omits temp from its saved resolved listing.'})
    coverage=json.loads((campaign/'coverage_gate.json').read_text())
    checks_result={'passed':True,'complex_method_shards_checked':300,'group_macro_metrics_recomputed':checks,'frozen_masks_and_split_unchanged':True,'teacher_coverage':coverage,
        'artifacts_sha256':{name:digest(campaign/name) for name in ('predictions.parquet','scores_set1.json','scores_set1.csv','scores_per_complex.csv','scores_per_group.csv','coverage.csv','representability.csv','linkage.json','scores_pooled_secondary.csv')},
        'production_allowed':False,'production_review':'Teacher >=80% gate is reported explicitly; method/site failures and scientific representativeness still require review.'}
    atomic_json(campaign/'verification.json',checks_result)
    print(json.dumps({k:v for k,v in checks_result.items() if k!='artifacts_sha256'},indent=2),flush=True)


def retry_jax(campaign,cid):
    require_compute()
    from .jobs import run_job
    campaign=Path(campaign); base=campaign/'jobs/jaxka'; sidepath=base/f'{cid}.json'
    side=json.loads(sidepath.read_text())
    assert side['status']=='failed' and side['errors'] and all(e['type']=='TimeoutExpired' for e in side['errors'].values())
    history=campaign/'timeout-history'/cid; history.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(sidepath,history/'initial.json'); shutil.copyfile(sidepath.with_suffix('.parquet'),history/'initial.parquet')
    atomic_json(history/'retry.json',{'method':'jaxka','complex_id':cid,'initial_budget_seconds':600,'retry_budget_seconds':1800,'job':os.environ['SLURM_JOB_ID'],'reason':'Uniform scientific inputs; initial AB/A/B total timeout too short, retain first-attempt runtime/failure evidence.'})
    run_job(campaign,cid,'jaxka',1800)


def diagnostics(campaign):
    require_compute()
    from .schema import key
    from collections import defaultdict
    import numpy as np
    import csv
    campaign=Path(campaign); masks={key(r):r for r in read_table(campaign/'site_masks.parquet')}
    rows=read_table(campaign/'predictions.parquet'); failures=defaultdict(list)
    for r in rows:
        if r['method']=='jaxka' and r['status']=='failed': failures[r['complex_id'],r['state']].append(r)
    from jaxpropka.parameters import ModelConfig
    tolerance=ModelConfig().residual_tolerance
    detail=[]
    for (cid,state),rr in failures.items():
        side=json.loads((campaign/'jobs/jaxka'/f'{cid}.json').read_text())
        eligible=[r for r in rr if masks[key(r)]['training_eligible'] or masks[key(r)]['evaluation_eligible']]
        residual=side['extra'].get(state,{}).get('max_residual')
        category='grid_nonconvergence' if residual is not None and residual>=tolerance else 'midpoint_validity_failed'
        detail.append({'complex_id':cid,'state':state,'failed_rows':len(rr),'eligible_failed_rows':len(eligible),'eligible_interface_failed_rows':sum(masks[key(r)]['interface'] for r in eligible),
            'worker_exception':side['errors'].get(state),'max_residual':side['extra'].get(state,{}).get('max_residual'),
            'category':category,'interpretation':'Saved maximum grid residual exceeds configured tolerance; adapter invalidates active-site rows for this state.' if category=='grid_nonconvergence' else 'Grid residual is below tolerance; failed rows instead reflect bracketed but invalid midpoint readouts. No numerical validity flag is relaxed.'})
    with (campaign/'runtime_by_residues.csv').open() as f: rt=list(csv.DictReader(f))
    performance={}
    for method in METHODS:
        rr=[r for r in rt if r['method']==method and r['status']=='complete']; seconds=[float(r['final_attempt_seconds']) for r in rr]
        performance[method]={'completed':len(rr),'median_seconds':float(np.median(seconds)) if seconds else None,'max_seconds':max(seconds,default=None)}
    with (campaign/'coverage.csv').open() as f: cov=list(csv.DictReader(f))
    interface=[r for r in cov if r['method']=='null' and r['subset']=='interface']
    result={'jax_failed_states':detail,'jax_affected_complexes':len({cid for cid,_ in failures}),'jax_convergence_complexes':len({r['complex_id'] for r in detail if r['category']=='grid_nonconvergence'}),'jax_failure_categories':dict(Counter({category:sum(r['failed_rows'] for r in detail if r['category']==category) for category in ('grid_nonconvergence','midpoint_validity_failed')})),
        'interface_coverage':{k:sum(int(r[k]) for r in interface) for k in ('eligible_sites','teacher_sites','all_method_common_sites')},
        'completed_runtime_summary':performance,'teacher_partial_timeouts':json.loads((campaign/'completion.json').read_text())['failed_jobs'],
        'limits':'Convergence flags must be investigated without relaxing them to admit invalid predictions. Large per-complex skill magnitudes can result from tiny reference-shift denominators; smoke scores are diagnostic. Runtime statistics exclude failed attempts unless explicitly included in the CSV.'}
    atomic_json(campaign/'diagnostics.json',result); print(json.dumps(result,indent=2),flush=True)
