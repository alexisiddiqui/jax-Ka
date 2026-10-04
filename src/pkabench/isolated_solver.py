"""Versioned iteration validation with one process per state/configuration."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import asdict
from .runtime import require_compute, atomic_json, digest, config_hash


def root(campaign):
    return Path(campaign).parent/'frozen-smoke-isolated-v1'


def initialise(campaign):
    require_compute()
    from jaxpropka.parameters import ModelConfig
    campaign=Path(campaign); out=root(campaign); out.mkdir(exist_ok=False)
    source=json.loads((campaign/'manifest.json').read_text())
    atomic_json(out/'contract.json',{
        'source':str(campaign.resolve()),'manifest_sha256':digest(campaign/'manifest.json'),
        'candidates':[r['complex_id'] for r in source['candidates']],
        'baseline':asdict(ModelConfig()),'candidate':asdict(ModelConfig(steps=1024)),
        'selection':'1024 iterations chosen from training-only convergence diagnostics before held-out validation.',
        'implementation_sha256':digest(Path(__file__)),
        'jax_sources':{p.name:digest(p) for p in (Path(__file__).parent.parent/'jaxpropka').glob('*.py')},
        'baseline_curve_atol':1e-6,'baseline_pka_atol':1e-4,
        'rule':'Validate all 50 smoke pairs at 64 and 1024 iterations; unchanged residual, slope and monotonicity criteria. No teacher labels read for solver selection.',
        'production_allowed':False})
    shutil.copyfile(Path(__file__),out/'isolated_solver.py')
    print(str(out),flush=True)


def state_worker(campaign,cid,state,steps):
    require_compute()
    import numpy as np
    from jaxpropka import prepare,TitrationModel
    from jaxpropka.parameters import ModelConfig
    from jaxpropka.model import _grid_pka_result
    from .prep import read_cif
    from .schema import PH,GROUPS,KEY,read_table,write_table
    campaign=Path(campaign); out=root(campaign)/cid/f'{state}-{steps}'
    out.mkdir(parents=True,exist_ok=False)
    contract=json.loads((root(campaign)/'contract.json').read_text())
    assert digest(Path(__file__))==contract['implementation_sha256']
    config=ModelConfig(steps=steps); started=time.monotonic()
    cif=campaign/'structures'/cid/f'{state}.cif'
    cache=prepare(read_cif(cif),topology_options={'gap_policy':'cap','freeze_disulfides':True},geometry_options={'missing_sidechain':'error'})
    model=TitrationModel(cache,config=config,backend='packed')
    curves=model.curves(PH)(model.native_probabilities)
    # Apply the original readout to the already-computed grid. Avoid compiling
    # a second full-structure solve for midpoint extraction.
    mid=_grid_pka_result(model._d,curves,curves.ph,config)
    y=np.asarray(curves.protonated); active=np.asarray(curves.probability)>0
    valid=np.asarray(mid.valid); bracket=np.asarray(mid.bracketed); vals=np.asarray(mid.value)
    residual=np.asarray(curves.residual); converged=bool(np.asarray(curves.converged).all())
    monotone=np.asarray(mid.sampled_monotone); slopes=np.asarray(mid.slope)
    lookup={(k.chain,k.number,k.insertion):i for i,k in enumerate(cache.keys)}
    settings={'model':'jaxka','grid':[-2,16,.25],'gap_policy':'cap','solver':asdict(config),'readout':'existing _grid_pka_result applied to shared curves'}
    rows=[]
    for s in read_table(campaign/'structures'/cid/'sites.parquet'):
        if state!='AB' and s['partner']!=state: continue
        r={k:s[k] for k in KEY}; r.update(state=state,method='jaxka',method_version='isolated-v1',
            config_sha256=config_hash(settings),status='not_reported',pka=None,curve=None,curve_source=None,intrinsic_pka=None)
        i=lookup.get((s['chain'],s['resnum'],s['icode'])); g=GROUPS.index(s['group'])
        if i is not None:
            if not active[i,g]:
                if s['group']=='CYS' and bool(cache.frozen[i]):
                    r.update(status='not_titrating',curve=[0.]*73,curve_source='native')
            else:
                status='ok' if valid[i,g] else 'out_of_range' if not bracket[i,g] else 'failed'
                if not converged: status='failed'
                r.update(status=status,pka=float(vals[i,g]) if valid[i,g] else None,
                    curve=y[:,i,g].tolist() if status!='failed' else None,curve_source='native',
                    intrinsic_pka=float(curves.intrinsic_pka[i,g]))
        rows.append(r)
    write_table(out/'predictions.parquet','predictions',rows)
    np.savez_compressed(out/'curves.npz',protonated=y,active=active,residual=residual,midpoint=vals,valid=valid)
    atomic_json(out/'receipt.json',{'state':state,'steps':steps,'config':settings,'input_sha256':digest(cif),
        'predictions_sha256':digest(out/'predictions.parquet'),'job':os.environ['SLURM_JOB_ID'],
        'max_residual':float(residual.max()),'grid_converged':converged,'wall_seconds':time.monotonic()-started,
        'nonmonotone_bracketed':int((active&bracket&~monotone).sum()),
        'slope_failed_bracketed':int((active&bracket&(slopes>=-config.slope_min)).sum())})


def run(campaign,cid):
    require_compute()
    campaign=Path(campaign); contract=json.loads((root(campaign)/'contract.json').read_text())
    assert cid in contract['candidates']
    folder=root(campaign)/cid; folder.mkdir(exist_ok=False)
    for state in ('AB','A','B'):
        for steps in (64,1024):
            command=[sys.executable,'-m','pkabench.isolated_solver','state',str(campaign),cid,state,str(steps)]
            with (folder/f'{state}-{steps}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    validate_one(campaign,cid)


def validate_one(campaign,cid):
    import numpy as np
    from .schema import read_table,key
    folder=root(campaign)/cid; contract=json.loads((root(campaign)/'contract.json').read_text())
    old={(key(r),r['state']):r for r in read_table(Path(campaign)/'jobs/jaxka'/f'{cid}.parquet')}
    checks=[]; receipts=[]
    for state in ('AB','A','B'):
        baseline=read_table(folder/f'{state}-64/predictions.parquet')
        for r in baseline:
            o=old[(key(r),state)]
            assert r['status']==o['status'],(cid,state,key(r),r['status'],o['status'])
            if r['pka'] is not None:
                assert abs(r['pka']-o['pka'])<=contract['baseline_pka_atol']
            if r['curve'] is not None:
                assert np.allclose(r['curve'],o['curve'],rtol=0,atol=contract['baseline_curve_atol'])
        checks.append({'state':state,'baseline_rows_matched':len(baseline)})
        receipts.append(json.loads((folder/f'{state}-1024/receipt.json').read_text()))
    atomic_json(folder/'validation.json',{'baseline_equivalent':True,'checks':checks,'candidate_states':receipts})
    print(json.dumps({'complex_id':cid,'baseline_equivalent':True,'candidate_converged':[r['grid_converged'] for r in receipts]}),flush=True)


def teacher_report(campaign):
    require_compute()
    from .frozen_smoke import collect,verify
    from .jobs import inputs
    campaign=Path(campaign); out=campaign.parent/'frozen-smoke-teacher-complete-v1'
    out.mkdir(exist_ok=False)
    for name in ('manifest.json','structures.parquet','sites.parquet','site_masks.parquet','assignments.parquet','release_gates.json'):
        shutil.copyfile(campaign/name,out/name)
    (out/'structures').symlink_to(campaign/'structures',target_is_directory=True)
    source=json.loads((campaign/'manifest.json').read_text()); lineage=[]
    for method in source['methods']:
        dest=out/'jobs'/method; dest.mkdir(parents=True)
        for c in source['candidates']:
            cid=c['complex_id']; origin=campaign/'jobs'/method/f'{cid}.json'
            if method=='pypka':
                retry=campaign.parent/'frozen-smoke-numerical-v1/pypka'/cid/'jobs/pypka'/f'{cid}.json'
                if retry.exists(): origin=retry
            side=json.loads(origin.read_text())
            assert side['status']=='complete' and not side['errors']
            assert side['inputs']==inputs(out,cid,method)
            assert digest(origin.with_suffix('.parquet'))==side['output_sha256']
            for ext in ('.json','.parquet'): shutil.copyfile(origin.with_suffix(ext),dest/f'{cid}{ext}')
            lineage.append({'method':method,'complex_id':cid,'receipt':str(origin),'sha256':digest(origin)})
    atomic_json(out/'derivation.json',{'parent':str(campaign),'changes':'Four successful PypKa timeout retries replace partial teacher shards. JAX solver remains original 64 steps.','source_receipts':lineage})
    collect(out); verify(out)


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='state': state_worker(Path(sys.argv[2]),sys.argv[3],sys.argv[4],int(sys.argv[5]))
