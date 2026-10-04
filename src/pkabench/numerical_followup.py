"""Isolated smoke follow-up; never overwrite the frozen smoke results."""
import json
import os
import shutil
import time
from pathlib import Path
from dataclasses import asdict
from .runtime import require_compute, atomic_json, digest


def run(campaign, cid, method):
    require_compute()
    campaign = Path(campaign).resolve()
    if method == 'review':
        return review(campaign)
    out = campaign.parent / 'frozen-smoke-numerical-v1' / method / cid
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'provenance.json', {
        'source_campaign': str(campaign), 'manifest_sha256': digest(campaign/'manifest.json'),
        'job': os.environ['SLURM_JOB_ID'], 'node': os.environ['SLURMD_NODENAME'],
        'source_sha256': digest(Path(__file__)), 'method': method,
        'policy': 'Diagnostic only; original predictions and numerical thresholds unchanged.'})
    if method == 'pypka':
        from .jobs import run_job
        side = json.loads((campaign/'jobs/pypka'/f'{cid}.json').read_text())
        assert side['status'] == 'failed' and side['errors']
        assert all(e['type'] == 'TimeoutExpired' for e in side['errors'].values())
        for name in ('manifest.json', 'structures.parquet', 'sites.parquet'):
            shutil.copyfile(campaign/name, out/name)
        (out/'structures').mkdir()
        (out/'structures'/cid).symlink_to((campaign/'structures'/cid).resolve(), target_is_directory=True)
        atomic_json(out/'retry_policy.json', {'original_budget_seconds':1800, 'budget_seconds':5400,
                    'original_receipt':side, 'science_changes':False})
        run_job(out,cid,'pypka',5400)
        return
    import numpy as np
    from jaxpropka import TitrationModel, prepare
    from jaxpropka.parameters import ModelConfig
    from .prep import read_cif
    from .schema import PH, GROUPS
    manifest = json.loads((campaign/'manifest.json').read_text())
    split = next(r['split'] for r in manifest['candidates'] if r['complex_id']==cid)
    # Iteration exploration is restricted to training structures. No teacher
    # labels are read and no configuration is promoted into benchmark results.
    steps = (64,256,1024) if split == 'train' else (64,)
    records=[]
    for state in ('AB','A','B'):
        cif=campaign/'structures'/cid/f'{state}.cif'
        cache=prepare(read_cif(cif), topology_options={'gap_policy':'cap','freeze_disulfides':True},
                      geometry_options={'missing_sidechain':'error'})
        for nsteps in steps:
            start=time.monotonic(); config=ModelConfig(steps=nsteps)
            model=TitrationModel(cache,config=config,backend='packed')
            curves=model.curves(PH)(model.native_probabilities)
            midpoint=model.pka_from_grid(PH)(model.native_probabilities)
            residual=np.asarray(curves.residual); active=np.asarray(curves.probability)>0
            valid=np.asarray(midpoint.valid); bracket=np.asarray(midpoint.bracketed)
            monotone=np.asarray(midpoint.sampled_monotone); slope=np.asarray(midpoint.slope)
            bad=[]
            for i,g in np.argwhere(active & bracket & ~valid):
                key=cache.keys[i]
                bad.append({'chain':key.chain,'resnum':key.number,'icode':key.insertion,'group':GROUPS[g],
                            'sampled_monotone':bool(monotone[i,g]),'slope':float(slope[i,g])})
            record={'state':state,'split':split,'config':asdict(config),'input_sha256':digest(cif),
                    'wall_seconds':time.monotonic()-start,'max_residual':float(residual.max()),
                    'failed_ph':[float(PH[i]) for i in np.flatnonzero(~np.asarray(curves.converged))],
                    'active_sites':int(active.sum()),'valid_midpoints':int((active&valid).sum()),
                    'invalid_bracketed_sites':bad}
            records.append(record)
            np.savez_compressed(out/f'{state}-{nsteps}.npz',ph=np.asarray(PH),residual=residual,
                                protonated=np.asarray(curves.protonated),valid=valid,
                                midpoint=np.asarray(midpoint.value),active=active)
            atomic_json(out/'results.json',records)
            print(json.dumps({k:v for k,v in record.items() if k not in ('invalid_bracketed_sites','config')}),flush=True)
            # Release compiled executables between configurations; retaining
            # all three models exceeded the 4 GB budget on one structure.
            del model, curves, midpoint
            import jax
            import gc
            jax.clear_caches()
            gc.collect()
    atomic_json(out/'complete.json',{'complete':True,'records':len(records),'split':split})


def review(campaign):
    import numpy as np
    from collections import Counter
    root=campaign.parent/'frozen-smoke-numerical-v1'
    folders={p.name:p for p in (root/'jaxka').iterdir() if p.is_dir()}
    retry=root/'jaxka-retry'
    if retry.exists():
        for p in retry.iterdir():
            if (p/'complete.json').exists(): folders[p.name]=p
    rows=[]
    for cid,folder in sorted(folders.items()):
        if not (folder/'results.json').exists(): continue
        records=json.loads((folder/'results.json').read_text())
        for rec in records:
            if rec['failed_ph']: continue
            path=folder/f"{rec['state']}-{rec['config']['steps']}.npz"
            with np.load(path) as data:
                y=data['protonated']; active=data['active']; valid=data['valid']
                bracket=(y[0]>=.5)&(y[-1]<=.5)&active
                for i,g in np.argwhere(bracket & ~valid):
                    curve=y[:,i,g]; increases=np.diff(curve)
                    crossings=int(np.count_nonzero((curve[:-1]-.5)*(curve[1:]-.5)<0))
                    rows.append({'complex_id':cid,'state':rec['state'],'split':rec['split'],
                        'steps':rec['config']['steps'],'site_index':[int(i),int(g)],
                        'max_upward_step':float(increases.max()),'crossings':crossings,
                        'nonmonotone':bool(increases.max()>1e-6)})
    teachers=[]
    for p in sorted((root/'pypka').glob('*/jobs/pypka/*.json')):
        r=json.loads(p.read_text())
        teachers.append({'complex_id':p.stem,'status':r['status'],'errors':r['errors'],
                         'wall_seconds':r['wall_seconds']})
    summary={'jax_complete':sum((p/'complete.json').exists() for p in folders.values()),
        'teacher_retries':teachers,'invalid_midpoints_on_converged_grids':rows,
        'counts_by_steps':dict(Counter(str(r['steps']) for r in rows)),
        'note':'Site-state counts repeat across iteration configurations; incomplete diagnostics remain partial. No validity rules or production settings changed.'}
    atomic_json(root/'review.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
