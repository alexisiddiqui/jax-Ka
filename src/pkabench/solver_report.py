"""Assemble and verify a diagnostic report from isolated solver receipts."""
import json
import os
import shutil
import subprocess
from pathlib import Path
from collections import Counter
from .runtime import require_compute,atomic_json,digest,config_hash
from .schema import read_table,write_table,key


def collect(campaign):
    require_compute()
    import numpy as np
    from .isolated_solver import root
    from .frozen_smoke import verify
    from .frozen_score import score
    from .frozen_score_secondary import run as secondary
    campaign=Path(campaign).resolve(); work=root(campaign)
    contract=json.loads((work/'contract.json').read_text())
    assert digest(campaign/'manifest.json')==contract['manifest_sha256']
    assert digest(Path(__file__).with_name('isolated_solver.py'))==contract['implementation_sha256']
    for name,sha in contract['jax_sources'].items():
        assert digest(Path(__file__).parent.parent/'jaxpropka'/name)==sha
    missing=[cid for cid in contract['candidates'] if not (work/cid/'validation.json').exists()]
    atomic_json(work/'collection_status.json',{'expected':50,'validated':50-len(missing),'missing':missing})
    if missing: raise RuntimeError(f'{len(missing)} isolated validations incomplete; no partial candidate score generated')
    teacher=campaign.parent/'frozen-smoke-teacher-complete-v1'
    assert json.loads((teacher/'verification.json').read_text())['passed']
    # Preserve original timeout history when refreshing runtime accounting.
    if (campaign/'timeout-history').exists() and not (teacher/'timeout-history').exists():
        shutil.copytree(campaign/'timeout-history',teacher/'timeout-history')
    secondary(teacher)
    atomic_json(teacher/'retry_accounting.json',{
        'jax_initial_timeouts':'Preserved from original smoke timeout-history.',
        'pypka_initial_timeouts':'Initial failed receipts retained in frozen-smoke-numerical-v1/pypka/*/retry_policy.json; runtime CSV reports final teacher attempt only.',
        'teacher_retries':4})
    out=campaign.parent/'frozen-smoke-jax1024-v1'; out.mkdir(exist_ok=False)
    manifest=json.loads((teacher/'manifest.json').read_text())
    manifest.update(version='frozen-smoke-jax1024-v1',jax_solver=contract['candidate'],
                    solver_validation_contract_sha256=digest(work/'contract.json'),production_allowed=False)
    atomic_json(out/'manifest.json',manifest)
    for name in ('structures.parquet','sites.parquet','site_masks.parquet','assignments.parquet'):
        shutil.copyfile(teacher/name,out/name)
    (out/'structures').symlink_to(campaign/'structures',target_is_directory=True)
    records=[]; allrows=[]; jobs=[]; comparisons=[]; source_receipts=[]
    for method in manifest['methods']:
        dest=out/'jobs'/method; dest.mkdir(parents=True)
        for cid in contract['candidates']:
            if method!='jaxka':
                sidepath=teacher/'jobs'/method/f'{cid}.json'; side=json.loads(sidepath.read_text())
                assert side['status']=='complete' and not side['errors']
                assert digest(sidepath.with_suffix('.parquet'))==side['output_sha256']
                for ext in ('.json','.parquet'): shutil.copyfile(sidepath.with_suffix(ext),dest/f'{cid}{ext}')
                rows=read_table(sidepath.with_suffix('.parquet'))
                source_receipts.append({'method':method,'complex_id':cid,'source':str(sidepath),'sha256':digest(sidepath)})
            else:
                rows=[]; state_seconds={}; extras={}; receipts=[]
                validation=json.loads((work/cid/'validation.json').read_text()); assert validation['baseline_equivalent']
                for state in ('AB','A','B'):
                    path=work/cid/f'{state}-1024'; receipt=json.loads((path/'receipt.json').read_text())
                    assert receipt['config']['solver']==contract['candidate']
                    assert receipt['input_sha256']==digest(campaign/'structures'/cid/f'{state}.cif')
                    assert receipt['predictions_sha256']==digest(path/'predictions.parquet')
                    rr=read_table(path/'predictions.parquet')
                    assert all(r['config_sha256']==config_hash(receipt['config']) for r in rr)
                    rows.extend(rr); state_seconds[state]=receipt['wall_seconds']; extras[state]=receipt
                    receipts.append({'path':str(path/'receipt.json'),'sha256':digest(path/'receipt.json')})
                    old=json.loads((work/cid/f'{state}-64/receipt.json').read_text())
                    with np.load(work/cid/f'{state}-64/curves.npz') as a, np.load(path/'curves.npz') as b:
                        common=a['active']&a['valid']&b['valid']
                        comparisons.append({'complex_id':cid,'state':state,'baseline_converged':old['grid_converged'],
                            'candidate_converged':receipt['grid_converged'],
                            'baseline_valid':int((a['active']&a['valid']).sum()),'candidate_valid':int((b['active']&b['valid']).sum()),
                            'newly_invalid':int((a['active']&a['valid']&~b['valid']).sum()),
                            'max_midpoint_change_common_valid':float(np.max(np.abs(a['midpoint'][common]-b['midpoint'][common]))) if common.any() else None})
                    records.append(receipt)
                write_table(dest/f'{cid}.parquet','predictions',rows)
                side={'status':'complete','errors':{},'state_seconds':state_seconds,'extra':extras,
                      'wall_seconds':sum(state_seconds.values()),'job':records[-1]['job'],'node':'see isolated Slurm receipts',
                      'output_sha256':digest(dest/f'{cid}.parquet'),'source_receipts':receipts,
                      'contract_sha256':digest(work/'contract.json'),
                      'note':'Imported isolated state predictions. This receipt is not a legacy jobs.run_job cache entry.'}
                atomic_json(dest/f'{cid}.json',side)
            assert all(r['method']==method and r['complex_id']==cid for r in rows)
            sites=read_table(campaign/'structures'/cid/'sites.parquet')
            expected={(key(s),state) for s in sites for state in ('AB',s['partner'])}
            assert {(key(r),r['state']) for r in rows}==expected and len(rows)==len(expected)
            allrows.extend(rows); jobs.append({'method':method,'complex_id':cid,'status':'complete','errors':{}})
    write_table(out/'predictions.parquet','predictions',allrows)
    atomic_json(out/'merge_report.json',{'missing':[],'jobs':jobs,'assembly':'Receipt-verified isolated JAX outputs plus unchanged completed teacher/control shards; not legacy cache merge.'})
    snapshot=out/'scoring_sources'; snapshot.mkdir()
    for name in ('frozen_score.py','frozen_score_secondary.py','frozen_smoke_plots.py','score.py','linkage.py','frozen_smoke.py','solver_report.py','isolated_solver.py'):
        shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    atomic_json(snapshot/'hashes.json',{p.name:digest(p) for p in snapshot.glob('*.py')})
    atomic_json(out/'derivation.json',{'teacher_report':str(teacher),'solver_contract':str(work/'contract.json'),
        'unchanged_method_receipts':source_receipts,'diagnostic_only':True,'heldout_use':'Validation of the training-selected 1024-iteration candidate; no teacher-label-driven selection.'})
    score(out); secondary(out)
    subprocess.run([str(Path(os.environ['PKABENCH_RUNTIME'])/'envs/radial-plots/bin/python'),'-m','pkabench.frozen_smoke_plots',str(out)],check=True)
    summary={'states':len(records),'all_baselines_equivalent':True,
        'baseline_nonconverged_states':sum(not r['baseline_converged'] for r in comparisons),
        'candidate_nonconverged_states':sum(not r['candidate_converged'] for r in comparisons),
        'newly_invalid_midpoints':sum(r['newly_invalid'] for r in comparisons),
        'candidate_status_counts':dict(Counter(r['status'] for r in allrows if r['method']=='jaxka')),
        'comparisons':comparisons,'production_allowed':False}
    atomic_json(out/'solver_validation.json',summary)
    atomic_json(out/'completion.json',{'complete':True,'failed_jobs':[],'production_allowed':False,
        'note':'Diagnostic 1024-iteration report; nonconverged states and non-monotonic sites remain invalid. Production adoption requires review.'})
    verify(out)
    print(json.dumps({k:v for k,v in summary.items() if k!='comparisons'},indent=2),flush=True)
