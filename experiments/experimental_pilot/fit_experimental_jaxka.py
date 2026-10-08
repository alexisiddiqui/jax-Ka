"""First scalar-label experimental fit of JAX-Ka's three physical scales.

This is a training-pipeline shakedown, not an independent method benchmark.
Scalar pKa labels supervise the unsaturated midpoint residual
effective_pka(pH=label) - label through the shared implicit solver.
"""
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from pkabench.runtime import atomic_json, digest, require_compute

require_compute(threads=2)

import jax
jax.config.update('jax_enable_x64', False)
import jax.numpy as jnp
import numpy as np
import optax

from jaxpropka import TitrationModel
from jaxpropka.cache import StructureCache
from jaxpropka.geometry import build_candidates
from jaxpropka.model import _grid_pka_result
from jaxpropka.optx_solver import SolverConfig, active_channels, local_terms_curve_kernel
from jaxpropka.parameters import GROUPS, ModelConfig
from jaxpropka.precompute import build_cache, native_identities
from jaxpropka.topology import load_topology
from pkabench.prep import read_cif
from pkatrain.adapters.jaxka import local_terms, physical_scales

runtime = Path(os.environ['PKABENCH_RUNTIME'])
repo = Path(os.environ['PKABENCH_SOURCE'])
admission = runtime/'experimental/experimental-admission-v8'
full = runtime/'experimental/pkadr-full-v1'
dsba = runtime/'experimental/pkadr-dsba-reduced-v1'
trx = runtime/'experimental/pkadr-human-thioredoxin-recovery-v4'
out = runtime/'experimental/experimental-fit-v1'
config = ModelConfig(steps=1024)
solver_config = SolverConfig(max_steps=32)
grid = np.linspace(-2,16,73,dtype=np.float32)


def read(path):
    return json.loads(Path(path).read_text())


def structure_path(row):
    pdb = row['structure_pdb']
    if pdb == '1A2L':
        return dsba/'structure/prepared.cif'
    if pdb in ('1TRW','1TRS','4TRX'):
        return trx/pdb/'prepared.cif'
    return full/'structures'/row['structure_task_id']/'prepared.cif'


def initialize():
    if out.exists():
        if (out/'release.json').exists():
            raise FileExistsError(f'completed immutable release exists: {out}')
        shutil.rmtree(out)
    (out/'records').mkdir(parents=True)
    (out/'prepared').mkdir()
    ledger = list(csv.DictReader((admission/'admission_ledger.csv').open()))
    selected = [r for r in ledger if r['eligible_for_model_fit'] == 'True']
    assert len(selected) == 12
    by_structure = defaultdict(list)
    records = []
    for row in selected:
        path = structure_path(row)
        assert path.exists() and digest(path) == row['prepared_sha256']
        record = dict(
            record_id=row['record_id'], family_id=row['family_id'], protein_name=row['protein_name'],
            structure_pdb=row['structure_pdb'], structure_task_id=row['structure_task_id'],
            structure=str(path), structure_sha256=row['prepared_sha256'], chain=row['chain'],
            resnum=int(row['resnum']), icode=row['icode'], group=row['group'],
            target=float(row['curated_experimental_pka']),
            uncertainty=float(row['curated_experimental_uncertainty']) if row['curated_experimental_uncertainty'] else None,
            null_pka=float(row['null_pka']), natural_gap_tier=row['natural_gap_tier'])
        records.append(record)
        by_structure[(record['structure_task_id'],str(path))].append(record)

    family_counts = Counter(r['family_id'] for r in records)
    assert len(family_counts) == 5
    for record in records:
        record['fit_weight'] = 1/(len(family_counts)*family_counts[record['family_id']])

    structures = []
    for (task_id,path_text), rows in sorted(by_structure.items()):
        path = Path(path_text); dest = out/'prepared'/task_id; dest.mkdir()
        topology = load_topology(read_cif(path),gap_policy='cap',freeze_disulfides=True)
        candidates = build_candidates(topology,missing_sidechain='error')
        cache = build_cache(topology,candidates,identities=native_identities(topology),dtype=np.float32)
        cache.save(dest/'cache.npz')
        lookup = {(k.chain,k.number,k.insertion):i for i,k in enumerate(cache.keys)}
        sites = []
        for record in sorted(rows,key=lambda r:float(r['target'])):
            key = (record['chain'],record['resnum'],record['icode'])
            assert key in lookup, (record['record_id'],key)
            index = lookup[key]; group = GROUPS.index(record['group'])
            assert cache.group_mask[index,group]
            sites.append(dict(record_id=record['record_id'],index=index,group_index=group,
                              target=record['target'],weight=record['fit_weight']))
        receipt = dict(task_id=task_id,pdb=rows[0]['structure_pdb'],structure=str(path),
                       structure_sha256=digest(path),cache_sha256=digest(dest/'cache.npz'),
                       cache_fingerprint=cache.fingerprint(),n_residues=cache.n_residues,sites=sites)
        atomic_json(dest/'receipt.json',receipt); structures.append(receipt)

    manifest = dict(version='experimental-fit-v1',created='2026-10-06',objective='Huber(delta=1) on effective_pka(pH=experimental_pKa)-experimental_pKa',
        interpretation='all-data scalar-label training shakedown; resubstitution diagnostics only',
        config=dict(dtype='float32',damped_seed_steps=1024,lm_steps=32,lm_tolerance=1e-6,
                    residual_tolerance=2e-5,learning_rate=.02,gradient_clip=1.,updates=80,
                    prior_weight=.01,scale_parameterization='exp(log(4)*tanh(theta))',family_uniform_weighting=True),
        labels=len(records),families=len(family_counts),structures=len(structures),
        record_ids=[r['record_id'] for r in records],family_counts=dict(sorted(family_counts.items())),
        records=records,structure_tasks=[r['task_id'] for r in structures],
        admission_release_sha256=digest(admission/'release.json'),admission_ledger_sha256=digest(admission/'admission_ledger.csv'),
        fit_gate_sha256=digest(repo/'experiments/1_benchmark/curation/experimental_fit_gates_v1.json'),
        implementation_sha256=digest(Path(__file__)),model_fit=True,headline_evaluation=False)
    atomic_json(out/'manifest.json',manifest)
    print(json.dumps({'prepared_structures':len(structures),'labels':len(records),'families':len(family_counts)},indent=2),flush=True)


def load_task(task_id):
    receipt = read(out/'prepared'/task_id/'receipt.json')
    assert digest(out/'prepared'/task_id/'cache.npz') == receipt['cache_sha256']
    cache = StructureCache.load(out/'prepared'/task_id/'cache.npz')
    model = TitrationModel(cache,config=config,backend='dense')
    arrays = {k:np.asarray(v,dtype=np.float32) if np.issubdtype(np.asarray(v).dtype,np.floating) else np.asarray(v)
              for k,v in model.arrays.items()}
    probabilities = np.asarray(model.native_probabilities,dtype=np.float32)
    active = active_channels(cache,probabilities)
    valid = np.ones(len(active),bool)
    sites = receipt['sites']
    ph = np.asarray([s['target'] for s in sites],np.float32)
    index = np.asarray([s['index'] for s in sites],np.int32)
    group = np.asarray([s['group_index'] for s in sites],np.int32)
    target = np.asarray([s['target'] for s in sites],np.float32)
    weight = np.asarray([s['weight'] for s in sites],np.float32)
    return receipt,arrays,probabilities,active,valid,ph,index,group,target,weight


def make_functions():
    def objective(theta,arrays,probabilities,active,valid,ph,index,group,target,weight,accepted):
        terms = local_terms(theta,arrays,probabilities,config)
        curves,_ = local_terms_curve_kernel(arrays,terms,ph,active,valid,config=config,
            solver_config=solver_config,initialization='production',gradient_mask=accepted,seed_steps=None)
        prediction = curves.effective_pka[jnp.arange(len(ph)),index,group]
        error = prediction-target
        loss = jnp.sum(weight*optax.huber_loss(error,delta=1.))
        return loss,(prediction,error,curves.converged,curves.residual)

    def midpoint(theta,arrays,probabilities,active,valid,index,group):
        terms = local_terms(theta,arrays,probabilities,config)
        curves,_ = local_terms_curve_kernel(arrays,terms,jnp.asarray(grid),active,valid,config=config,
            solver_config=solver_config,initialization='production',gradient_mask=jnp.ones(len(grid),bool),seed_steps=None)
        result = _grid_pka_result(arrays,curves,jnp.asarray(grid),config)
        return result.value[index,group],result.valid[index,group],curves.converged,curves.residual
    return jax.jit(jax.value_and_grad(objective,has_aux=True)),jax.jit(midpoint)


def fit():
    manifest = read(out/'manifest.json')
    assert digest(admission/'release.json') == manifest['admission_release_sha256']
    assert digest(admission/'admission_ledger.csv') == manifest['admission_ledger_sha256']
    tasks = [load_task(task_id) for task_id in manifest['structure_tasks']]
    value_grad,midpoint = make_functions()
    theta = jnp.zeros(3,jnp.float32)
    optimizer = optax.chain(optax.clip_by_global_norm(manifest['config']['gradient_clip']),optax.adam(manifest['config']['learning_rate']))
    state = optimizer.init(theta); history=[]; started=time.monotonic()

    def audit_and_grad(parameters):
        total=jnp.asarray(0.,jnp.float32); gradients=[]; observations=[]
        for task in tasks:
            receipt,arrays,p,active,valid,ph,index,group,target,weight = task
            accepted=np.ones(len(ph),bool)
            (loss,aux),gradient=value_grad(parameters,arrays,p,active,valid,ph,index,group,target,weight,accepted)
            prediction,error,converged,residual=map(np.asarray,aux)
            if not converged.all() or not np.isfinite(prediction).all() or float(np.max(residual)) >= config.residual_tolerance:
                raise RuntimeError(f"scalar solve failed for {receipt['task_id']}: converged={converged.tolist()} residual={residual.tolist()}")
            if not np.isfinite(np.asarray(gradient)).all():
                raise FloatingPointError(f"nonfinite gradient for {receipt['task_id']}")
            total=total+loss;gradients.append(gradient)
            observations.extend(dict(record_id=s['record_id'],effective_pka=float(y),residual=float(e),solver_residual=float(r))
                                for s,y,e,r in zip(receipt['sites'],prediction,error,residual))
        prior=manifest['config']['prior_weight']*jnp.mean(jnp.log(jnp.stack(physical_scales(parameters)))**2)
        prior_grad=jax.grad(lambda x:manifest['config']['prior_weight']*jnp.mean(jnp.log(jnp.stack(physical_scales(x)))**2))(parameters)
        return total+prior,jnp.sum(jnp.stack(gradients),axis=0)+prior_grad,observations,float(prior)

    initial_loss,initial_gradient,initial_observations,initial_prior=audit_and_grad(theta)
    if not np.isfinite(float(initial_loss)) or not np.any(np.asarray(initial_gradient)!=0):
        raise RuntimeError('initial scalar objective/gradient gate failed')
    for step in range(1,manifest['config']['updates']+1):
        loss,gradient,observations,prior=audit_and_grad(theta)
        updates,state=optimizer.update(gradient,state,theta);theta=optax.apply_updates(theta,updates)
        row=dict(step=step,loss=float(loss),prior=prior,gradient=np.asarray(gradient).tolist(),theta=np.asarray(theta).tolist(),
                 scales=[float(x) for x in physical_scales(theta)],seconds=time.monotonic()-started)
        history.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('gradient','theta')}),flush=True)

    final_loss,final_gradient,final_observations,final_prior=audit_and_grad(theta)
    records={r['record_id']:r for r in manifest['records']}; predictions=[]
    initial_theta=jnp.zeros(3,jnp.float32)
    for task in tasks:
        receipt,arrays,p,active,valid,ph,index,group,target,weight=task
        before=midpoint(initial_theta,arrays,p,active,valid,index,group)
        after=midpoint(theta,arrays,p,active,valid,index,group)
        for j,site in enumerate(receipt['sites']):
            row=records[site['record_id']]
            predictions.append(dict(record_id=site['record_id'],family_id=row['family_id'],protein_name=row['protein_name'],
                structure_pdb=row['structure_pdb'],chain=row['chain'],resnum=row['resnum'],group=row['group'],target=row['target'],
                null_pka=row['null_pka'],weight=row['fit_weight'],frozen_pka=float(np.asarray(before[0])[j]),
                frozen_valid=bool(np.asarray(before[1])[j]),fitted_pka=float(np.asarray(after[0])[j]),fitted_valid=bool(np.asarray(after[1])[j])))
    with (out/'predictions.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    with (out/'history.jsonl').open('w') as handle:
        for row in history:handle.write(json.dumps(row,sort_keys=True)+'\n')
    atomic_json(out/'fit.json',dict(status='complete',updates=len(history),initial_loss=float(initial_loss),final_loss=float(final_loss),
        initial_gradient=np.asarray(initial_gradient).tolist(),final_gradient=np.asarray(final_gradient).tolist(),
        theta=np.asarray(theta).tolist(),scales=[float(x) for x in physical_scales(theta)],
        initial_prior=initial_prior,final_prior=final_prior,wall_seconds=time.monotonic()-started,
        initial_observations=initial_observations,final_observations=final_observations,
        manifest_sha256=digest(out/'manifest.json'),model_fit=True,headline_evaluation=False))
    print(json.dumps(read(out/'fit.json'),indent=2),flush=True)


def metrics(rows,key):
    valid=[r for r in rows if key=='null_pka' or r[key.replace('_pka','_valid')]=='True']
    errors=np.asarray([float(r[key])-float(r['target']) for r in valid])
    by_family=defaultdict(list)
    for r,e in zip(valid,errors):by_family[r['family_id']].append(abs(e))
    return dict(n=len(valid),mae=float(np.mean(abs(errors))),rmse=float(np.sqrt(np.mean(errors**2))),
                family_macro_mae=float(np.mean([np.mean(v) for v in by_family.values()])),families=len(by_family))


def report():
    rows=list(csv.DictReader((out/'predictions.csv').open()));fit_result=read(out/'fit.json');manifest=read(out/'manifest.json')
    scores={name:metrics(rows,key) for name,key in [('Residue-type null','null_pka'),('Frozen JAX-Ka','frozen_pka'),('Fitted JAX-Ka','fitted_pka')]}
    null_mse=scores['Residue-type null']['rmse']**2
    for score in scores.values():score['skill_vs_null']=1-score['rmse']**2/null_mse
    atomic_json(out/'scores.json',scores)
    lines=['# First experimental scalar-label fit — JAX-Ka','',
        '**This is an all-data training shakedown on 12 labels from five sequence families. Its fitted scores are resubstitution diagnostics, not independent performance estimates.**','',
        'The fit changes only JAX-Ka’s three shared physical scales. Each experimental scalar supervises `effective_pka(pH = measured pKa) - measured pKa` through the same implicit solver used by the paired-curve trainer. Families receive equal total weight; labels within a family split that weight. No titration curves were fabricated.','',
        '## Fit result','', '| Scale | Initial | Fitted |','|---|---:|---:|']
    for name,value in zip(('Desolvation','Hydrogen bond + reorganization','Coulomb'),fit_result['scales']):lines.append(f'| {name} | 1.000 | {value:.4f} |')
    lines += ['',f"Weighted Huber objective: {fit_result['initial_loss']:.6f} → {fit_result['final_loss']:.6f} over {fit_result['updates']} fixed updates.",'',
        '## Training-set diagnostics','', '| Model | Valid labels | MAE | RMSE | Family-macro MAE | Skill vs null |','|---|---:|---:|---:|---:|---:|']
    for name,score in scores.items():lines.append(f"| {name} | {score['n']} | {score['mae']:.3f} | {score['rmse']:.3f} | {score['family_macro_mae']:.3f} | {score['skill_vs_null']:.3f} |")
    lines += ['', '## Per-label readout','', '| Record | Family/system | Site | Target | Frozen | Fitted |','|---:|---|---|---:|---:|---:|']
    for r in sorted(rows,key=lambda x:int(x['record_id'])):
        frozen=f"{float(r['frozen_pka']):.3f}" if r['frozen_valid']=='True' else 'invalid'
        fitted=f"{float(r['fitted_pka']):.3f}" if r['fitted_valid']=='True' else 'invalid'
        lines.append(f"| {r['record_id']} | {r['protein_name']} | {r['group']} {r['resnum']} | {float(r['target']):.2f} | {frozen} | {fitted} |")
    lines += ['', '## Interpretation','',
        'A falling objective and changed finite parameters establish that experimental scalar labels propagate through the shared LocalTerms/implicit-solver training path. They do not establish generalization. The next statistical test is leave-one-family-out fitting across the five families; no member of a held-out family may influence its prediction.','',
        f"Inputs are frozen by `{digest(out/'manifest.json')}`. `predictions.csv`, `scores.json`, `fit.json` and `history.jsonl` are machine-readable outputs."]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    outputs={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='release.json'}
    atomic_json(out/'release.json',dict(version='experimental-fit-v1',created='2026-10-06',manifest_sha256=digest(out/'manifest.json'),
        admission_release_sha256=manifest['admission_release_sha256'],outputs=outputs,model_fit=True,
        independent_evaluation=False,training_labels=12,training_families=5,job=os.environ['SLURM_JOB_ID']))
    print(json.dumps(scores,indent=2),flush=True)


if __name__ == '__main__':
    {'prepare':initialize,'fit':fit,'report':report}[sys.argv[1]]()
