"""Leave-one-family-out experimental fitting of JAX-Ka physical scales."""
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
jax.config.update('jax_enable_x64',False)
import jax.numpy as jnp
import numpy as np
import optax

from fit_experimental_jaxka import load_task, make_functions, physical_scales

runtime=Path(os.environ['PKABENCH_RUNTIME'])
source=runtime/'experimental/experimental-fit-v1'
out=runtime/'experimental/experimental-lofo-v1'


def read(path):return json.loads(Path(path).read_text())


def task_family(task,record_family):
    families={record_family[s['record_id']] for s in task[0]['sites']}
    assert len(families)==1
    return next(iter(families))


def with_fold_weights(task,family_counts,nfamilies):
    values=list(task);family=task_family(task,_record_family)
    values[-1]=np.asarray([1/(nfamilies*family_counts[family]) for _ in task[0]['sites']],np.float32)
    return tuple(values)


def midpoint_rows(midpoint,tasks,theta):
    rows={}
    for task in tasks:
        receipt,arrays,p,active,valid,ph,index,group,target,weight=task
        value,ok,converged,residual=map(np.asarray,midpoint(theta,arrays,p,active,valid,index,group))
        for j,site in enumerate(receipt['sites']):
            rows[site['record_id']]=dict(value=float(value[j]),valid=bool(ok[j]),
                grid_converged=bool(converged.all()),max_grid_residual=float(np.max(residual)))
    return rows


def objective_and_gradient(value_grad,tasks,theta,prior_weight,need_gradient=True):
    total=jnp.asarray(0.,jnp.float32);gradients=[]
    for task in tasks:
        receipt,arrays,p,active,valid,ph,index,group,target,weight=task
        accepted=np.ones(len(ph),bool)
        (loss,aux),gradient=value_grad(theta,arrays,p,active,valid,ph,index,group,target,weight,accepted)
        if not np.asarray(aux[2]).all() or not np.isfinite(np.asarray(aux[0])).all():
            raise RuntimeError(f"target-pH solve failed: {receipt['task_id']}")
        total=total+loss
        if need_gradient:gradients.append(gradient)
    prior=prior_weight*jnp.mean(jnp.log(jnp.stack(physical_scales(theta)))**2)
    total=total+prior
    if not need_gradient:return float(total)
    prior_gradient=jax.grad(lambda x:prior_weight*jnp.mean(jnp.log(jnp.stack(physical_scales(x)))**2))(theta)
    gradient=jnp.sum(jnp.stack(gradients),axis=0)+prior_gradient
    if not np.isfinite(float(total)) or not np.isfinite(np.asarray(gradient)).all():raise FloatingPointError('nonfinite fold objective')
    return float(total),gradient


def initialize():
    if out.exists():
        if (out/'release.json').exists():raise FileExistsError(f'completed immutable release exists: {out}')
        shutil.rmtree(out)
    out.mkdir(parents=True)
    source_manifest=read(source/'manifest.json')
    families=sorted(source_manifest['family_counts'])
    assert len(families)==5 and source_manifest['labels']==12
    manifest=dict(version='experimental-lofo-v1',created='2026-10-06',families=families,folds=len(families),
        training_policy='For each fold, train on four families with equal total family weight; held-out family is not loaded until checkpoint selection is complete.',
        selection_policy='Minimum training scalar objective among saved checkpoints retaining every initially valid training-family production midpoint.',
        updates=80,source=str(source),source_release_sha256=digest(source/'release.json'),
        source_manifest_sha256=digest(source/'manifest.json'),implementation_sha256=digest(Path(__file__)),
        model_fit=True,independent_evaluation=True)
    atomic_json(out/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)


def fit_fold(index):
    global _record_family
    manifest=read(out/'manifest.json');source_manifest=read(source/'manifest.json')
    assert digest(source/'release.json')==manifest['source_release_sha256']
    heldout=manifest['families'][index];dest=out/f'fold-{index}'
    if dest.exists():
        if (dest/'result.json').exists():raise FileExistsError(f'completed fold exists: {dest}')
        shutil.rmtree(dest)
    dest.mkdir()
    _record_family={r['record_id']:r['family_id'] for r in source_manifest['records']}
    records={r['record_id']:r for r in source_manifest['records']}
    all_tasks=[load_task(task_id) for task_id in source_manifest['structure_tasks']]
    train_raw=[task for task in all_tasks if task_family(task,_record_family)!=heldout]
    heldout_ids={r['record_id'] for r in source_manifest['records'] if r['family_id']==heldout}
    # Do not retain held-out task payloads while training or selecting.
    del all_tasks
    train_counts=Counter(r['family_id'] for r in source_manifest['records'] if r['family_id']!=heldout)
    assert len(train_counts)==4
    train=[with_fold_weights(task,train_counts,4) for task in train_raw]
    value_grad,midpoint=make_functions();theta=jnp.zeros(3,jnp.float32)
    optimizer=optax.chain(optax.clip_by_global_norm(source_manifest['config']['gradient_clip']),
                          optax.adam(source_manifest['config']['learning_rate']))
    state=optimizer.init(theta);history=[];started=time.monotonic()
    initial_loss,initial_gradient=objective_and_gradient(value_grad,train,theta,source_manifest['config']['prior_weight'])
    assert np.any(np.asarray(initial_gradient)!=0)
    for step in range(1,source_manifest['config']['updates']+1):
        loss,gradient=objective_and_gradient(value_grad,train,theta,source_manifest['config']['prior_weight'])
        updates,state=optimizer.update(gradient,state,theta);theta=optax.apply_updates(theta,updates)
        history.append(dict(step=step,objective_before_update=loss,theta=np.asarray(theta).tolist(),
                            scales=[float(x) for x in physical_scales(theta)]))
    with (dest/'history.jsonl').open('w') as handle:
        for row in history:handle.write(json.dumps(row,sort_keys=True)+'\n')

    zero=jnp.zeros(3,jnp.float32);baseline_train=midpoint_rows(midpoint,train,zero)
    required={record_id for record_id,row in baseline_train.items() if row['valid']}
    candidates=[dict(step=0,theta=[0.,0.,0.])]+history
    audit=[];states={}
    for candidate in candidates:
        candidate_theta=jnp.asarray(candidate['theta'],jnp.float32)
        state_rows=midpoint_rows(midpoint,train,candidate_theta)
        valid={record_id for record_id,row in state_rows.items() if row['valid']}
        loss=objective_and_gradient(value_grad,train,candidate_theta,source_manifest['config']['prior_weight'],need_gradient=False)
        row=dict(step=candidate['step'],training_objective=loss,training_valid_labels=len(valid),
                 required_training_valid_retained=len(required&valid),coverage_preserving=required<=valid,
                 scales=[float(x) for x in physical_scales(candidate_theta)])
        audit.append(row);states[candidate['step']]=(candidate_theta,state_rows)
    admissible=[row for row in audit if row['coverage_preserving']]
    assert admissible
    chosen=min(admissible,key=lambda row:(row['training_objective'],-row['training_valid_labels'],row['step']))
    selected_theta,selected_train=states[chosen['step']]
    with (dest/'checkpoint_audit.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)

    # Held-out structures and targets are loaded only after the checkpoint is fixed.
    all_tasks=[load_task(task_id) for task_id in source_manifest['structure_tasks']]
    heldout_tasks=[task for task in all_tasks if task_family(task,_record_family)==heldout]
    assert {s['record_id'] for task in heldout_tasks for s in task[0]['sites']}==heldout_ids
    frozen=midpoint_rows(midpoint,heldout_tasks,zero)
    prediction=midpoint_rows(midpoint,heldout_tasks,selected_theta)
    rows=[]
    for record_id in sorted(heldout_ids,key=int):
        record=records[record_id]
        rows.append(dict(record_id=record_id,family_id=heldout,protein_name=record['protein_name'],structure_pdb=record['structure_pdb'],
            chain=record['chain'],resnum=record['resnum'],group=record['group'],target=record['target'],null_pka=record['null_pka'],
            frozen_pka=frozen[record_id]['value'],frozen_valid=frozen[record_id]['valid'],
            lofo_pka=prediction[record_id]['value'],lofo_valid=prediction[record_id]['valid']))
    with (dest/'heldout_predictions.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(fold=index,heldout_family=heldout,heldout_record_ids=sorted(heldout_ids,key=int),
        training_labels=sum(len(task[0]['sites']) for task in train),training_families=4,
        initial_training_objective=initial_loss,initial_training_valid_labels=len(required),
        selected_step=chosen['step'],selected_training_objective=chosen['training_objective'],
        selected_training_valid_labels=chosen['training_valid_labels'],selected_scales=chosen['scales'],
        heldout_labels=len(rows),heldout_frozen_valid=sum(r['frozen_valid'] for r in rows),
        heldout_lofo_valid=sum(r['lofo_valid'] for r in rows),selection_used_heldout=False,
        manifest_sha256=digest(out/'manifest.json'),history_sha256=digest(dest/'history.jsonl'),
        audit_sha256=digest(dest/'checkpoint_audit.csv'),predictions_sha256=digest(dest/'heldout_predictions.csv'),
        wall_seconds=time.monotonic()-started,job=os.environ['SLURM_JOB_ID'])
    atomic_json(dest/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


def measures(rows,key,common):
    use=[r for r in rows if r['record_id'] in common]
    error=np.asarray([float(r[key])-float(r['target']) for r in use])
    by_family=defaultdict(list)
    for row,e in zip(use,error):by_family[row['family_id']].append(abs(e))
    return dict(n=len(use),families=len(by_family),mae=float(np.mean(abs(error))),rmse=float(np.sqrt(np.mean(error**2))),
                family_macro_mae=float(np.mean([np.mean(v) for v in by_family.values()])))


def collect():
    manifest=read(out/'manifest.json');rows=[];folds=[]
    for index,family in enumerate(manifest['families']):
        dest=out/f'fold-{index}';result=read(dest/'result.json')
        assert result['heldout_family']==family and result['selection_used_heldout'] is False
        assert result['manifest_sha256']==digest(out/'manifest.json')
        for name,key in [('history.jsonl','history_sha256'),('checkpoint_audit.csv','audit_sha256'),('heldout_predictions.csv','predictions_sha256')]:
            assert digest(dest/name)==result[key]
        folds.append(result);rows.extend(csv.DictReader((dest/'heldout_predictions.csv').open()))
    assert len(rows)==12 and len({r['record_id'] for r in rows})==12
    with (out/'heldout_predictions.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    common={r['record_id'] for r in rows if r['frozen_valid']=='True' and r['lofo_valid']=='True'}
    scores={name:measures(rows,key,common) for name,key in (
        ('Residue-type null','null_pka'),('Frozen JAX-Ka','frozen_pka'),('LOFO-fitted JAX-Ka','lofo_pka'))}
    null_mse=scores['Residue-type null']['rmse']**2
    for score in scores.values():score['skill_vs_null']=1-score['rmse']**2/null_mse
    family_rows=[]
    for family in manifest['families']:
        family_common={r['record_id'] for r in rows if r['family_id']==family and r['record_id'] in common}
        if not family_common:continue
        f=measures(rows,'frozen_pka',family_common);l=measures(rows,'lofo_pka',family_common)
        family_rows.append(dict(family_id=family,protein_names='; '.join(sorted({r['protein_name'] for r in rows if r['family_id']==family})),
            n=len(family_common),frozen_mae=f['mae'],lofo_mae=l['mae'],delta_mae=l['mae']-f['mae']))
    with (out/'family_scores.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(family_rows[0]));writer.writeheader();writer.writerows(family_rows)
    atomic_json(out/'scores.json',dict(common_support_record_ids=sorted(common,key=int),methods=scores,folds=folds))
    lines=['# Leave-one-family-out experimental fit — JAX-Ka','',
        '**Each prediction comes from a model fitted without any label from that sequence family. This is the first transfer diagnostic, but it contains only five families and is not a stable benchmark ranking.**','',
        'Every fold recomputes equal-family training weights over its four training families. Checkpoint selection uses only training objective and training midpoint coverage; held-out structures and labels are loaded only after selection.','',
        '## Common-support results','', '| Model | Labels | Families | MAE | RMSE | Family-macro MAE | Skill vs null |','|---|---:|---:|---:|---:|---:|---:|']
    for name,score in scores.items():lines.append(f"| {name} | {score['n']} | {score['families']} | {score['mae']:.3f} | {score['rmse']:.3f} | {score['family_macro_mae']:.3f} | {score['skill_vs_null']:.3f} |")
    lines += ['', '## Held-out family changes','', '| Family/system | Labels | Frozen MAE | LOFO MAE | Change |','|---|---:|---:|---:|---:|']
    for row in family_rows:lines.append(f"| {row['protein_names']} | {row['n']} | {row['frozen_mae']:.3f} | {row['lofo_mae']:.3f} | {row['delta_mae']:+.3f} |")
    lines += ['', '## Fold selection','', '| Held-out family | Training labels | Selected step | Training objective | Held-out valid frozen → fitted | Scales |','|---|---:|---:|---:|---:|---|']
    for row in folds:lines.append(f"| {row['heldout_family']} | {row['training_labels']} | {row['selected_step']} | {row['selected_training_objective']:.4f} | {row['heldout_frozen_valid']} → {row['heldout_lofo_valid']} | " + ', '.join(f"{x:.3f}" for x in row['selected_scales']) + ' |')
    lines += ['', 'The comparison uses only labels with valid frozen and LOFO production midpoints. DsbA Cys30 and T4 lysozyme His31 were already invalid before fitting and do not enter the common-support errors. With five families, uncertainty is dominated by which biochemical systems are present; no significance claim is made.','']
    (out/'report.md').write_text('\n'.join(lines))
    outputs={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='release.json'}
    atomic_json(out/'release.json',dict(version='experimental-lofo-v1',created='2026-10-06',manifest_sha256=digest(out/'manifest.json'),
        source_release_sha256=manifest['source_release_sha256'],outputs=outputs,fold_results_sha256={f"fold-{r['fold']}":digest(out/f"fold-{r['fold']}"/'result.json') for r in folds},
        model_fit=True,independent_evaluation=True,families=5,labels=12,common_support=len(common),job=os.environ['SLURM_JOB_ID']))
    print(json.dumps({'scores':scores,'families':family_rows,'common_support':sorted(common,key=int)},indent=2),flush=True)


_record_family={}
if __name__=='__main__':
    command=sys.argv[1]
    if command=='init':initialize()
    elif command=='fit':fit_fold(int(os.environ['SLURM_ARRAY_TASK_ID']))
    elif command=='collect':collect()
    else:raise ValueError(command)
