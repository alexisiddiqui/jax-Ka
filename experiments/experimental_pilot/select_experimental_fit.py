"""Select a coverage-preserving checkpoint from experimental-fit-v1."""
import csv
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

from pkabench.runtime import atomic_json, digest, require_compute

require_compute(threads=2)

import jax
jax.config.update('jax_enable_x64',False)
import jax.numpy as jnp
import numpy as np

from fit_experimental_jaxka import load_task, make_functions, physical_scales

runtime=Path(os.environ['PKABENCH_RUNTIME'])
source=runtime/'experimental/experimental-fit-v1'
out=runtime/'experimental/experimental-fit-v2'


def read(path):return json.loads(Path(path).read_text())


def evaluate_midpoints(midpoint,tasks,theta):
    rows={}
    for task in tasks:
        receipt,arrays,p,active,valid,ph,index,group,target,weight=task
        result=midpoint(theta,arrays,p,active,valid,index,group)
        value,ok,converged,residual=map(np.asarray,result)
        for j,site in enumerate(receipt['sites']):
            rows[site['record_id']]=dict(value=float(value[j]),valid=bool(ok[j]),
                grid_converged=bool(converged.all()),max_grid_residual=float(np.max(residual)))
    return rows


def scalar_objective(value_grad,tasks,theta,prior_weight):
    total=0.
    for task in tasks:
        receipt,arrays,p,active,valid,ph,index,group,target,weight=task
        accepted=np.ones(len(ph),bool)
        (loss,aux),_=value_grad(theta,arrays,p,active,valid,ph,index,group,target,weight,accepted)
        if not np.asarray(aux[2]).all():raise RuntimeError(f"target-pH solve failed: {receipt['task_id']}")
        total+=float(loss)
    prior=prior_weight*float(jnp.mean(jnp.log(jnp.stack(physical_scales(theta)))**2))
    return total+prior


def metric(rows,key,common):
    selected=[r for r in rows if r['record_id'] in common]
    error=np.asarray([float(r[key])-float(r['target']) for r in selected])
    fam=defaultdict(list)
    for row,e in zip(selected,error):fam[row['family_id']].append(abs(e))
    return dict(n=len(selected),families=len(fam),mae=float(np.mean(abs(error))),rmse=float(np.sqrt(np.mean(error**2))),
                family_macro_mae=float(np.mean([np.mean(v) for v in fam.values()])))


def main():
    if out.exists():
        if (out/'release.json').exists():raise FileExistsError(f'completed immutable release exists: {out}')
        shutil.rmtree(out)
    out.mkdir(parents=True)
    manifest=read(source/'manifest.json');fit=read(source/'fit.json')
    assert digest(source/'manifest.json')==fit['manifest_sha256']
    tasks=[load_task(task_id) for task_id in manifest['structure_tasks']]
    value_grad,midpoint=make_functions();started=time.monotonic()
    initial=jnp.zeros(3,jnp.float32); baseline=evaluate_midpoints(midpoint,tasks,initial)
    required={record_id for record_id,row in baseline.items() if row['valid']}
    assert len(required)==10
    candidates=[dict(step=0,theta=[0.,0.,0.])]
    candidates += [dict(step=row['step'],theta=row['theta']) for row in map(json.loads,(source/'history.jsonl').read_text().splitlines())]
    audit=[];states={}
    for candidate in candidates:
        theta=jnp.asarray(candidate['theta'],jnp.float32)
        midpoint_rows=evaluate_midpoints(midpoint,tasks,theta)
        valid={record_id for record_id,row in midpoint_rows.items() if row['valid']}
        loss=scalar_objective(value_grad,tasks,theta,manifest['config']['prior_weight'])
        accepted=required<=valid
        row=dict(step=candidate['step'],objective=loss,valid_labels=len(valid),required_valid_retained=len(required&valid),
                 coverage_preserving=accepted,scales=[float(x) for x in physical_scales(theta)])
        audit.append(row);states[candidate['step']]=(theta,midpoint_rows)
        print(json.dumps(row),flush=True)
    admissible=[r for r in audit if r['coverage_preserving']]
    assert admissible
    chosen=min(admissible,key=lambda r:(r['objective'],-r['valid_labels'],r['step']))
    theta,selected_state=states[chosen['step']]
    final_state=states[80][1]
    records={r['record_id']:r for r in manifest['records']};predictions=[]
    for record_id in manifest['record_ids']:
        row=records[record_id]
        predictions.append(dict(record_id=record_id,family_id=row['family_id'],protein_name=row['protein_name'],
            structure_pdb=row['structure_pdb'],chain=row['chain'],resnum=row['resnum'],group=row['group'],target=row['target'],
            null_pka=row['null_pka'],weight=row['fit_weight'],frozen_pka=baseline[record_id]['value'],
            frozen_valid=baseline[record_id]['valid'],selected_pka=selected_state[record_id]['value'],
            selected_valid=selected_state[record_id]['valid'],unconstrained_final_pka=final_state[record_id]['value'],
            unconstrained_final_valid=final_state[record_id]['valid']))
    with (out/'predictions.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    with (out/'checkpoint_audit.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
    common={r['record_id'] for r in predictions if r['frozen_valid'] and r['selected_valid']}
    scores={name:metric(predictions,key,common) for name,key in (
        ('Residue-type null','null_pka'),('Frozen JAX-Ka','frozen_pka'),('Coverage-preserving fitted JAX-Ka','selected_pka'))}
    null_mse=scores['Residue-type null']['rmse']**2
    for score in scores.values():score['skill_vs_null']=1-score['rmse']**2/null_mse
    atomic_json(out/'scores.json',scores)
    selection=dict(policy='Minimum scalar objective among saved checkpoints retaining every initially valid production midpoint.',
        posthoc_reason='The unconstrained v1 endpoint reduced production midpoint coverage from 10 to 6 labels.',
        required_record_ids=sorted(required,key=int),selected_step=chosen['step'],selected_objective=chosen['objective'],
        selected_scales=chosen['scales'],selected_valid_labels=chosen['valid_labels'],unconstrained_final_valid_labels=sum(r['valid'] for r in final_state.values()),
        checkpoints_audited=len(audit),wall_seconds=time.monotonic()-started)
    atomic_json(out/'selection.json',selection)
    manifest_v2=dict(version='experimental-fit-v2',created='2026-10-06',parent=str(source),parent_release_sha256=digest(source/'release.json'),
        parent_manifest_sha256=digest(source/'manifest.json'),selection=selection,model_fit=True,independent_evaluation=False,
        training_labels=12,training_families=5,implementation_sha256=digest(Path(__file__)))
    atomic_json(out/'manifest.json',manifest_v2)
    lines=['# Coverage-preserving experimental scalar-label fit — JAX-Ka','',
        '**This remains an all-data training shakedown, not an independent performance estimate.**','',
        'The unconstrained 80-update endpoint reduced valid production midpoint coverage from 10/12 to 6/12 even though its target-pH objective improved. This shows that a local scalar residual alone does not protect the global titration curve. The v2 selection gate audits every saved checkpoint and requires every initially valid midpoint to remain valid.','',
        '## Selected checkpoint','', '| Quantity | Result |','|---|---:|',
        f"| Saved update | {chosen['step']} |",f"| Scalar objective | {chosen['objective']:.6f} |",
        f"| Valid midpoint labels | {chosen['valid_labels']}/12 |",f"| Unconstrained endpoint coverage | {selection['unconstrained_final_valid_labels']}/12 |",
        f"| Desolvation scale | {chosen['scales'][0]:.4f} |",f"| Hydrogen bond + reorganization scale | {chosen['scales'][1]:.4f} |",f"| Coulomb scale | {chosen['scales'][2]:.4f} |",'',
        '## Common-support training diagnostics','', '| Model | Labels | Families | MAE | RMSE | Family-macro MAE | Skill vs null |','|---|---:|---:|---:|---:|---:|---:|']
    for name,score in scores.items():lines.append(f"| {name} | {score['n']} | {score['families']} | {score['mae']:.3f} | {score['rmse']:.3f} | {score['family_macro_mae']:.3f} | {score['skill_vs_null']:.3f} |")
    lines += ['', '## Per-label readout','', '| Record | System | Site | Target | Frozen | Selected | Endpoint |','|---:|---|---|---:|---:|---:|---:|']
    for row in sorted(predictions,key=lambda r:int(r['record_id'])):
        display=lambda value,valid:f"{value:.3f}" if valid else f"invalid ({value:.3f})"
        lines.append(f"| {row['record_id']} | {row['protein_name']} | {row['group']} {row['resnum']} | {row['target']:.2f} | {display(row['frozen_pka'],row['frozen_valid'])} | {display(row['selected_pka'],row['selected_valid'])} | {display(row['unconstrained_final_pka'],row['unconstrained_final_valid'])} |")
    lines += ['', 'The common-support table uses the ten labels valid both before and after fitting. It is resubstitution performance and can only diagnose whether the training path moves in a useful direction. Leave-one-family-out fitting is required before estimating transfer to an unseen family.','']
    (out/'report.md').write_text('\n'.join(lines))
    outputs={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='release.json'}
    atomic_json(out/'release.json',dict(version='experimental-fit-v2',created='2026-10-06',manifest_sha256=digest(out/'manifest.json'),
        parent_release_sha256=digest(source/'release.json'),outputs=outputs,model_fit=True,independent_evaluation=False,
        coverage_gate_passed=True,job=os.environ['SLURM_JOB_ID']))
    print(json.dumps({'selection':selection,'scores':scores},indent=2),flush=True)


if __name__=='__main__':main()
