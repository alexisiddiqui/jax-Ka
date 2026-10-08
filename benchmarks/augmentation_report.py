"""Paired common-support augmentation comparisons and shared-mask audit."""
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.frozen_score import write_csv
from pkatrain.context_augmentation import epoch_mask,mask_digest
from pkpdb_pretraining_report import read_predictions,summary


def read(path):return json.loads(Path(path).read_text())


def report(out):
    paths={f'gqt-{a}':out/f'gqt-{a}/seed-17' for a in ('baseline','dropout','mask','both')}
    paths.update({f'pkai-{a}':out/f'pkai-{a}' for a in ('baseline','mask')})
    predictions={}
    for arm,path in paths.items():
        assert read(path/'verification.json')['passed']
        predictions[arm]=read_predictions(path/'validation_predictions.csv')
    common=set.intersection(*(set(rows) for rows in predictions.values()));assert common
    for key in common:
        values=[rows[key] for rows in predictions.values()]
        assert len({r['component_id'] for r in values})==1
        y=[r['teacher_pka'] for r in values];assert max(y)-min(y)<1e-5
    histories={arm:read(paths[arm]/'history.json') for arm in ('gqt-mask','gqt-both','pkai-mask')}
    epochs=min(len(h) for h in histories.values());assert epochs>0
    plan=read(out/'contexts/plan.json')
    receipts=read(out/'contexts/receipts.json')
    native_only={r['complex_id']:r.get('native_only_residues',[]) for r in receipts if r.get('native_only_residues')}
    for epoch in range(1,epochs+1):
        expected=mask_digest(epoch_mask(plan,17,epoch))
        assert all(h[epoch-1]['context_mask_sha256']==expected for h in histories.values()),epoch
    rows=[];pergroup={}
    for arm,pred in predictions.items():
        metrics,complexes=summary(pred,common);groups=defaultdict(list)
        for r in complexes:groups[r['component_id']].append(r['mae'])
        pergroup[arm]={g:float(np.mean(v)) for g,v in groups.items()}
        rows.append(dict(arm=arm,sites=len(common),mae=metrics['mae'],ci_low=metrics['mae_ci95'][0],
                         ci_high=metrics['mae_ci95'][1],rmse=metrics['rmse']))
    differences=[]
    for arm,values in pergroup.items():
        if arm.endswith('baseline'):continue
        baseline=pergroup[arm.split('-')[0]+'-baseline'];keys=sorted(values)
        assert set(keys)==set(baseline)
        delta=np.array([values[k]-baseline[k] for k in keys]);rng=np.random.default_rng(20261006)
        boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
        differences.append(dict(arm=arm,delta_mae=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    write_csv(out/'comparison.csv',rows)
    atomic_json(out/'comparison.json',dict(rows=rows,differences=differences,matched_mask_epochs=epochs,native_only_context=native_only,
        prediction_sha256={a:digest(p/'validation_predictions.csv') for a,p in paths.items()}))
    lines=['# Cleaned 5k augmentation comparison','',
        'Seed 17, from scratch. Identical unaugmented validation sites. No test evaluation.',
        '', '| Arm | Matched sites | Group-macro MAE | 95% group-bootstrap CI |', '|---|---:|---:|---|']
    for r in rows:lines.append(f"| {r['arm']} | {r['sites']:,} | {r['mae']:.4f} | {r['ci_low']:.4f}–{r['ci_high']:.4f} |")
    lines+=['','| Change versus own control | ΔMAE | 95% paired group-bootstrap CI |','|---|---:|---|']
    for r in differences:lines.append(f"| {r['arm']} | {r['delta_mae']:+.4f} | {r['ci95'][0]:+.4f}–{r['ci95'][1]:+.4f} |")
    lines+=['',f'Matching per-structure context-mask hashes verified for {epochs} shared epochs. Negative ΔMAE favors augmentation.',
        f'Native-only pKAI context in {len(native_only)} training structures was kept fixed in both pKAI arms; identities are listed in comparison.json.',
        'GQT: fixed epoch 20; pKAI: validation-selected checkpoint. Different native inputs/capacities/recipes; single-seed development results.',
        'These results measure clean-validation accuracy, not robustness to real missing regions. Quality masks and targets were preserved.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'verification.json',dict(passed=True,arms=6,common_sites=len(common),matched_mask_epochs=epochs))


if __name__=='__main__':
    # Explicit user authorization for this CPU-only report on comp1400.
    require_compute(threads=2, allow_comp1400=True)
    report(Path(os.environ['PKABENCH_RUNTIME'])/'pretraining/augmentation-v1')
