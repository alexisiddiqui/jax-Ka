"""Matched side-chain GQT augmentation comparison."""
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.frozen_score import write_csv
from pkatrain.context_augmentation import epoch_mask,mask_digest
from pkpdb_pretraining_report import read_predictions,summary


def read(path):return json.loads(Path(path).read_text())


def report(out):
    arms=('baseline','dropout','mask','both')
    paths={arm:out/f'gqt-{arm}/seed-17' for arm in arms}
    predictions={arm:read_predictions(path/'validation_predictions.csv') for arm,path in paths.items()}
    assert all(read(path/'verification.json')['passed'] for path in paths.values())
    common=set.intersection(*(set(rows) for rows in predictions.values()));assert common
    for key in common:
        values=[rows[key] for rows in predictions.values()]
        assert len({r['component_id'] for r in values})==1
        assert max(r['teacher_pka'] for r in values)-min(r['teacher_pka'] for r in values)<1e-5
    histories={arm:read(paths[arm]/'history.json') for arm in ('mask','both')}
    epochs=min(map(len,histories.values()));plan=read(out/'contexts/plan.json')
    for epoch in range(1,epochs+1):
        expected=mask_digest(epoch_mask(plan,17,epoch))
        assert all(h[epoch-1]['context_mask_sha256']==expected for h in histories.values())
    rows=[];pergroup={}
    for arm,pred in predictions.items():
        metrics,complexes=summary(pred,common);groups=defaultdict(list)
        for row in complexes:groups[row['component_id']].append(row['mae'])
        pergroup[arm]={group:float(np.mean(values)) for group,values in groups.items()}
        rows.append(dict(arm=arm,sites=len(common),mae=metrics['mae'],ci_low=metrics['mae_ci95'][0],
            ci_high=metrics['mae_ci95'][1],rmse=metrics['rmse']))
    differences=[];baseline=pergroup['baseline']
    for arm in ('dropout','mask','both'):
        keys=sorted(baseline);assert set(keys)==set(pergroup[arm])
        delta=np.array([pergroup[arm][key]-baseline[key] for key in keys])
        rng=np.random.default_rng(20261006)
        boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
        differences.append(dict(arm=arm,delta_mae=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    write_csv(out/'comparison.csv',rows)
    atomic_json(out/'comparison.json',dict(rows=rows,differences=differences,matched_mask_epochs=epochs,
        prediction_sha256={arm:digest(path/'validation_predictions.csv') for arm,path in paths.items()}))
    lines=['# Cleaned 5k side-chain GQT augmentation comparison','',
        'Seed 17, from scratch. Identical unaugmented validation sites. No test evaluation.','',
        '| Arm | Matched sites | Group-macro MAE | 95% group-bootstrap CI |','|---|---:|---:|---|']
    for row in rows:lines.append(f"| {row['arm']} | {row['sites']:,} | {row['mae']:.4f} | {row['ci_low']:.4f}–{row['ci_high']:.4f} |")
    lines+=['','| Change versus side-chain baseline | ΔMAE | 95% paired group-bootstrap CI |','|---|---:|---|']
    for row in differences:lines.append(f"| {row['arm']} | {row['delta_mae']:+.4f} | {row['ci95'][0]:+.4f}–{row['ci95'][1]:+.4f} |")
    lines+=['',f'Matching per-structure context masks verified for {epochs} epochs. Negative ΔMAE favors augmentation.',
        'Masked side-chain residues retain amino-acid identity but lose side-chain coordinates, atom-presence bits, frame validity, and all incident geometric edges.',
        'Fixed epoch 20; single-seed development results against PypKa teacher labels.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'verification.json',dict(passed=True,arms=4,common_sites=len(common),matched_mask_epochs=epochs))


if __name__=='__main__':
    require_compute(threads=2,allow_comp1400=True)
    report(Path(os.environ['PKABENCH_RUNTIME'])/'pretraining/augmentation-sidechains-v1')
