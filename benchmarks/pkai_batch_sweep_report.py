"""Matched validation and throughput report for the pKAI batch sweep."""
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.frozen_score import write_csv
from pkpdb_pretraining_report import read_predictions,summary


def read(path):return json.loads(Path(path).read_text())


def report(out):
    batches=(64,128,256,512,1024);paths={b:out/f'batch-{b}' for b in batches}
    predictions={b:read_predictions(path/'validation_predictions.csv') for b,path in paths.items()}
    checks={b:read(path/'verification.json') for b,path in paths.items()};assert all(v['passed'] for v in checks.values())
    common=set.intersection(*(set(rows) for rows in predictions.values()));assert common
    for key in common:
        values=[rows[key] for rows in predictions.values()]
        assert len({r['component_id'] for r in values})==1
        assert max(r['teacher_pka'] for r in values)-min(r['teacher_pka'] for r in values)<1e-5
    rows=[];pergroup={}
    for batch,pred in predictions.items():
        metrics,complexes=summary(pred,common);groups=defaultdict(list)
        for row in complexes:groups[row['component_id']].append(row['mae'])
        pergroup[batch]={group:float(np.mean(values)) for group,values in groups.items()}
        history=read(paths[batch]/'history.json');seconds=np.array([r['seconds'] for r in history]);final=read(paths[batch]/'final.json')
        rows.append(dict(batch_size=batch,sites=len(common),mae=metrics['mae'],ci_low=metrics['mae_ci95'][0],
            ci_high=metrics['mae_ci95'][1],rmse=metrics['rmse'],median_epoch_seconds=float(np.median(seconds)),
            total_train_seconds=float(seconds.sum()),epochs=len(history),best_epoch=final['best_epoch'],
            peak_allocated_gib=checks[batch]['gpu_peak_allocated_bytes']/2**30,
            peak_reserved_gib=checks[batch]['gpu_peak_reserved_bytes']/2**30))
    differences=[];baseline=pergroup[256]
    for batch in batches:
        if batch==256:continue
        keys=sorted(baseline);assert set(keys)==set(pergroup[batch])
        delta=np.array([pergroup[batch][key]-baseline[key] for key in keys]);rng=np.random.default_rng(20261006)
        boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
        differences.append(dict(batch_size=batch,delta_mae=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    write_csv(out/'comparison.csv',rows)
    atomic_json(out/'comparison.json',dict(rows=rows,differences=differences,
        prediction_sha256={str(b):digest(path/'validation_predictions.csv') for b,path in paths.items()}))
    lines=['# pKAI batch-size sweep','',
        'Cleaned 5k cohort, seed 17, native optimizer/dropout/early-stopping recipe, full float32, unaugmented validation. No test evaluation.','',
        '| Batch | Matched sites | Group-macro MAE | 95% CI | Epochs / best | Median epoch | Peak allocated / reserved VRAM |',
        '|---:|---:|---:|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['batch_size']} | {row['sites']:,} | {row['mae']:.4f} | {row['ci_low']:.4f}–{row['ci_high']:.4f} | {row['epochs']} / {row['best_epoch']} | {row['median_epoch_seconds']:.2f} s | {row['peak_allocated_gib']:.2f} / {row['peak_reserved_gib']:.2f} GiB |")
    lines+=['','| Batch versus native 256 | ΔMAE | 95% paired group-bootstrap CI |','|---:|---:|---|']
    for row in differences:lines.append(f"| {row['batch_size']} | {row['delta_mae']:+.4f} | {row['ci95'][0]:+.4f}–{row['ci95'][1]:+.4f} |")
    lines+=['','Negative ΔMAE favors the candidate over batch 256.',
        'Learning rate and early-stopping settings are fixed, so batch size changes updates per epoch. Single-seed development results against PypKa teacher labels.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'verification.json',dict(passed=True,batches=list(batches),common_sites=len(common)))


if __name__=='__main__':
    require_compute(threads=2,allow_comp1400=True)
    report(Path(os.environ['PKABENCH_RUNTIME'])/'pretraining/pkai-batch-sweep-v1')
