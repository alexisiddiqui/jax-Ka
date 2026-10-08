"""Three-seed backbone training curves and paired epoch-20/100 diagnostics."""
import csv
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pkabench.runtime import require_compute,atomic_json,digest

require_compute()
root=Path(os.environ['PKABENCH_RUNTIME'])/'pretraining'
out=root/'graph-backbone-100e-report';out.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())


def rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def group_errors(data,restype=None):
    complexes=defaultdict(list);component={}
    for r in data:
        if restype is not None and r['group']!=restype:continue
        cid=r['complex_id'];component[cid]=r['component_id']
        complexes[cid].append(abs(float(r['predicted_pka'])-float(r['teacher_pka'])))
    groups=defaultdict(list)
    for cid,values in complexes.items():groups[component[cid]].append(float(np.mean(values)))
    return {g:float(np.mean(v)) for g,v in groups.items()}


fig,axes=plt.subplots(1,3,figsize=(14,4));summary=[];pertype=[];sources={}
for seed in (17,29,43):
    run=root/f'graph-backbone-100e-seed{seed}-v1';dest=run/f'seed-{seed}'
    assert read(dest/'verification.json')['passed']
    h=read(dest/'history.json');assert [r['epoch'] for r in h]==list(range(1,101))
    old=rows(dest/'validation_epoch_020.csv');new=rows(dest/'validation_predictions.csv')
    identity=lambda r:tuple(r[k] for k in ('complex_id','chain','resnum','icode','group','teacher_pka'))
    assert {identity(r) for r in old}=={identity(r) for r in new}
    a=group_errors(old);b=group_errors(new);assert a.keys()==b.keys()
    delta=np.array([b[k]-a[k] for k in sorted(a)])
    rng=np.random.default_rng(20261004);boot=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(axis=1)
    summary.append(dict(seed=seed,epoch20_mae=h[19]['validation']['graph_query']['mae'],
        epoch100_mae=h[-1]['validation']['graph_query']['mae'],delta_100_minus_20=float(delta.mean()),
        delta_ci95=np.quantile(boot,[.025,.975]).tolist(),sites=len(new),groups=len(delta)))
    for group in sorted({r['group'] for r in new}):
        values=group_errors(new,group)
        pertype.append(dict(seed=seed,group=group,mae=float(np.mean(list(values.values()))),components=len(values),
            sites=sum(r['group']==group for r in new)))
    epochs=[r['epoch'] for r in h]
    axes[0].plot(epochs,[r['train_mse'] for r in h],label=f'seed {seed}')
    axes[1].plot(epochs,[r['validation']['graph_query']['mae'] for r in h],label=f'seed {seed}')
    axes[2].plot(epochs,[r['learning_rate'] for r in h],label=f'seed {seed}')
    for p in (run/'manifest.json',dest/'history.json',dest/'validation_epoch_020.csv',dest/'validation_predictions.csv'):
        sources[str(p)]=digest(p)
for ax,title,y in zip(axes,['Sampled training loss','Validation: group-macro MAE','Learning-rate schedule'],['MSE (pKa²)','MAE (pKa)','Learning rate']):
    ax.set(xlabel='Epoch',ylabel=y,title=title);ax.axvline(20,color='gray',ls=':',lw=1);ax.legend()
axes[2].set_yscale('log');fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
groups=sorted({r['group'] for r in pertype});fig,ax=plt.subplots(figsize=(9,4))
for seed in (17,29,43):
    values={r['group']:r['mae'] for r in pertype if r['seed']==seed}
    ax.plot(groups,[values[g] for g in groups],marker='o',label=f'seed {seed}')
ax.set(ylabel='Validation group-macro MAE (pKa)',title='Final-epoch error by titratable group');ax.legend()
fig.tight_layout();fig.savefig(out/'per_type_errors.png',dpi=180);plt.close(fig)
atomic_json(out/'summary.json',dict(runs=summary,per_type=pertype,sources=sources,
    cutoff='20 Å Cα–Cα',strict_backbone=True,bootstrap='2,000 paired sequence-component replicates within each seed'))
text=['# Backbone-only 100-epoch comparison','',
    '49,709 parameters; 20 Å Cα radius; no side-chain coordinates or disulfide flag. Same 477/142 split and masks.',
    '', '| Seed | Epoch 20 MAE | Epoch 100 MAE | Change, 100 − 20 (95% CI) |', '|---|---:|---:|---|']
for r in summary:
    lo,hi=r['delta_ci95'];text.append(f'| {r["seed"]} | {r["epoch20_mae"]:.4f} | {r["epoch100_mae"]:.4f} | {r["delta_100_minus_20"]:+.4f} ({lo:+.4f}, {hi:+.4f}) |')
text+=['','Negative changes indicate improvement. Final epoch was fixed in advance; validation was not used to select checkpoints.',
    'Epoch 20 within each run is the matched strict-backbone comparator. The earlier pilot retained a disulfide flag.',
    'These are current PypKa teacher-agreement results, not experimental accuracy.',
    '', '![Learning curves](learning_curves.png)','', '![Per-type errors](per_type_errors.png)']
(out/'report.md').write_text('\n'.join(text)+'\n')
print('\n'.join(text),flush=True)
