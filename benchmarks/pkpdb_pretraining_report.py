"""Common-site, group-bootstrap comparison of the four pretraining arms."""
import csv
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from pkabench.runtime import atomic_json, digest, require_compute
from pkabench.frozen_score import measures, aggregate, write_csv


def read_predictions(path):
    with path.open() as stream: rows=list(csv.DictReader(stream))
    result={}
    for row in rows:
        key=tuple(row[k] for k in ('complex_id','chain','resnum','icode','group'))
        assert key not in result
        result[key]=dict(row,teacher_pka=float(row['teacher_pka']),predicted_pka=float(row['predicted_pka']))
    return result


def summary(data, keys):
    grouped=defaultdict(list)
    for k in keys:grouped[data[k]['complex_id']].append(data[k])
    values=[]
    for cid,rows in grouped.items():
        values.append(dict(complex_id=cid,component_id=rows[0]['component_id'],n=len(rows),
            **measures([r['teacher_pka'] for r in rows],[r['predicted_pka'] for r in rows])))
    return aggregate(values,replicates=2000)[0],values


def report(out):
    paths={'gqt-raw':out/'gqt-batched-raw/seed-17','gqt-clean':out/'gqt-batched-clean/seed-17',
           'pkai-raw':out/'pkai-raw','pkai-clean':out/'pkai-clean'}
    data={}
    for arm,path in paths.items():
        assert json.loads((path/'verification.json').read_text())['passed']
        data[arm]=read_predictions(path/'validation_predictions.csv')
    common=set.intersection(*(set(d) for d in data.values()));assert common
    for k in common:
        targets=[d[k]['teacher_pka'] for d in data.values()]
        assert max(targets)-min(targets)<1e-5
        assert len({d[k]['component_id'] for d in data.values()})==1
    rows=[];values={}
    for arm,d in data.items():
        for support,keys in (('all_supported',set(d)),('common',common)):
            scores,percomplex=summary(d,keys)
            rows.append(dict(model=arm,support=support,sites=len(keys),mae=scores['mae'],
                ci_low=scores['mae_ci95'][0],ci_high=scores['mae_ci95'][1],rmse=scores['rmse']))
            if support=='common':values[arm]=percomplex
    paired=[]
    for model in ('gqt','pkai'):
        byarm={}
        for arm in ('raw','clean'):
            groups=defaultdict(list)
            for row in values[f'{model}-{arm}']:groups[row['component_id']].append(row['mae'])
            byarm[arm]={g:float(np.mean(v)) for g,v in groups.items()}
        groups=sorted(byarm['raw']);assert groups==sorted(byarm['clean'])
        delta=np.array([byarm['clean'][g]-byarm['raw'][g] for g in groups])
        rng=np.random.default_rng(20261006)
        boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
        paired.append(dict(model=model,clean_minus_raw_mae=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),groups=len(groups)))
    write_csv(out/'comparison.csv',rows)
    atomic_json(out/'comparison.json',dict(rows=rows,paired=paired,common_sites=len(common),
        predictions_sha256={a:digest(p/'validation_predictions.csv') for a,p in paths.items()}))
    lines=['# 5k pKPDB pretraining comparison','',
        'Seed 17; identical 5,000-structure source cohort and clean validation set. No test evaluation.',
        'GQT uses backbone inputs and a fixed 20th epoch. pKAI uses native side-chain features and a validation-selected checkpoint; its score is a development result.',
        'GQT preserves its first serial-accumulation epoch, then resumes with true eight-structure batches and strict float32 matrix multiplication.',
        '', '| Model / labels | Matched sites | Group-macro MAE | 95% group-bootstrap CI |', '|---|---:|---:|---|']
    for r in rows:
        if r['support']=='common':lines.append(f"| {r['model']} | {r['sites']:,} | {r['mae']:.4f} | {r['ci_low']:.4f}–{r['ci_high']:.4f} |")
    lines += ['', '| Cleaning effect | Clean − raw MAE | 95% paired group-bootstrap CI |', '|---|---:|---|']
    for r in paired:lines.append(f"| {r['model']} | {r['clean_minus_raw_mae']:+.4f} | {r['ci95'][0]:+.4f}–{r['ci95'][1]:+.4f} |")
    lines += ['','Negative cleaning effects favor cleaned labels. Native-support scores are retained in comparison.csv.',
        'Capacity, input information, training recipe, and checkpoint selection differ between models. Historical training and current validation teacher settings also differ.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    require_compute(threads=2)
    report(Path(os.environ['PKABENCH_RUNTIME'])/'pretraining/pkpdb-5k-comparison-v1')
