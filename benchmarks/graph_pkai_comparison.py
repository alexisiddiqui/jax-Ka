"""Score frozen pKAI and graph pilots on identical validation AB site support."""
import csv
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from pkabench.runtime import require_compute,atomic_json,digest
from pkabench.schema import KEY,key,NULL_PKA
from pkabench.frozen_score import measures,aggregate

require_compute()
root=Path(os.environ['PKABENCH_RUNTIME']);base=root/'pretraining'
paths={name:base/folder/'seed-17/validation_predictions.csv' for name,folder in
    [('GQT 10k','graph-pilot-10k-v1'),('GQT 29k','graph-pilot-v1'),('GQT 50k','graph-pilot-50k-v1')]}
for name,folder in [('GQT side chains 50k','graph-sidechains-50k-v1'),('GQT matched backbone control','graph-sidechains-control-50k-v1')]:
    path=base/folder/'seed-17/validation_predictions.csv'
    if path.exists():paths[name]=path
models={}
for name,path in paths.items():
    with path.open() as stream:rows=list(csv.DictReader(stream))
    for r in rows:r['resnum']=int(r['resnum'])
    models[name]={key(r):r for r in rows}
reference=models['GQT 29k'];assert all(set(rows)==set(reference) for rows in models.values())
ids=sorted({k[0] for k in reference})
path=root/'campaigns/production-nojax-v1/predictions.parquet'
pred=pq.read_table(path,columns=list(KEY)+['pka','status'],filters=[('method','=','pkai'),('state','=','AB'),('complex_id','in',ids)]).to_pylist()
pkai={key(r):r['pka'] for r in pred if r['status']=='ok' and r['pka'] is not None and np.isfinite(r['pka'])}
support=set(reference)&set(pkai);assert support
scores={};natives=[]
for name in ['pKAI',*models,'Train type mean']:
    groups=defaultdict(list)
    for k in sorted(support):
        row=reference[k];ref=float(row['teacher_pka']);baseline=NULL_PKA[k[-1]]
        value=pkai[k] if name=='pKAI' else float(row['train_type_mean']) if name=='Train type mean' else float(models[name][k]['predicted_pka'])
        groups[k[0]].append((ref-baseline,value-baseline,row['component_id']))
    complexes=[]
    for cid,rr in groups.items():
        a,b,_=zip(*rr);complexes.append(dict(complex_id=cid,component_id=rr[0][2],n=len(rr),**measures(a,b)))
    scores[name]=aggregate(complexes)[0]
report=dict(target='Single-state AB scalar pKa, current PypKa teacher',eligible_sites=len(reference),matched_sites=len(support),
    missing_pkai=len(reference)-len(support),scores=scores,sources={str(p):digest(p) for p in [path,*paths.values()]})
atomic_json(base/'graph-pkai-comparison.json',report)
lines=['# Matched validation pKa comparison','',f'{len(support)} matched sites of {len(reference)} eligible; {len(reference)-len(support)} lack valid pKAI predictions.',
    '', '| Model | Group-macro MAE | 95% CI |','|---|---:|---|']
for name,s in scores.items():
    lo,hi=s['mae_ci95'];lines.append(f'| {name} | {s["mae"]:.4f} | {lo:.4f}–{hi:.4f} |')
lines+=['','Final epoch for all GQTs. Frozen pKAI, no fitting. Same validation sites and group weighting; 2,000 component-bootstrap replicates.',
    'pKAI was pretrained on a much larger historical PypKa dataset; historical pretraining overlap is not ruled out. These are teacher-agreement scores, not experimental accuracy.']
(base/'graph-pkai-comparison.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines),flush=True)
