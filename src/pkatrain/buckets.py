"""Registered shape buckets; sampling probabilities remain group-uniform."""
from collections import defaultdict
import numpy as np
from .records import read
from pkabench.runtime import atomic_json,digest

def build(out):
    manifest=read(out/'manifest.json');groups=defaultdict(list)
    for cid in manifest['train']+manifest['val']:
        layout=read(out/'prepared'/cid/'receipt.json')['layout']
        edge=next((limit for limit in (384,768,1536,3072) if layout['N']<=limit),None)
        if edge is None:edge=((layout['N']+1023)//1024)*1024
        groups[edge].append((cid,layout))
    rows=[];assignment={}
    for edge,items in sorted(groups.items()):
        # Reserve enough padded rows even when a short record shares a large M.
        m=max(l['M'] for _,l in items)
        n=max(l['N'] for _,l in items)+(m+8)//9
        n=((n+63)//64)*64
        ke=max(l['Ke'] for _,l in items);kc=max(l['Kc'] for _,l in items)
        # Conservative envelope calibrated against measured FP32 live allocation.
        # GPU preflight validates the proposed batch size before release.
        estimate=12000*n*kc+73*m*m*32
        size=next((b for b in (8,4,2,1) if b*estimate<=12e9),1)
        name=f'n{edge}';ids=[cid for cid,_ in items]
        train=[cid for cid in ids if cid in manifest['train']]
        for cid in ids:assignment[cid]=name
        rows.append(dict(name=name,capacities=[n,ke,kc,m],batch_size=size,
            estimated_live_bytes=size*estimate,train_count=len(train),val_count=len(ids)-len(train),
            representative_ids=[cid for cid,l in sorted(items,key=lambda x:x[1]['N'],reverse=True) if cid in train][:size]))
    atomic_json(out/'buckets.json',dict(buckets=rows,assignment=assignment,
        manifest_sha256=digest(out/'manifest.json'),policy='Original group-uniform draws, reordered within epoch by randomly ordered size buckets'))
    print(__import__('json').dumps(rows),flush=True)

def ordered_epoch(order,plan,rng):
    groups=defaultdict(list)
    for cid in order:groups[plan['assignment'][cid]].append(cid)
    names=sorted(groups);rng.shuffle(names)
    return [cid for name in names for cid in groups[name]]

def microbatches(ids,plan):
    configs={b['name']:b for b in plan['buckets']};groups=defaultdict(list)
    for cid in ids:groups[plan['assignment'][cid]].append(cid)
    for name,group in groups.items():
        b=configs[name]
        for start in range(0,len(group),b['batch_size']):yield group[start:start+b['batch_size']],b
