"""Check legacy fields for failed reference contracts and silent zero padding."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from pkabench.runtime import atomic_json,require_compute,digest
from pkatrain.augmentation_store import decode


def inspect_shard(task):
    path,entries=task;rows=[]
    with Path(path).open('rb') as stream:
        for entry in sorted(entries,key=lambda x:x['offset']):
            row=dict(id=entry['id'],status='audited',hard_reasons=[],review_reasons=[])
            try:
                stream.seek(entry['offset']);frame=stream.read(entry['bytes'])
                if hashlib.sha256(frame).hexdigest()!=entry['sha256']:raise ValueError('compressed frame hash mismatch')
                record=decode(frame,max_output_size=entry['raw_bytes']);a=record['arrays'];receipt=record['metadata']['receipt']
                if receipt['checkpoint_sha256']!=entry['checkpoint_sha256']:row['hard_reasons'].append('checkpoint_mismatch')
                if receipt.get('inference_backend')=='jax-gpu' and (receipt.get('reference_patches_checked')!=3 or receipt.get('descriptor_max_reference_error',1)>1e-4):
                    row['hard_reasons'].append('GPU_reference_contract')
                e=a['raw_features'][:,2];zero=e==0;nonzero=np.flatnonzero(~zero)
                tail=len(e)-int(nonzero[-1])-1 if len(nonzero) else len(e)
                row.update(vertices=len(e),electrostatic_zero_vertices=int(zero.sum()),electrostatic_zero_tail=tail,
                           descriptor_zero_rows=int(np.all(a['descriptors']==0,axis=1).sum()))
                if zero.all():row['review_reasons'].append('all_zero_electrostatics')
                elif tail>=max(32,int(np.ceil(.01*len(e)))):row['review_reasons'].append('electrostatic_zero_tail')
                if row['descriptor_zero_rows']:row['review_reasons'].append('zero_descriptor_rows')
            except Exception as exc:row.update(status='audit_error',error=str(exc))
            rows.append(row)
    return rows


def run(root,workers):
    root=Path(root);data=json.loads((root/'full/catalog.json').read_text());groups=defaultdict(list)
    for entry in data['entries']:groups[entry['path']].append(entry)
    rows=[];began=time.monotonic()
    with ProcessPoolExecutor(workers) as pool:
        for batch in pool.map(inspect_shard,sorted(groups.items())):
            rows.extend(batch);atomic_json(root/'field-progress.json',dict(audited=len(rows),total=len(data['entries']),wall_s=time.monotonic()-began))
    if len(rows)!=len(data['entries']) or len({r['id'] for r in rows})!=len(rows):raise ValueError('field audit coverage mismatch')
    atomic_json(root/'field-audit.json',dict(status='complete',records=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
        hard_reasons=dict(Counter(x for r in rows for x in r['hard_reasons'])),
        review_reasons=dict(Counter(x for r in rows for x in r['review_reasons'])),
        flagged=[r for r in rows if r['hard_reasons'] or r['review_reasons'] or r['status']!='audited'],
        wall_s=time.monotonic()-began,catalog_sha256=digest(root/'full/catalog.json'),
        limitations=['Exact zero tails/all-zero fields are suspects for old unchecked-wrapper zero padding; replay is needed before declaring them corrupted',
                     'No scientific fields or payloads are modified by this check']))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True);run(a.root,a.workers)
