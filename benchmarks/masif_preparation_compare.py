"""Paired old/new preparation comparison with unchanged-preparation repeats."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial import cKDTree

from pkabench.runtime import atomic_json,require_compute
from pkatrain.augmentation_store import AugmentationStore,decode


def frame(entry):
    with Path(entry['path']).open('rb') as f:f.seek(entry['offset']);data=f.read(entry['bytes'])
    if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('old frame hash differs')
    return decode(data,max_output_size=entry['raw_bytes'])


def prepare(root):
    root=Path(root);b=root.parent;old={x['id']:x for x in json.loads((root/'full/catalog.json').read_text())['entries']}
    units=json.loads((b/'masif-pkpdb-abag-v1/inventory/inventory.json').read_text())['units'];pairs=[]
    with AugmentationStore(b/'masif-pkpdb-abag-v1/release') as store:
        for u in units:
            if u['split']!='train' or len(u['chain_indices'])!=1 or u['id'] not in store.by_id:continue
            oid='pkpdb/'+u['cid']+'/chain-%04d'%u['chain_indices'][0]
            if oid not in old:continue
            record=store.raw(u['id']);source=record['metadata']['source']
            pairs.append(dict(old_id=oid,new_id=u['id'],n=u['n'],role=u['role'],old_entry=old[oid],
                added_OXT_canonical_ids=source['added_terminal_oxygen_canonical_ids']))
    selected=[];rng=np.random.default_rng(20261120)
    for lo,hi in [(0,150),(150,300),(300,600),(600,10**9)]:
        rows=sorted([x for x in pairs if lo<x['n']<=hi],key=lambda x:x['old_id'])
        selected.extend(rows[int(i)] for i in rng.choice(len(rows),min(4,len(rows)),replace=False))
    atomic_json(root/'preparation-pairs.json',dict(candidates=pairs,repeat_controls=selected))
    return selected


def compare_arrays(a,b,terminal_centers):
    if not np.array_equal(a['canonical_residue_ids'],b['canonical_residue_ids']):raise ValueError('canonical unit differs')
    distance,nearest=cKDTree(b['vertices']).query(a['vertices'])
    same=a['canonical_residue_ids'][a['owner']]==b['canonical_residue_ids'][b['owner'][nearest]]
    valid=(distance<=.5)&same
    x=a['descriptors'];y=b['descriptors'][nearest];norm=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
    valid&=norm>1e-12
    cosine=np.zeros(len(x));cosine[valid]=np.sum(x[valid]*y[valid],axis=1)/norm[valid]
    terminal_distance=(cKDTree(terminal_centers).query(a['vertices'])[0] if len(terminal_centers) else np.full(len(x),np.inf))
    def stats(mask):
        values=cosine[valid&mask]
        return dict(matched=int(len(values)),median_cosine=float(np.median(values)) if len(values) else None,
                    p05_cosine=float(np.percentile(values,5)) if len(values) else None)
    return dict(vertices_a=len(x),vertices_b=len(b['vertices']),matched_fraction=float(valid.mean()),
        surface_distance_mean_A=float(distance.mean()),surface_distance_p95_A=float(np.percentile(distance,95)),
        all=stats(np.ones(len(x),bool)),near_changed_termini_12A=stats(terminal_distance<=12),
        far_from_changed_termini_16A=stats(terminal_distance>16),
        raw_feature_MAE=np.mean(np.abs(a['raw_features'][valid]-b['raw_features'][nearest][valid]),axis=0).tolist() if valid.any() else None)


def compare(root,runtime):
    root=Path(root);b=root.parent;data=json.loads((root/'preparation-pairs.json').read_text());rows=[]
    from pkatrain.production_sidechains import _pkpdb_atoms
    repeat_root=root/'old-policy-repeat/fingerprints'
    with AugmentationStore(b/'masif-pkpdb-abag-v1/release') as new,AugmentationStore(repeat_root) as repeat:
        for pair in data['candidates']:
            a=frame(pair['old_entry']);bb,_,_=_pkpdb_atoms(Path(runtime),pair['old_entry']['summary']['cid'])
            centers=bb[pair['added_OXT_canonical_ids'],2] if pair['added_OXT_canonical_ids'] else np.empty((0,3))
            row=dict(old_id=pair['old_id'],new_id=pair['new_id'],n=pair['n'],added_terminal_oxygens=len(centers),
                old_to_current=compare_arrays(a['arrays'],new.raw(pair['new_id'])['arrays'],centers))
            if pair['old_id'] in repeat.by_id:row['old_to_old_repeat']=compare_arrays(a['arrays'],repeat.raw(pair['old_id'])['arrays'],centers)
            rows.append(row)
    atomic_json(root/'preparation-comparison.json',dict(cases=rows,
        comparison='same native chain/context, cached old vs current; old-policy repeats estimate remeshing variability',
        limitations=['Matched nearest vertices within 0.5 A and the same canonical residue; unmatched areas are separately counted',
                    'Euclidean terminal neighbourhoods approximate the 12 A geodesic patch locality',
                    'Results do not certify equality of every legacy record to the current preparation policy']))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['prepare','compare']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--runtime',type=Path);a=p.parse_args();require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True)
    if a.command=='prepare':prepare(a.root)
    else:compare(a.root,a.runtime)
