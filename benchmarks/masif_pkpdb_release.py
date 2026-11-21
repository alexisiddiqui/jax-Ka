"""Validate grouped pKPDB fingerprints and compact them for training transfer."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.augmentation_store import AugmentationStore,ReleaseWriter
from masif_pkpdb_abag import UNIT_POLICY,load_inventory
from masif_surface_quality import SURFACE_POLICY


def validate_record(unit,record,*,expected_profile=None):
    meta=record['metadata'];a=record['arrays'];receipt=meta['receipt']
    if meta.get('unit_policy')!=UNIT_POLICY or meta.get('role')!=unit['role']:
        raise ValueError('record partner policy/role mismatch')
    if meta['chain_indices']!=unit['chain_indices'] or meta['source_files_sha256']!=unit['source_files_sha256']:
        raise ValueError('record partner selection/source mismatch')
    if receipt['surface_policy']!=SURFACE_POLICY or receipt['status']!='complete':
        raise ValueError('unvalidated preparation policy')
    if expected_profile and any(receipt.get(k)!=value for k,value in expected_profile.items()):
        raise ValueError('record model/runtime profile mismatch')
    if receipt['descriptor_max_reference_error']>1e-4 or receipt['reference_patches_checked']!=3:
        raise ValueError('GPU/reference validation missing or failed')
    if receipt['mesh_coverage']['source_atoms_outside_bbox_plus_2A'] or receipt['mesh_coverage']['raw_surface_max_nearest_A']>3.:
        raise ValueError('surface coverage validation failed')
    if a['canonical_residue_ids'].tolist()!=unit['canonical_residue_ids']:
        raise ValueError('canonical residue mapping differs from inventory')
    n=len(a['vertices']);faces=a['faces'];owner=a['owner']
    if a['vertices'].shape!=(n,3) or a['descriptors'].shape!=(n,80) or a['normals'].shape!=(n,3):
        raise ValueError('surface/descriptor dimensions differ')
    if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.min()<0 or faces.max()>=n:
        raise ValueError('invalid surface triangles')
    if ((faces[:,0]==faces[:,1])|(faces[:,0]==faces[:,2])|(faces[:,1]==faces[:,2])).any():
        raise ValueError('collapsed surface triangle')
    tri=a['vertices'][faces]
    if (np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)<=1e-10).any():
        raise ValueError('zero-area surface triangle')
    if owner.shape!=(n,) or owner.min()<0 or owner.max()>=len(unit['canonical_residue_ids']):
        raise ValueError('invalid surface owner mapping')
    if not all(np.isfinite(x).all() for x in a.values()):raise ValueError('nonfinite packed surface arrays')


def publish(runtime,inventory,build,output,*,quarantine=None):
    build=Path(build);output=Path(output)
    units,source=load_inventory(inventory,runtime)
    report=json.loads((build/'report.json').read_text())
    failures={r['id']:r for r in report['parents'] if r['status']!='complete'}
    allowed={};quarantine_sha=None
    if quarantine:
        data=json.loads(Path(quarantine).read_text());quarantine_sha=digest(quarantine)
        if data['inventory_sha256']!=source['inventory_sha256']:raise ValueError('quarantine inventory differs')
        allowed={r['id']:r['reason'] for r in data['failures']}
    if (set(failures)!=set(allowed) or report['status'] not in ('complete','partial')
            or not failures and report['status']!='complete'):
        raise ValueError('full pKPDB Ab/Ag build has unresolved failures; publication held')
    if {r['id'] for r in report['parents']}!={u['id'] for u in units}:
        raise ValueError('full build and inventory do not contain exactly the same units')
    requested=len(units);held_pairs={key.rsplit('/',1)[0] for key in failures}
    held=[dict(id=u['id'],cid=u['cid'],role=u['role'],split=u['split'],
        reason=allowed[u['id']] if u['id'] in allowed else 'paired partner quarantined',
        stage=failures[u['id']]['stage'] if u['id'] in failures else 'pair_publication',
        error=failures[u['id']].get('error') if u['id'] in failures else None)
        for u in units if u['id'].rsplit('/',1)[0] in held_pairs]
    units=[u for u in units if u['id'].rsplit('/',1)[0] not in held_pairs]
    if not units:raise ValueError('quarantine would leave an empty release')
    pairs={}
    for unit in units:pairs.setdefault(unit['id'].rsplit('/',1)[0],{})[unit['role']]=unit
    if any(set(pair)!={'Ab','Ag'} for pair in pairs.values()):raise ValueError('unpaired antibody/antigen inventory')
    provenance=dict(kind='dense-native-masif-paper-v1',unit_policy=UNIT_POLICY,surface_policy=SURFACE_POLICY,
        source_inventory_sha256=source['inventory_sha256'],source_build_manifest_sha256=digest(build/'fingerprints/manifest.json'),
        checkpoint_sha256=report['checkpoint_sha256'],preprocessing_profile=report['preprocessing_profile'],
        paper_runtime_sha256=report['paper_runtime_sha256'],publisher_sha256=digest(__file__))
    provenance.update(quarantine_file_sha256=quarantine_sha,requested_units=requested,
        published_units=len(units),publication_policy='complete-pairs-explicit-quarantine-v1')
    expected_profile={k:report[k] for k in ['checkpoint_sha256','preprocessing_profile','paper_runtime_sha256']}
    # Validate the entire input before creating a publication destination.
    with AugmentationStore(build/'fingerprints') as store:
        if set(store.by_id)!={r['id'] for r in report['parents'] if r['status']=='complete'}:
            raise ValueError('packed input contains missing/extra completed units')
        for unit in units:validate_record(unit,store.raw(unit['id']),expected_profile=expected_profile)
        with ReleaseWriter(output,provenance,zstd_level=1) as writer:
            for unit in sorted(units,key=lambda x:x['id']):
                record=store.raw(unit['id'])
                writer.add(unit['id'],record['metadata'],record['arrays'],summary=unit)
            writer.finish(expected_ids=[u['id'] for u in units])
    # Decode the actual transfer release through the normal training reader.
    with AugmentationStore(output) as store:
        for unit in units:validate_record(unit,store.raw(unit['id']),expected_profile=expected_profile)
    pair_rows=[dict(id=key,cid=p['Ab']['cid'],split=p['Ab']['split'],group=p['Ab']['group'],
        Ab=p['Ab']['id'],Ag=p['Ag']['id'],Ab_chains=p['Ab']['source_chain_labels'],Ag_chains=p['Ag']['source_chain_labels'])
        for key,p in sorted(pairs.items())]
    atomic_json(output/'partner-pairs.json',dict(unit_policy=UNIT_POLICY,pairs=pair_rows))
    atomic_json(output/'quarantine.json',dict(source_inventory_sha256=source['inventory_sha256'],units=held))
    atomic_json(output/'quality.json',dict(status='complete_with_quarantine' if held else 'complete',
        requested_units=requested,validated_units=len(units),quarantined_units=len(held),pairs=len(pair_rows),
        unit_counts=dict(Counter(u['split']+'/'+u['role'] for u in units)),
        inventory_sha256=source['inventory_sha256'],source_build_manifest_sha256=provenance['source_build_manifest_sha256'],
        release_manifest_sha256=digest(output/'manifest.json'),partner_pairs_sha256=digest(output/'partner-pairs.json'),
        checks=['native source/partner mapping','surface coverage','finite arrays and 80D descriptors',
                'triangle indices','GPU vs TensorFlow','full transfer-release decode']))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--quarantine',type=Path,help='explicit inventory-pinned failure IDs; affected pairs are withheld together')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True)
    publish(a.runtime,a.inventory,a.build,a.output,quarantine=a.quarantine)
