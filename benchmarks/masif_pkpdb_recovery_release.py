"""Validate targeted candidates and write an overlay plan, without promoting it."""
from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np
from pkabench.runtime import atomic_json,digest
from pkatrain.augmentation_store import AugmentationStore,ReleaseWriter
from masif_pkpdb_recovery import POLICY,load_inventory
from masif_annotation_pilot import export_native
from masif_cache_audit import inspect_arrays,extent_metrics
from masif_surface_quality import SURFACE_POLICY


def validate(unit,record,runtime,profile):
    meta=record['metadata'];receipt=meta['receipt'];a=record['arrays']
    if meta.get('id')!=unit['id'] or meta.get('unit_policy')!=POLICY:raise ValueError('record identity/policy mismatch')
    if any(receipt.get(k)!=v for k,v in profile.items()):raise ValueError('scientific profile differs')
    if receipt['status']!='complete' or receipt.get('surface_policy')!=SURFACE_POLICY:raise ValueError('surface preparation failed')
    if receipt.get('inference_backend')!='jax-gpu' or receipt.get('reference_patches_checked')!=3 or receipt.get('descriptor_max_reference_error',1)>1e-4:
        raise ValueError('missing GPU/TF validation')
    coverage=receipt['mesh_coverage']
    if coverage['source_atoms_outside_bbox_plus_2A'] or coverage['raw_surface_max_nearest_A']>3:raise ValueError('raw surface coverage failed')
    hard,review,metrics=inspect_arrays(a)
    if hard:raise ValueError('invalid cached arrays: '+str(hard))
    with tempfile.TemporaryDirectory(prefix='masif-recovery-qa-',dir='/tmp') as directory:
        pdb=Path(directory)/'native.pdb';info=export_native(runtime,unit,pdb,surface_policy=SURFACE_POLICY)
        if info['pdb_sha256']!=meta['source']['pdb_sha256'] or not np.array_equal(a['canonical_residue_ids'],info['canonical_residue_ids']):
            raise ValueError('independent native source/hash/map differs')
        points=np.array([[float(line[k:k+8]) for k in [30,38,46]] for line in pdb.read_text().splitlines() if line.startswith('ATOM')])
        metrics.update(extent_metrics(a['vertices'],points))
        if metrics['atoms_outside_bbox_plus_2A'] or metrics['atoms_outside_convex_hull_plus_2A']:raise ValueError('native surface extent failed')
    return dict(metrics=metrics,review_reasons=review)


def publish(root,names=('chains','antibodies'),destination='candidate-release'):
    r=Path(root);runtime=r.parent.parent/'pkabench';units={};builds={};failures=[];profile=None;qa={}
    for name in names:
        inventory=r/'inventory'/(name+'.json');requested,_=load_inventory(inventory,runtime)
        report=json.loads((r/'build'/name/'report.json').read_text());rows={row['id']:row for row in report['parents']}
        if set(rows)!={u['id'] for u in requested}:raise ValueError('incomplete or extra recovery report')
        scientific={k:report[k] for k in ['checkpoint_sha256','paper_runtime_sha256','preprocessing_profile']}
        if profile is not None and scientific!=profile:raise ValueError('recovery builds have differing scientific profiles')
        profile=scientific
        with AugmentationStore(r/'build'/name/'fingerprints') as store:
            if set(store.by_id)!={rid for rid,row in rows.items() if row['status']=='complete'}:raise ValueError('report/store mismatch')
            for unit in requested:
                rid=unit['id'];units[rid]=unit;builds[rid]=name
                if rows[rid]['status']!='complete':
                    failures.append(dict(id=rid,stage=rows[rid].get('stage'),error=rows[rid].get('error')));continue
                try:qa[rid]=validate(unit,store.raw(rid),runtime,profile)
                except Exception as exc:failures.append(dict(id=rid,stage='independent_QA',error=str(exc)))
    provenance=dict(kind='dense-native-masif-recovery-candidates-v1',unit_policy=POLICY,surface_policy=SURFACE_POLICY,
        publication_status='validated_candidates_not_integrated',**profile,
        input_inventory_sha256={name:digest(r/'inventory'/(name+'.json')) for name in names})
    output=r/destination
    # All candidates have independent native coverage checks before packing.
    with ReleaseWriter(output,provenance,zstd_level=1) as writer:
        for name in names:
            with AugmentationStore(r/'build'/name/'fingerprints') as store:
                for rid in sorted(qa):
                    if builds[rid]!=name:continue
                    record=store.raw(rid);writer.add(rid,record['metadata'],record['arrays'],summary=units[rid])
        writer.finish(expected_ids=sorted(qa))
    with AugmentationStore(output) as store:
        if set(store.by_id)!=set(qa):raise ValueError('packed candidate inventory mismatch')
        for rid in qa:validate(units[rid],store.raw(rid),runtime,profile)
    superseded={rid:units[rid].get('supersedes',[rid]) for rid in qa}
    atomic_json(output/'integration-plan.json',dict(status='not_applied',source_audit=str(r.parent/'masif-pkpdb-cache-audit-v2'),
        replacement_chain_ids={rid:ids for rid,ids in superseded.items()},failures=failures,
        original_cache_changed=False,legacy_preparation_profile_must_remain_explicit=True))
    atomic_json(output/'quality.json',dict(status='validated_candidates_with_holds' if failures else 'validated_candidates',
        requested=len(units),validated=len(qa),failed=len(failures),failures=failures,
        unit_counts=dict(Counter(units[rid]['family'] for rid in qa)),mesh_review_flags=dict(Counter(reason for result in qa.values() for reason in result['review_reasons'])),
        release_manifest_sha256=digest(output/'manifest.json'),checks=['independent native source hashes/maps and convex hull',
            'raw MSMS coverage receipts','finite features and valid triangles','GPU/TF reference','full packed release decode']))
    atomic_json(r/('recovery-status.json' if destination=='candidate-release' else 'resource-recovery-status.json'),json.loads((output/'quality.json').read_text()))
