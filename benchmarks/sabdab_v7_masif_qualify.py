"""Qualify native SAbDab surfaces and exercise the assembly MaSIF reader."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.sabdab_units import surface_inventory
from pkatrain.augmentation_store import AugmentationStore,FORMAT,verify_records
from pkatrain.masif_store import MaSIFStore
from masif_cache_audit import inspect_arrays


def run(args):
    require_compute(threads=4)
    if args.output.exists():raise FileExistsError(args.output)
    units,inventory=surface_inventory(args.inventory);lookup={u['id']:u for u in units}
    report=json.loads((args.build/'report.json').read_text())
    if report['status']!='complete' or report['counts'].get('failed',0):raise ValueError('MaSIF build is incomplete')
    checks=[];began=time.monotonic();rows=[];position=0
    args.output.mkdir(parents=True)
    with AugmentationStore(args.build/'fingerprints') as store, (args.output/'records-00000.bin').open('wb') as stream:
        if set(store.by_id)!=set(lookup):raise ValueError('MaSIF build does not cover selected native AB/A/B inventory')
        for rid,unit in sorted(lookup.items()):
            record=store.raw(rid);arrays=record['arrays'];metadata=record['metadata']
            if any(metadata.get(k)!=unit[k] for k in ['dataset','cid','split','state']):raise ValueError('Native feature source identity differs')
            if arrays['canonical_residue_ids'].tolist()!=unit['canonical_residue_ids']:raise ValueError('Native feature canonical map differs')
            if metadata['prepared_unit']!=unit['prepared_unit']:raise ValueError('Native feature input provenance differs')
            receipt=metadata['receipt']
            if receipt.get('descriptor_max_reference_error',1)>1e-4 or receipt.get('reference_patches_checked',0)!=3:
                raise ValueError('Native MaSIF GPU/reference check is absent or failed')
            hard,review,metrics=inspect_arrays(arrays)
            if hard:raise ValueError('Invalid native MaSIF arrays '+rid+': '+repr(hard))
            norm=np.linalg.norm(arrays['descriptors'],axis=1)
            if not (norm>1e-6).all():raise ValueError('Empty native MaSIF descriptors')
            checks.append(dict(id=rid,review_flags=review,metrics=metrics,
                descriptor_norm_min=float(norm.min()),descriptor_norm_max=float(norm.max()),
                reference_error=receipt['descriptor_max_reference_error']))
            row=store.by_id[rid]
            with (store.path/row['shard']).open('rb') as source:
                source.seek(row['offset']);blob=source.read(row['bytes'])
            if hashlib.sha256(blob).hexdigest()!=row['sha256']:raise ValueError('Native surface frame changed')
            stream.write(blob);rows.append(dict(row,shard='records-00000.bin',offset=position));position+=len(blob)
    parents=[]
    for unit in json.loads(args.inventory.read_text())['units']:
        cid=unit['unit_id'].split(':',1)[1]
        parents.append(dict(dataset='sabdab',cid=cid,split={'train':'train','validation':'val'}[unit['split']],
            group=unit['component'],status='covered',reasons=[],canonical_residues=unit['n'],
            required_units=[dict(id='sabdab/'+cid+'/'+state,state=state,role=role) for state,role in [('A','Ab'),('B','Ag')]],
            bound_unit='sabdab/'+cid+'/AB'))
    atomic_json(args.output/'index.json',dict(format=FORMAT,records=rows))
    verification=verify_records(args.output,rows);atomic_json(args.output/'verification.json',verification)
    atomic_json(args.output/'coverage.json',dict(format='masif-native-assembly-coverage-v1',parents=parents,
        source_inventory_sha256=digest(args.inventory),pilot_only=True))
    files={name:digest(args.output/name) for name in ['records-00000.bin','index.json','verification.json','coverage.json']}
    manifest=dict(format=FORMAT,verified=True,zstd_level=1,records=len(rows),shard_bytes=2**30,
        compressed_bytes=position,raw_bytes=sum(r['raw_bytes'] for r in rows),files=files,
        provenance=dict(kind='sabdab-native-masif-pilot-v1',pilot_only=True,
            source_build_manifest_sha256=digest(args.build/'fingerprints/manifest.json'),
            source_inventory_sha256=digest(args.inventory),preprocessing_profile=report['preprocessing_profile'],
            checkpoint_sha256=report['checkpoint_sha256'],paper_runtime_sha256=report['paper_runtime_sha256']))
    atomic_json(args.output/'manifest.json',manifest)
    t=time.monotonic();loaded=0
    with MaSIFStore(args.output) as store:
        for parent in parents:
            native=store.load_assembly('sabdab',parent['cid'],split=parent['split'])
            if {u['role'] for u in native['units']}!={'Ab','Ag'}:raise ValueError('Incomplete native biological sides')
            bound=store.store.raw(parent['bound_unit'])
            if bound['arrays']['canonical_residue_ids'].tolist()!=list(range(parent['canonical_residues'])):
                raise ValueError('Bound surface does not map to the same native canonical assembly')
            loaded+=1
    atomic_json(args.report,dict(status='native_surfaces_and_assembly_reader_verified',complexes_loaded=loaded,
        surface_records=len(rows),files=len(list(args.output.iterdir())),zstd_level=1,packed_bytes=position,
        raw_bytes=manifest['raw_bytes'],reader_s=time.monotonic()-t,wall_s=time.monotonic()-began,
        checks=checks,active_release_changed=False,model_training_performed=False,release_ready=False))
    print(json.dumps(dict(complexes_loaded=loaded,surface_records=len(rows),files=len(list(args.output.iterdir())),packed_bytes=position)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inventory','build','output','report']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
