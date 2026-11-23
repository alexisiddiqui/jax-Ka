"""Versioned native cache views and bounded remote-frame smoke requests."""
import argparse
from collections import Counter,defaultdict
import copy
import json
from pathlib import Path

import pyarrow.parquet as pq
from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.loading import BucketPolicy
from pkatrain.production_graphs import DIMS


def read(path):return json.loads(Path(path).read_text())


def run(args):
    require_compute(threads=2)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    members=pq.read_table(args.pools/'membership.parquet').to_pylist()
    patch=read(args.patches/'summary.json');patch_rows={r['id']:r for r in read(Path(patch['patch_store'])/'records.json')}
    native={r['unit_id']:r for r in pq.read_table(args.audit/'existing-native/units.parquet').to_pylist()}
    requests=[];reports={}
    for dataset in ['pkpdb','pinder']:
        old=read(args.metadata/dataset/'manifest-v1.json');old_rows={r['id']:r for r in old['records']}
        selected=[r for r in members if r['source']==dataset];records=[]
        for member in selected:
            cid=member['id'];record=copy.deepcopy(old_rows[cid])
            record.update(split='train' if member['operational_role']=='train' else 'val',
                source=dataset,source_native_id=cid,native_source_split=member['native_source_split'],
                native_component=member['native_component'],operational_role=member['operational_role'],
                input_scope_id=native[member['unit_id']]['input_scope_id'],component_id=member['group'],
                operational_group=member['group'],native_coordinate_sha256=native[member['unit_id']]['prepared_student_context_sha256'])
            if dataset=='pinder':
                context=native[member['unit_id']];sides={r['chain']:r['side'] for r in context['side_by_chain']}
                record['residue_side']=[0 if sides[c['chain_id']]=='A' else 1 for c in context['chains'] for _ in range(c['residues'])]
            if member['operational_role']=='train':record['min_fraction']=member['min_fraction']
            else:record.pop('min_fraction',None)
            if dataset=='pinder' and cid in patch_rows:
                for field in [*DIMS,'w_burial_mean','w_interface_mean']:record[field]=patch_rows[cid][field]
            records.append(record)
        policy=BucketPolicy.from_json(old['bucket_policy'])
        result=dict(version=old['version'],dataset=dataset,data_kind='paired' if dataset=='pinder' else 'site_only',
            source_manifest_schema=1,store=old['store'],store_verification_sha256=old['store_verification_sha256'],
            legacy_store_metadata_sha256=digest(args.metadata/dataset/'metadata.json'),
            base_manifest_sha256=digest(args.metadata/dataset/'manifest-v1.json'),
            pool=str(args.pools/(dataset+'.tsv')),pool_sha256=digest(args.pools/(dataset+'.tsv')),
            bucket_policy=old['bucket_policy'],capacities=policy.capacities(records),records=records,
            augmentation_role_policy=dict(schema_version=1,records={},unit_roles={r['unit_id']:r['operational_role'] for r in members}),
            release_active=False,test_data_included=False)
        if dataset=='pinder':
            result['record_overrides']=[dict(store=patch['patch_store'],store_verification_sha256=digest(Path(patch['patch_store'])/'verification.json'))]
        atomic_json(args.output/(dataset+'.json'),result)
        # Bounded real-cache verification covers every occupied size/role cell
        # and both routing paths, without transferring the full native payload.
        cells=defaultdict(list)
        for row in records:
            cells[row['split'],policy.bucket(row['n']),dataset=='pinder' and row['id'] in patch_rows].append(row)
        ids=set()
        for group in cells.values():ids.add(max(group,key=lambda r:(r['n'],r['id']))['id'])
        requests.append(dict(dataset=dataset,store=old['store'],ids=sorted(ids),
            metadata_sha256=result['legacy_store_metadata_sha256'],verification_sha256=old['store_verification_sha256']))
        reports[dataset]=dict(records=len(records),counts=dict(Counter(r['split'] for r in records)),
            patched_records=sum(r['id'] in patch_rows for r in records) if dataset=='pinder' else 0,
            real_cache_probe_records=len(ids),manifest_sha256=digest(args.output/(dataset+'.json')))
    atomic_json(args.output/'remote-frame-requests.json',requests)
    atomic_json(args.output/'summary.json',dict(status='inactive_native_cache_routes_complete',datasets=reports,
        untouched_native_payloads_reused=True,release_active=False,augmentation_unassessed_donors='native geometry fallback; never silently used'))
    print(json.dumps(reports,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['audit','pools','patches','metadata','output']:p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
