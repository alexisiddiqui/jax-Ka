"""Pack bounded real native-cache samples and assemble the inactive candidate."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pyarrow.parquet as pq
from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain import production_graphs as pg
from pkatrain import production_loading as pl


def read(path):return json.loads(Path(path).read_text())


def pack(root,dataset,records,arrays,blobs,metadata):
    out=pg.output(root,dataset);out.mkdir(parents=True)
    atomic_json(out/'ids.json',dict(ids=[[r['id'],r['split']] for r in records]))
    pg._write_store(out,dataset,records,metadata['dtypes'],blobs,{},2,fields=metadata['fields'],
        data_kind=metadata.get('data_kind'),zstd_level=metadata['zstd_level'])
    return out/'store-v2'


def run(args):
    require_compute(threads=3)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);root=args.root
    patches=args.patches or root/'native-patches'
    routes=args.routes or root/'native-routes'
    graphs=args.graphs or root/'graphs'
    patch=pg.ProductionStore(patches/'training/gqt-production-v2/pinder/store-v2')
    patch_rows={r['id']:r for r in read(patch.path/'records.json')}
    candidate=args.candidate or root/'candidate-manifests';candidate.mkdir()
    policies=read(routes/'pinder.json')['augmentation_role_policy']
    reports={};controls=[]
    with tarfile.open(root/'native-frame-sample.tar','r') as archive:
        receipt=json.load(archive.extractfile('receipt.json'))
        for dataset in ['pkpdb','pinder']:
            requests=[r for r in receipt['records'] if r['dataset']==dataset]
            metadata=json.load(archive.extractfile(dataset+'/metadata.json'))
            if hashlib.sha256(archive.extractfile(dataset+'/metadata.json').read()).hexdigest()!=digest(root/'native-metadata'/dataset/'metadata.json'):
                raise ValueError('Sample native metadata differs from pinned base')
            old={r['id']:r for r in read(root/'native-metadata'/dataset/'manifest-v1.json')['records']}
            view=read(routes/(dataset+'.json'));view['augmentation_role_policy']=policies
            atomic_json(candidate/(dataset+'.json'),view)
            records=[];data={};encoded=[]
            for request in requests:
                cid=request['id'];blob=archive.extractfile(request['archive_member']).read()
                if hashlib.sha256(blob).hexdigest()!=request['sha256']:raise ValueError('Remote native frame transport mismatch')
                raw=pg.decode(blob,metadata['fields'],metadata['dtypes'],request['dims']);data[cid]=raw;encoded.append(blob)
                records.append(dict(id=cid,split=old[cid]['split'],**dict(zip(pg.DIMS,request['dims'])),
                    arrays_sha256=pg.arrays_digest(raw,metadata['fields'])))
                if dataset=='pinder' and cid in patch.by_id:
                    updated=patch.raw(cid)
                    differences={}
                    for name in pg.GEOMETRY_FIELDS:
                        if name not in updated:continue
                        same=(raw[name].shape==updated[name].shape)
                        if same:
                            same=np.allclose(raw[name],updated[name],atol=pg.EDGE_TOLERANCE,rtol=0) if name in pg.FLOAT_TOLERANT else np.array_equal(raw[name],updated[name])
                        if not same:differences[name]='different'
                    def targets(a):return {(int(r),int(g)):a['targets'][:,i] for i,(r,g) in enumerate(zip(a['query_residue'],a['query_group']))}
                    before,after=targets(raw),targets(updated);common=before.keys()&after.keys()
                    if any(not np.array_equal(before[k],after[k]) for k in common):raise ValueError('Role patch changed a frozen teacher target: '+cid)
                    controls.append(dict(id=cid,common_targets=len(common),geometry_differences=differences))
                    if differences:raise ValueError('Role patch unexpectedly changed native geometry: '+cid+' '+str(differences))
            store=pack(args.output/'base',dataset,records,data,encoded,metadata)
            selected=set(data);smoke=copy.deepcopy(view);smoke['store']=str(store)
            smoke['store_verification_sha256']=digest(store/'verification.json');smoke['legacy_store_metadata_sha256']=digest(store/'metadata.json')
            smoke['records']=[r for r in smoke['records'] if r['id'] in selected]
            if dataset=='pinder':
                subset=[patch_rows[cid] for cid in sorted(selected&patch.by_id.keys())]
                patch_arrays=[patch.raw(r['id']) for r in subset]
                patched=pack(args.output/'patch','pinder',subset,{},[pg.encode(a,pg.FIELDS['pinder'],level=1) for a in patch_arrays],patch.metadata)
                smoke['record_overrides']=[dict(store=str(patched),store_verification_sha256=digest(patched/'verification.json'))]
            atomic_json(args.output/(dataset+'.json'),smoke)
            reports[dataset]=dict(sample_records=len(records),compressed_bytes=pg.read(store/'metadata.json')['compressed_bytes'])
    patch.close()
    sabdab=read(graphs/'training/gqt-production-v2/sabdab/manifest-v1.json')
    prepared={r['source_native_id']:r for r in pq.read_table(root.parent/'sabdab-v7-build-audit-v1/sabdab/units.parquet').to_pylist()}
    for row in sabdab['records']:
        row['residue_side']=[0 if c['side']=='A' else 1 for c in prepared[row['id']]['chains'] for _ in range(len(c['sequence']))]
    sabdab['augmentation_role_policy']=policies;sabdab['release_active']=False
    atomic_json(candidate/'sabdab.json',sabdab)
    index=dict(format='v7-native-training-candidate-v1',release_active=False,masif=False,
        sources=[dict(dataset=d,data_kind='site_only' if d=='pkpdb' else 'paired',manifest=str(candidate/(d+'.json')),
            manifest_sha256=digest(candidate/(d+'.json'))) for d in ['pkpdb','pinder','sabdab']],
        source_precedence=['pkpdb','sabdab','pinder'],pool_summary_sha256=digest(root/'pools-v2/summary.json'),
        ownership_policy_sha256=digest(root/'ownership/policy.json'),native_cache_probe='bounded real frames; full native cache is not copied or rehashed',
        activation_gates=['relocate and verify new stores on target filesystem','freeze independent novel-test registry','explicit production recipe selection'])
    atomic_json(candidate/'candidate.json',index)
    report=dict(status='real_native_cache_smoke_samples_ready',datasets=reports,patch_controls=controls,
        archive_sha256=digest(root/'native-frame-sample.tar'),candidate_index_sha256=digest(candidate/'candidate.json'),release_active=False)
    atomic_json(args.output/'summary.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for n in ['patches','routes','graphs','candidate']:p.add_argument('--'+n,type=Path)
    run(p.parse_args())
