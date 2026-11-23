"""Pack and load real native pKAI pilot records; never train a model."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.sabdab_units import load_prepared
from pkatrain import production_graphs as pg
from pkatrain import production_loading as pl
from pkatrain.loading import BucketPolicy,LoaderConfig
from pkatrain.native_paired_graph import build


def run(args):
    require_compute(threads=4)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);report=json.loads((args.labels/'summary.json').read_text())
    arrays=[];records=[];provenance={};skipped={};details=[];t=time.monotonic()
    for receipt in report['records']:
        unit=receipt['unit'];cid=unit['unit_id'].split(':',1)[1];entry=args.labels/'entries'/cid
        if receipt['status']!='complete':
            skipped[cid]=dict(id=cid,reason='native label preparation failed');continue
        for name,expected in receipt['files_sha256'].items():
            if digest(entry/name)!=expected:raise ValueError('Native label input changed: '+str(entry/name))
        topology,meta=load_prepared(unit);sites=json.loads((entry/'sites.json').read_text());labels=json.loads((entry/'labels.json').read_text())
        real=json.loads((entry/'terminal-evidence.json').read_text())['real']
        split={'train':'train','validation':'val'}[unit['split']]
        graph,detail=build(topology,meta['side_by_chain'],sites,labels,real,split)
        details.append(dict(id=cid,split=split,**detail))
        if graph is None:
            skipped[cid]=dict(id=cid,split=split,**detail);continue
        n=topology.n_residues;k=graph['neighbors'].shape[1];q=len(graph['query_residue'])
        s=len(graph['site_residue']);sk=graph['site_neighbors'].shape[1]
        records.append(dict(id=cid,split=split,n=n,k=k,q=q,s=s,sk=sk,arrays_sha256=pg.arrays_digest(graph,pg.FIELDS['pinder'])))
        arrays.append(graph)
        provenance[cid]=dict(source='sabdab',source_native_id=cid,native_source_split=unit['native_source_split'],
            native_component=unit['component'],operational_role=unit['split'],input_scope_id=meta['policy'],
            preparation_sha256=meta['preparation_sha256'],teacher_receipt_sha256=digest(entry/'receipt.json'),
            mask_policy=sites[0]['mask_policy'],
            supervision_scope_policy=receipt.get('supervision_scope_policy','deposited-full-sequence-v1'),pilot_only=True)
    result=dict(status='no_qualified_native_records' if not records else 'packed_native_loader_verified',
        source='sabdab',native_records=len(records),skipped=list(skipped.values()),graph_checks=details,
        active_release_changed=False,release_ready=False,model_training_performed=False)
    if records:
        out=pg.output(args.output,'sabdab');out.mkdir(parents=True)
        atomic_json(out/'ids.json',dict(ids=[[r['id'],r['split']] for r in records]))
        fields=pg.FIELDS['pinder'];dtypes={f:arrays[0][f].dtype.str for f in fields}
        if any(any(a[f].dtype.str!=dtypes[f] for f in fields) for a in arrays):raise ValueError('Graph dtypes differ')
        pg._write_store(out,'sabdab',records,dtypes,(pg.encode(a,fields,level=1) for a in arrays),skipped,4,
            fields=fields,data_kind='paired',zstd_level=1)
        # This full-inclusion smoke pool is explicitly pilot-only. It does not
        # substitute for regenerating operational groups and nested pool ranks.
        pool=args.output/'pilot-only-pool.tsv'
        pool.write_text('id\tmin_fraction\n'+''.join(r['id']+'\t1.0\n' for r in records if r['split']=='train'))
        policy=BucketPolicy((256,512,768),(2,2,1))
        manifest=pl.build_manifest(args.output,'sabdab',policy,4,pool_path=pool,data_kind='paired',record_provenance=provenance)
        train=pl.select(manifest,'train');val=pl.select(manifest,'val')
        if train:source=pl.source_from_manifest(manifest,config=LoaderConfig(workers=2))
        else:source=pl.source_from_manifest(manifest,config=LoaderConfig(workers=2),norms={'burial':1.,'interface':1.})
        before=time.monotonic();loaded=0;by_id={r['id']:a for r,a in zip(records,arrays)}
        try:
            for row in records:
                batch=source.load([row['id']]);raw=by_id[row['id']];q=row['q'];n=row['n']
                if not np.array_equal(batch[1][0,:,:q],raw['targets']):raise ValueError('Loaded native labels differ')
                if not batch[2][0,:q].all() or batch[2][0,q:].any() or batch[2][1:].any():raise ValueError('Loaded native masks/padding differ')
                if not np.array_equal(batch[0]['nodes'][0,0,:n],raw['nodes']):raise ValueError('Loaded native nodes differ')
                if not batch[-1][0] or batch[-1][1:].any():raise ValueError('Native valid padding differs')
                loaded+=1
            result.update(loader_provenance=source.provenance(),train_records=len(train),validation_records=len(val),
                records_loaded=loaded,load_s=time.monotonic()-before,
                packed_store_files=len(list((out/'store-v2').iterdir())),packed_bytes=source.store.metadata['compressed_bytes'],
                raw_bytes=source.store.metadata['raw_bytes'],zstd_level=source.store.metadata['zstd_level'],
                normalization_scope='pilot training complexes only' if train else 'explicit unit smoke weights; no training records')
        finally:source.close()
    result['wall_s']=time.monotonic()-t;atomic_json(args.output/'report.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
