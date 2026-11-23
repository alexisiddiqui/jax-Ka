"""Build source-aware SAbDab paired graphs into the existing five-file store."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import io
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pyarrow.parquet as pq
from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.augmentation_store import AugmentationStore
from pkatrain import production_graphs as pg
from pkatrain import production_loading as pl
from pkatrain.loading import BucketPolicy
from pkatrain.native_paired_graph import build

PREPARED=None;LABELS=None


def read(path):return json.loads(Path(path).read_text())


def initialise(prepared,labels):
    global PREPARED,LABELS
    PREPARED=AugmentationStore(prepared);LABELS=AugmentationStore(labels)


def one(row):
    from biotite.structure.io import pdbx
    from jaxpropka.topology import load_topology
    uid=row['unit_id'];frame=PREPARED.raw(uid);meta=frame['metadata']['structure'];label=LABELS.raw(uid)['metadata']
    if label['result']['status']!='qualified' or label['prepared_frame_sha256']!=PREPARED.by_id[uid]['sha256']:
        raise ValueError('Native teacher or prepared context not qualified: '+uid)
    file=pdbx.CIFFile.read(io.StringIO(frame['arrays']['canonical_cif'].tobytes().decode()))
    atoms=pdbx.get_structure(file,model=1,altloc='occupancy',use_author_fields=True,include_bonds=True)
    topology=load_topology(atoms,gap_policy='cap',freeze_disulfides=True)
    real=[(r['chain'],r['number'],r['insertion'],r['group']) for r in label['true_termini']]
    split='train' if row['operational_role']=='train' else 'val'
    graph,detail=build(topology,meta['side_by_chain'],label['sites'],label['labels'],real,split,
        node_termini=(topology.nterm,topology.cterm))
    if graph is None:raise ValueError('Previously qualified teacher has no paired graph: '+uid)
    fields=pg.FIELDS['pinder'];record=dict(id=row['id'],split=split,n=topology.n_residues,k=graph['neighbors'].shape[1],
        q=len(graph['query_residue']),s=len(graph['site_residue']),sk=graph['site_neighbors'].shape[1],
        component_id=row['group'],arrays_sha256=pg.arrays_digest(graph,fields),**detail)
    provenance=dict(source='sabdab',source_native_id=row['id'],native_source_split=meta['native_source_split'],
        native_component=meta['native_component'],operational_role=row['operational_role'],input_scope_id=meta['policy'],
        preparation_sha256=meta['preparation_sha256'],native_coordinate_sha256=meta['native_coordinate_sha256'],
        teacher_frame_sha256=LABELS.by_id[uid]['sha256'],supervision_scope_policy=meta['antibody_scope']['policy'],
        operational_group=row['group'],release_active=False)
    provenance['node_endpoint_policy']='legacy-observed-topology-segment-endpoints-v1'
    return record,{f:graph[f].dtype.str for f in fields},pg.encode(graph,fields,level=1),provenance


def run(args):
    require_compute(threads=args.workers+1)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);started=time.monotonic()
    rows=[r for r in pq.read_table(args.pools/'membership.parquet').to_pylist() if r['source']=='sabdab']
    if args.canary:
        chosen={}
        for role in ['train','validation']:
            ordered=sorted([r for r in rows if r['operational_role']==role],key=lambda r:(r['n_res'],r['id']))
            for i in np.linspace(0,len(ordered)-1,args.canary//2,dtype=int):chosen[ordered[int(i)]['id']]=ordered[int(i)]
        rows=list(chosen.values())
    if args.require_canary and read(args.require_canary/'report.json')['status']!='packed_native_graphs_complete':
        raise ValueError('Graph canary did not qualify')
    out=pg.output(args.output,'sabdab');out.mkdir(parents=True)
    atomic_json(out/'ids.json',dict(ids=[[r['id'],'train' if r['operational_role']=='train' else 'val'] for r in rows]))
    records=[];provenance={};positions={};dtypes=None
    # One temporary binary spool, no permanent per-structure files or arrays.
    with tempfile.TemporaryFile(dir='/tmp') as spool:
        with ProcessPoolExecutor(args.workers,initializer=initialise,
                initargs=(args.audit/'sabdab/prepared',args.labels/'labels')) as pool:
            tasks=iter(sorted(rows,key=lambda r:-r['n_res']))
            pending={pool.submit(one,r) for r in [next(tasks,None) for _ in range(args.workers*2)] if r is not None}
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    record,types,blob,pin=future.result()
                    if dtypes is not None and types!=dtypes:raise ValueError('Graph tensor dtypes differ')
                    dtypes=types;positions[record['id']]=(spool.tell(),len(blob));spool.write(blob)
                    records.append(record);provenance[record['id']]=pin
                    if len(records)%100==0:
                        progress=dict(processed=len(records),total=len(rows),compressed_bytes=spool.tell(),wall_s=time.monotonic()-started)
                        atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
                    row=next(tasks,None)
                    if row is not None:pending.add(pool.submit(one,row))
        records.sort(key=lambda r:r['id']);spool.flush()
        def blobs():
            for r in records:
                start,size=positions[r['id']];spool.seek(start);yield spool.read(size)
        pg._write_store(out,'sabdab',records,dtypes,blobs(),{},args.workers,fields=pg.FIELDS['pinder'],data_kind='paired',zstd_level=1)
    pool=args.output/'sabdab-train.tsv'
    with (args.pools/'sabdab.tsv').open() as source:
        import csv
        ranking={r['id']:r for r in csv.DictReader(source,delimiter='\t')}
    pool.write_text('id\tmin_fraction\n'+''.join(r['id']+'\t'+ranking[r['id']]['min_fraction']+'\n' for r in records if r['split']=='train'))
    policy=BucketPolicy((256,512,768,1024,1536),(4,2,2,1,1))
    manifest=pl.build_manifest(args.output,'sabdab',policy,args.workers,pool_path=pool,data_kind='paired',record_provenance=provenance)
    report=dict(status='packed_native_graphs_complete',records=len(records),counts=dict(Counter(r['split'] for r in records)),
        store=str(out/'store-v2'),reader_files=len(list((out/'store-v2').iterdir())),
        compressed_bytes=pg.read(out/'store-v2/metadata.json')['compressed_bytes'],
        sites=sum(r['q'] for r in records),wall_s=time.monotonic()-started,zstd_level=1,
        loader_manifest_sha256=digest(out/pl.MANIFEST),source_pools_sha256=digest(args.pools/'membership.parquet'),
        release_active=False,masif=False,model_training_performed=False)
    atomic_json(args.output/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['audit','labels','pools','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--canary',type=int);p.add_argument('--require-canary',type=Path)
    run(p.parse_args())
