"""Rebuild only PINDER graphs whose operational query mask changed."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain import production_graphs as pg
from pkatrain.native_paired_graph import build


def read(path):return json.loads(Path(path).read_text())


def one(task):
    row,native,runtime,baseline=task
    from biotite.structure.io import pdbx
    from jaxpropka.topology import load_topology
    uid=row['unit_id'];cid=row['source_native_id'];split='train' if row['operational_role']=='train' else 'val'
    fv=native['input_scope_id']=='v6c-native-observed-Fv-coordinates'
    entry=(Path(baseline)/'fv-build/pretraining/pinder-pkai-fv-v1/labels-v3/entries'/cid if fv else
        pg.source(Path(runtime),pg.PINDER+'/entries/'+cid))
    pins={p['path']:p['sha256'] for p in native['input_pins']}
    for path,pin in pins.items():
        if digest(path)!=pin:raise ValueError('Frozen native input changed: '+path)
    with gzip.open(entry/'AB.cif.gz','rt') as f:file=pdbx.CIFFile.read(f)
    atoms=pdbx.get_structure(file,model=1,altloc='occupancy',use_author_fields=True,include_bonds=True)
    topology=load_topology(atoms,gap_policy='cap',freeze_disulfides=True)
    labels=read(entry/'labels.json');sites=copy.deepcopy(read(entry/'sites.json'))
    alias={'NTR':'NTERM','CTR':'CTERM'}
    values={}
    for model in ['pkai','pkai_plus']:
        if model not in labels:continue
        for state,rows in labels[model].items():
            labels[model][state]=[[c,int(n),i,alias.get(g,g),float(v)] for c,n,i,g,v in rows if v is not None and np.isfinite(v)]
        if model=='pkai':values={s:{tuple(r[:4]) for r in rows} for s,rows in labels[model].items()}
    for site in sites:
        key=tuple(site[k] for k in ['chain','resnum','icode','group'])
        available=(key in values['AB'] and key in values[site['partner']] and key[3]!='ARG'
            and site.get('w_burial') is not None and site.get('w_interface') is not None)
        for mask in ['train_mask','eval_mask']:site[mask]=bool(site[mask] and available)
    sides={r['chain']:r['side'] for r in native['side_by_chain']}
    if fv:real=read(entry/'meta.json')['real_termini']
    else:
        # Preserve the existing legacy STUDENT node flags exactly. These are
        # never used to invent termini or modify the frozen teacher labels.
        real=[(k.chain,k.number,k.insertion,group) for index,k in enumerate(topology.keys)
            for flag,group in [(topology.nterm[index],'NTERM'),(topology.cterm[index],'CTERM')] if flag]
    graph,detail=build(topology,sides,sites,labels,real,split,node_termini=(topology.nterm,topology.cterm))
    result=dict(unit_id=uid,id=cid,split=split,operational_role=row['operational_role'],**detail,
        source_pins={str(entry/'sites.json'):digest(entry/'sites.json'),str(entry/'labels.json'):digest(entry/'labels.json')})
    if graph is None:return dict(result,status='withheld_no_qualified_interface_for_new_role'),None,None
    fields=pg.FIELDS['pinder']
    result.update(status='rebuilt',n=topology.n_residues,k=graph['neighbors'].shape[1],q=len(graph['query_residue']),
        s=len(graph['site_residue']),sk=graph['site_neighbors'].shape[1],arrays_sha256=pg.arrays_digest(graph,fields),
        w_burial_mean=float(np.mean(graph['w_burial'])),w_interface_mean=float(np.mean(graph['w_interface'])))
    return result,{f:graph[f].dtype.str for f in fields},pg.encode(graph,fields,level=1)


def run(args):
    require_compute(threads=args.workers+1)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);started=time.monotonic()
    plan=pq.read_table(args.audit/'build-manifest-v2/units.parquet').to_pylist()
    selected=[r for r in plan if r['source']=='pinder' and r['eligible_for_build'] and r['graph_action']=='rebuild_query_masks_for_new_role']
    native={r['unit_id']:r for r in pq.read_table(args.audit/'existing-native/units.parquet').to_pylist() if r['source']=='pinder'}
    tasks=[(r,native[r['unit_id']],str(args.runtime),str(args.baseline)) for r in selected]
    out=pg.output(args.output,'pinder');out.mkdir(parents=True)
    records=[];held={};all_rows=[];positions={};types=None
    with tempfile.TemporaryFile(dir='/tmp') as spool:
        with ProcessPoolExecutor(args.workers) as pool:
            for result,dtypes,blob in pool.map(one,tasks,chunksize=1):
                all_rows.append(result)
                if blob is None:held[result['id']]=result;continue
                if types is not None and types!=dtypes:raise ValueError('Native patch dtypes differ')
                types=dtypes;positions[result['id']]=(spool.tell(),len(blob));spool.write(blob);records.append(result)
        atomic_json(out/'ids.json',dict(ids=[[r['id'],r['split']] for r in all_rows]))
        spool.flush()
        def blobs():
            for r in records:
                offset,size=positions[r['id']];spool.seek(offset);yield spool.read(size)
        if records:pg._write_store(out,'pinder',records,types,blobs(),held,args.workers,fields=pg.FIELDS['pinder'],data_kind='paired',zstd_level=1)
    # Arbitrary source paths are records, avoiding thousands of Arrow columns.
    rows=[dict({k:v for k,v in r.items() if k!='source_pins'},source_pins=[dict(path=p,sha256=h) for p,h in r['source_pins'].items()]) for r in all_rows]
    pq.write_table(pa.Table.from_pylist(rows),args.output/'units.parquet',compression='zstd')
    report=dict(status='native_role_patch_build_complete',attempted=len(all_rows),counts=dict(Counter(r['status'] for r in all_rows)),
        withheld_ids=[r['unit_id'] for r in all_rows if r['status']!='rebuilt'],new_teacher_predictions=False,
        full_native_cache_rebuilt=False,patch_store=str(out/'store-v2'),wall_s=time.monotonic()-started,
        units_sha256=digest(args.output/'units.parquet'),release_active=False)
    atomic_json(args.output/'summary.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['audit','runtime','baseline','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);run(p.parse_args())
