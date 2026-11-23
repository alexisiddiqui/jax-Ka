"""Whole existing-pool native-scope audit, using accepted source receipts."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq

from pkabench.runtime import atomic_json,digest,require_compute
from sabdab_v7_native_scope import check_one


def read(path):return json.loads(Path(path).read_text())


def initialise(cpus):
    import multiprocessing
    index=(multiprocessing.current_process()._identity[-1]-1)%len(cpus)
    os.sched_setaffinity(0,{cpus[index]})


def process(task):
    row=check_one(task)
    for chain in row.get('chains',[]):chain.pop('residue_keys',None)
    row['input_pins']=[dict(path=p,sha256=h) for p,h in sorted(row['input_pins'].items())]
    return row


def run(args):
    require_compute(threads=args.workers+1,allow_comp1400=True)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);data=read(args.audit/'inputs.json')
    inventory=read(Path(data['inventory'])/'inventory-manifest.json')
    units=[u for u in pq.read_table(args.audit/'units.parquet').to_pylist() if u['source']!='sabdab']
    groups={}
    for member in pq.read_table(args.audit/'scope-membership.parquet').to_pylist():
        groups.setdefault(member['unit_id'],set()).add(member['sequence_key'])
    tasks=[(inventory['runtime'],dict(u,_baseline_root=inventory['baseline']),sorted(groups.get(u['unit_id'],set())),['whole_pool_native_build_audit']) for u in units]
    rows=[];started=time.monotonic()
    with ProcessPoolExecutor(args.workers,initializer=initialise,initargs=(sorted(os.sched_getaffinity(0))[:args.workers],)) as pool:
        for row in pool.map(process,tasks,chunksize=8):
            rows.append(row)
            if len(rows)%1000==0:
                progress=dict(processed=len(rows),total=len(units),counts=dict(Counter(r['status'] for r in rows)),wall_s=time.monotonic()-started)
                atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    columns=sorted({k for r in rows for k in r})
    pq.write_table(pa.Table.from_pylist([{k:r.get(k) for k in columns} for r in rows]),args.output/'units.parquet',compression='zstd')
    memberships=[dict(unit_id=r['unit_id'],source=r['source'],chain_id=c['chain_id'],sequence_key=c['sequence_key'],sequence=c['sequence']) for r in rows if r['status']=='checked' for c in r['chains']]
    pq.write_table(pa.Table.from_pylist(memberships),args.output/'observed-membership.parquet',compression='zstd')
    summary=dict(status='whole_existing_pool_native_scope_complete',units=len(rows),
        counts={source:dict(Counter(r['status'] for r in rows if r['source']==source)) for source in ['pkpdb','pinder']},
        inventory_scope_changed=sum(r.get('coordinate_scope_matches_inventory') is False for r in rows if r['status']=='checked'),
        workers=args.workers,wall_s=time.monotonic()-started,units_sha256=digest(args.output/'units.parquet'),
        inputs_sha256=digest(args.audit/'inputs.json'),source_units_sha256=digest(args.audit/'units.parquet'),
        no_teacher_or_features_built=True,release_active=False,
        limitations=['Native coordinate checks do not alone certify historical teacher context or feature reuse.'])
    atomic_json(args.output/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['audit','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=10);run(parser.parse_args())
