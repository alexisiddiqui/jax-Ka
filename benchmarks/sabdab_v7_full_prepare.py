"""Bounded, packed native SAbDab preparation and pre-teacher eligibility census."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.sabdab_units import prepare
from pkabench.sabdab_labels import source_evidence,make_sites
from pkatrain.augmentation_store import ReleaseWriter,AugmentationStore


def read(path):return json.loads(Path(path).read_text())


def sequence_key(sequence):return 's'+hashlib.sha256(sequence.encode()).hexdigest()[:20]


def initialise(cpus):
    import multiprocessing
    index=(multiprocessing.current_process()._identity[-1]-1)%len(cpus)
    os.sched_setaffinity(0,{cpus[index]})


def process(task):
    row,raw,role,capacity=task;start=time.monotonic();stage='prepare'
    result=dict(unit_id='sabdab:'+row['INSTANCE'],source='sabdab',source_native_id=row['INSTANCE'],
        pdb_id=row['PDB_ID'][-4:].lower(),native_source_split=row['ab_ag_split'],
        operational_role=role,native_component=row['ab_ag_cluster'],antibody_type=row['type'],
        construct=row.get('construct',''),archive_member=Path(row['cif_file']).name,
        source_cif_sha256=hashlib.sha256(raw).hexdigest())
    payload=None
    try:
        from biotite.structure.io import pdbx
        atoms,topology,side,meta=prepare(pdbx.CIFFile.read(io.StringIO(raw.decode())),row)
        meta['source_cif_sha256']=result['source_cif_sha256']
        file=pdbx.CIFFile();pdbx.set_structure(file,atoms);stream=io.StringIO();file.write(stream)
        canonical=stream.getvalue().encode()
        result.update(status='prepared',n_res=topology.n_residues,n_atoms=len(atoms),
            native_coordinate_sha256=meta['native_coordinate_sha256'],
            side_by_chain=[dict(chain=c,side=s) for c,s in meta['side_by_chain'].items()],chains=[dict(chain=c['chain'],side=c['side'],
                sequence=c['observed_sequence'],sequence_key=sequence_key(c['observed_sequence']),
                source_author_chains=c['source_author_chains']) for c in meta['chains']],
            scope_policy=meta['antibody_scope']['policy'],within_model_capacity=topology.n_residues<=capacity)
        payload=dict(structure=meta,canonical=canonical,source=raw,side=side,sites=[],evidence=None)
        if topology.n_residues>capacity:
            result['eligibility_status']='over_model_capacity_not_curated'
        else:
            stage='curation';t=time.monotonic()
            evidence,real,known,residues=source_evidence({},topology,meta,source_cif_bytes=raw)
            sites=make_sites(topology,meta,evidence,real,known,residues)
            counts={}
            for mask in ['train_mask','eval_mask']:
                rows=[s for s in sites if s[mask] and s['group']!='ARG']
                counts[mask]=len(rows);counts[mask+'_interface']=sum(s['interface'] for s in rows)
            selected='train_mask' if role=='train' else 'eval_mask'
            result.update(eligibility_status='potential_paired_interface' if counts[selected+'_interface'] else 'no_qualified_interface_geometry',
                preteacher_counts=counts,mask_reasons=dict(Counter(s['gap_rule'] for s in sites)),
                curation_s=time.monotonic()-t,
                true_termini=[dict(chain=c,number=n,insertion=i,group=g) for c,n,i,g in sorted(real)],
                known_termini=[dict(chain=c,number=n,insertion=i,group=g) for c,n,i,g in sorted(known)],
                missing_atom_residues=sum(bool(a['missing_atoms']) for a in evidence['atoms']),
                gap_regions=sum(d['kind'].endswith('_gap') for d in evidence['defects']))
            payload.update(sites=sites,evidence=evidence)
    except Exception as exc:
        result.update(status='withheld_preparation' if payload is None else 'prepared_curation_unassessed',
            eligibility_status='unassessed',error_stage=stage,error=repr(exc),traceback=traceback.format_exc()[-3000:])
    result['wall_s']=time.monotonic()-start
    return result,payload


def run(args):
    require_compute(threads=args.workers+1,allow_comp1400=True)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);data=read(args.audit/'inputs.json')
    assignments={r['unit_id']:r for r in pq.read_table(args.assignments).to_pylist()}
    selected=[r for r in data['official'] if assignments.get('sabdab:'+r['INSTANCE'],{}).get('operational_role') in ('train','validation')]
    if len(selected)!=7684:raise ValueError('Frozen supervised SAbDab candidate count changed')
    wanted={Path(r['cif_file']).name:r for r in selected}
    if len(wanted)!=len(selected):raise ValueError('Duplicate source archive basenames')
    archive_pin=read(args.archive.parent/'download.json')['sha256']
    if digest(args.archive)!=archive_pin:raise ValueError('Pinned ML archive changed')
    provenance=dict(kind='v7-native-sabdab-preparation-audit-v1',archive_sha256=archive_pin,
        assignments_sha256=digest(args.assignments),input_sha256=digest(args.audit/'inputs.json'),
        model_capacity=args.capacity,teacher_or_masif_built=False,code_sha256=digest(__file__))
    atomic_json(args.output/'inputs.json',provenance)
    rows=[];seen=set();prepared=[];started=time.monotonic();cpus=sorted(os.sched_getaffinity(0))[:args.workers]
    with ReleaseWriter(args.output/'prepared',provenance,zstd_level=1) as writer:
        def consume(future):
            result,payload=future.result();rows.append(result)
            if payload is not None:
                uid=result['unit_id'];writer.add(uid,dict(result=result,structure=payload['structure'],sites=payload['sites'],evidence=payload['evidence']),
                    dict(source_cif=np.frombuffer(payload['source'],np.uint8),canonical_cif=np.frombuffer(payload['canonical'],np.uint8),side=payload['side']),
                    summary={k:result.get(k) for k in ['unit_id','pdb_id','operational_role','native_component','n_res','status','eligibility_status']})
                prepared.append(uid)
            if len(rows)%100==0:
                progress=dict(processed=len(rows),selected=len(selected),counts=dict(Counter(r['status'] for r in rows)),wall_s=time.monotonic()-started)
                atomic_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        with ProcessPoolExecutor(args.workers,initializer=initialise,initargs=(cpus,)) as pool:
            pending=set()
            with tarfile.open(args.archive,'r|gz') as archive:
                for member in archive:
                    name=Path(member.name).name
                    if not member.isfile() or name not in wanted:continue
                    if name in seen:raise ValueError('Duplicate selected archive member')
                    seen.add(name);row=wanted[name]
                    raw=archive.extractfile(member).read()
                    role=assignments['sabdab:'+row['INSTANCE']]['operational_role']
                    pending.add(pool.submit(process,(row,raw,role,args.capacity)))
                    if len(pending)>=2*args.workers:
                        done,pending=wait(pending,return_when=FIRST_COMPLETED)
                        for future in done:consume(future)
            if seen!=set(wanted):raise ValueError('Selected ML coordinates missing from archive')
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:consume(future)
        manifest=writer.finish(expected_ids=prepared)
    rows.sort(key=lambda r:r['unit_id']);columns=sorted({k for r in rows for k in r})
    pq.write_table(pa.Table.from_pylist([{k:r.get(k) for k in columns} for r in rows]),args.output/'units.parquet',compression='zstd')
    memberships=[dict(unit_id=r['unit_id'],source='sabdab',chain_id=c['chain'],sequence_key=c['sequence_key'],sequence=c['sequence'],
        annotation_role='antibody' if c['side']=='A' else 'antigen',side=c['side']) for r in rows for c in r.get('chains',[])]
    pq.write_table(pa.Table.from_pylist(memberships),args.output/'observed-membership.parquet',compression='zstd')
    summary=dict(status='full_sabdab_preparation_and_preteacher_curation_complete',selected=len(rows),
        prepared=len(prepared),counts=dict(Counter(r['status'] for r in rows)),
        eligibility=dict(Counter(r.get('eligibility_status','unassessed') for r in rows)),
        by_role={role:dict(Counter(r.get('eligibility_status','unassessed') for r in rows if r['operational_role']==role)) for role in ['train','validation']},
        packed_bytes=manifest['compressed_bytes'],raw_bytes=manifest['raw_bytes'],
        prepared_reader_files=len(manifest['files'])+1,workers=args.workers,wall_s=time.monotonic()-started,
        authoritative_split_preserved=True,actual_teacher_eligibility_not_yet_established=True,
        teacher_or_masif_built=False,release_active=False,units_sha256=digest(args.output/'units.parquet'),
        limitations=['Pre-teacher masks are an upper bound; real paired teacher availability remains required.',
            'Preparation and packing do not establish final source ownership or full-pool split clearance.'])
    atomic_json(args.output/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['audit','assignments','archive','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12);parser.add_argument('--capacity',type=int,default=1536)
    run(parser.parse_args())
