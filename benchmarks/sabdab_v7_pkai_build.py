"""Packed native SAbDab pKAI/pKAI+ annotations; no MaSIF or pool activation."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pkabench.runtime import atomic_json,config_hash,digest,require_compute
from pkabench.sabdab_labels import export_states,qualify_labels
from pkatrain.augmentation_store import AugmentationStore,ReleaseWriter
from sabdab_v7_native_scope import atom_signature

STORE=None;CANARY_STORE=None;CONFIG=None


def read(path):return json.loads(Path(path).read_text())


def validate_labels(labels):
    if labels.get('errors'):raise ValueError('Teacher reported errors: '+repr(labels['errors']))
    for model in ['pkai','pkai_plus']:
        if set(labels[model])!={'AB','A','B'}:raise ValueError('Incomplete teacher state inventory')
        for state,values in labels[model].items():
            keys=[]
            for row in values:
                if len(row)!=5 or not math.isfinite(float(row[4])):raise ValueError('Invalid/nonfinite teacher target')
                keys.append(tuple(row[:4]))
            if len(keys)!=len(set(keys)):raise ValueError('Duplicate teacher target identity')


def canary(rows,count):
    reuse=[r for r in rows if r['teacher_action'].startswith('reuse_verified')]
    chosen={r['unit_id']:r for r in reuse}
    # Deterministic spread over actual prepared sizes and BOTH assigned roles.
    for role in ['train','validation']:
        group=sorted([r for r in rows if r['operational_role']==role],key=lambda r:(r['n_res'],r['unit_id']))
        for i in np.linspace(0,len(group)-1,max(2,(count-len(reuse))//2),dtype=int):
            chosen[group[int(i)]['unit_id']]=group[int(i)]
    return sorted(chosen.values(),key=lambda r:r['unit_id'])


def initialise(config,cpus):
    global STORE,CANARY_STORE,CONFIG
    import multiprocessing
    index=(multiprocessing.current_process()._identity[-1]-1)%len(cpus)
    os.sched_setaffinity(0,set(cpus[index]));CONFIG=config
    STORE=AugmentationStore(config['prepared'])
    if config.get('canary_store'):CANARY_STORE=AugmentationStore(config['canary_store'])


def reused_labels(uid):
    if CANARY_STORE is not None and uid in CANARY_STORE.by_id:
        frame=CANARY_STORE.raw(uid);result=frame['metadata']['result']
        if result['status']!='qualified':raise ValueError('Canary teacher was not qualified')
        return copy.deepcopy(frame['metadata']['labels']),dict(kind='verified_same_run_canary',
            native_coordinate_sha256=result['native_coordinate_sha256'],
            frame_sha256=CANARY_STORE.by_id[uid]['sha256'],source_origin=result['teacher_origin'])
    if uid in CONFIG['pilot']:
        proof=CONFIG['pilot'][uid];entry=Path(proof['entry'])
        if digest(entry/'receipt.json')!=proof['receipt_sha256'] or digest(entry/'labels.json')!=proof['labels_sha256']:
            raise ValueError('Verified pilot teacher changed')
        labels=read(entry/'labels.json')
        return labels,dict(kind='same_native_pilot',labels_sha256=proof['labels_sha256'],entry=str(entry))
    proof=CONFIG['compatible'].get(uid)
    if proof:
        entry=Path(proof['native_teacher_entry'])
        if digest(entry/'labels.json')!=proof['labels_sha256']:raise ValueError('Compatible PINDER teacher changed')
        mapping={tuple(r['native']):tuple(r['sabdab']) for r in proof['residue_key_map']}
        labels=copy.deepcopy(read(entry/'labels.json'))
        for model in ['pkai','pkai_plus']:
            for state,values in labels[model].items():
                labels[model][state]=[[*mapping[tuple(row[:3])],*row[3:]] for row in values]
        return labels,dict(kind='compatible_pinder_teacher',labels_sha256=proof['labels_sha256'],
            entry=str(entry),teacher_context_id=proof['teacher_context_id'])
    return None,None


def one(row):
    started=time.monotonic();uid=row['unit_id'];stage='packed_native_input';arrays={}
    result=dict(unit_id=uid,source='sabdab',operational_role=row['operational_role'],n_res=row['n_res'],
        teacher_action=row['teacher_action'],ownership_ready=row['ownership_ready'],release_active=False)
    try:
        from biotite.structure.io import pdbx
        from jaxpropka.topology import load_topology
        frame=STORE.raw(uid);meta=frame['metadata']['structure']
        file=pdbx.CIFFile.read(io.StringIO(frame['arrays']['canonical_cif'].tobytes().decode()))
        atoms=pdbx.get_structure(file,model=1,altloc='occupancy',use_author_fields=True,include_bonds=True)
        if atom_signature(atoms)!=meta['native_coordinate_sha256']:raise ValueError('Packed native atom context changed')
        topology=load_topology(atoms,gap_policy='cap',freeze_disulfides=True)
        if [(k.chain,k.number,k.insertion) for k in topology.keys]!=[(r['chain'],r['number'],r['insertion']) for r in meta['residue_map']]:
            raise ValueError('Packed native residue order changed')
        if topology.n_residues!=row['n_res']:raise ValueError('Native residue count differs from build plan')
        side=np.asarray([0 if meta['side_by_chain'][k.chain]=='A' else 1 for k in topology.keys],np.int8)
        if not np.array_equal(side,frame['arrays']['side']):raise ValueError('Native biological-side partition changed')
        source=frame['metadata']['result'];real={(r['chain'],r['number'],r['insertion'],r['group']) for r in source['true_termini']}
        sites=copy.deepcopy(frame['metadata']['sites'])
        with tempfile.TemporaryDirectory(prefix='sabdab-pkai-',dir=CONFIG['scratch']) as directory:
            directory=Path(directory);stage='teacher_export'
            mappings=export_states(topology,meta,real,directory)
            request=dict(mappings=mappings,teacher_input_pins=CONFIG['teacher_pins'])
            atomic_json(directory/'request.json',request)
            labels,origin=reused_labels(uid)
            if origin and origin.get('native_coordinate_sha256',meta['native_coordinate_sha256'])!=meta['native_coordinate_sha256']:
                raise ValueError('Canary teacher native context differs')
            if labels is None:
                if row['teacher_action']!='build_native_pkai_pkai_plus_AB_A_B':raise ValueError('Unproved teacher reuse action')
                stage='native_teacher_prediction';t=time.monotonic()
                with (directory/'teacher.log').open('w') as log:
                    subprocess.run([CONFIG['teacher_python'],'-m','pkabench.pkai_native',str(directory/'request.json'),str(directory/'labels.json')],
                        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
                labels=read(directory/'labels.json');result['teacher_process_s']=time.monotonic()-t
                origin=dict(kind='new_native_prediction',teacher_python=CONFIG['teacher_python'])
            stage='teacher_qualification';validate_labels(labels);counts=qualify_labels(sites,labels)
            mask='train_mask_interface' if row['operational_role']=='train' else 'eval_mask_interface'
            result.update(status='qualified' if counts.get(mask,0)>0 else 'withheld_no_paired_interface_targets',
                counts=counts,native_coordinate_sha256=meta['native_coordinate_sha256'],
                prepared_frame_sha256=STORE.by_id[uid]['sha256'],source_cif_sha256=meta['source_cif_sha256'],
                native_source_split=meta['native_source_split'],native_component=meta['native_component'],
                teacher_origin=origin,teacher_timings_s=labels.get('timings_s',{}))
            for state in ['AB','A','B']:arrays['teacher_pdb_'+state]=np.frombuffer((directory/(state+'.pdb')).read_bytes(),np.uint8)
            metadata=dict(labels=labels,sites=sites,request=request,true_termini=source['true_termini'],
                known_termini=source['known_termini'],source_owner=row['source_owner'],
                supervision_scope_policy=meta['antibody_scope']['policy'],
                prepared_store=CONFIG['prepared'],prepared_frame_sha256=STORE.by_id[uid]['sha256'])
    except Exception as exc:
        result.update(status='withheld_teacher_failure',error_stage=stage,error=repr(exc),traceback=traceback.format_exc()[-4000:])
        metadata={}
    result['wall_s']=time.monotonic()-started
    return result,metadata,arrays


def run(args):
    require_compute(threads=args.workers*args.cpus_per_worker+1)
    root=args.audit_root;plan=root/'build-manifest-v2';report=read(plan/'report.json')
    if digest(plan/'units.parquet')!=report['output_pins']['units.parquet']:raise ValueError('Build plan changed')
    rows=[r for r in pq.read_table(plan/'units.parquet').to_pylist() if r['source']=='sabdab' and r['eligible_for_build']]
    if len(rows)!=4926:raise ValueError('Frozen native candidate count changed')
    if args.canary:rows=canary(rows,args.canary)
    if args.require_canary:
        previous=read(args.require_canary/'summary.json')
        if previous['status']!='complete' or previous['counts']!={'qualified':previous['attempted']}:
            raise ValueError('Native packed canary has unresolved failures')
    reference=read(root.parent/'jax-Ka/pkabench/pretraining/pinder-pkai-fv-v1/labels-v3/manifest.json')
    package=args.teacher_python.parent.parent/'lib/python3.11/site-packages/pkai'
    teacher_pins={str(package/n):reference['input_pins'][str(package/n)] for n in
        ['pKAI.py','protein.py','residue.py','atom.py','models/pKAI_model.pt','models/pKAI+_model.pt']}
    for path,pin in teacher_pins.items():
        if digest(path)!=pin:raise ValueError('Teacher package/model changed: '+path)
    pilot={r['unit_id']:r for r in read(root/'reuse/pilot-reuse.json')['verified']}
    compatible={r['unit_id']:r for r in read(root/'reuse/teacher-context.json')['records'] if r['teacher_reuse_approved']}
    paths=[plan/'units.parquet',plan/'report.json',root/'sabdab/prepared/manifest.json',
        root/'reuse/pilot-reuse.json',root/'reuse/teacher-context.json']
    code=[Path(__file__),*Path(__file__).parents[1].joinpath('src/pkabench').glob('*.py')]
    provenance=dict(kind='sabdab-v7-native-pkai-labels-v1',selection=[r['unit_id'] for r in rows],
        inputs_sha256={str(p):digest(p) for p in paths},code_sha256={str(p):digest(p) for p in code},
        teacher_input_pins=teacher_pins,workers=args.workers,cpus_per_worker=args.cpus_per_worker,
        protocol='fixed-coordinate CPU pKAI+pKAI+ AB/A/B; frozen native preparation and existing masks',
        new_apo_labels=False,masif=False,release_active=False,zstd_level=1)
    if args.require_canary:
        prior=read(args.require_canary/'build-inputs.json')
        for field in ['inputs_sha256','code_sha256','teacher_input_pins','protocol']:
            if prior[field]!=provenance[field]:raise ValueError('Canary input/code/protocol differs: '+field)
        provenance['qualified_canary_sha256']=digest(args.require_canary/'labels/manifest.json')
    if args.output.exists() and not args.resume:raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=args.resume)
    manifest_path=args.output/'build-inputs.json'
    if manifest_path.exists() and read(manifest_path)!=provenance:raise ValueError('Resume input/code/protocol differs')
    atomic_json(manifest_path,provenance)
    config=dict(prepared=str(root/'sabdab/prepared'),teacher_python=str(args.teacher_python),
        teacher_pins=teacher_pins,pilot=pilot,compatible=compatible,scratch='/tmp',
        canary_store=str(args.require_canary/'labels') if args.require_canary else None)
    cpus=sorted(os.sched_getaffinity(0));cpu_groups=[cpus[1+i*args.cpus_per_worker:1+(i+1)*args.cpus_per_worker] for i in range(args.workers)]
    started=time.monotonic()
    with ReleaseWriter(args.output/'labels',provenance,zstd_level=1) as writer:
        results=[r['summary'] for r in writer.rows]
        pending=[r for r in sorted(rows,key=lambda r:(-r['n_res'],r['unit_id'])) if r['unit_id'] not in writer.ids]
        with ProcessPoolExecutor(args.workers,initializer=initialise,initargs=(config,cpu_groups)) as pool:
            futures=[pool.submit(one,row) for row in pending]
            for future in as_completed(futures):
                result,metadata,arrays=future.result();metadata['result']=result
                writer.add(result['unit_id'],metadata,arrays,summary=result);results.append(result)
                progress=dict(processed=len(results),total=len(rows),counts=dict(Counter(r['status'] for r in results)),
                    new_predictions=sum(r.get('teacher_origin',{}).get('kind')=='new_native_prediction' for r in results),
                    wall_s=time.monotonic()-started,release_active=False)
                atomic_json(args.output/'progress.json',progress)
                if len(results)%25==0 or args.canary:print(json.dumps(dict(progress,latest_id=result['unit_id'],latest_status=result['status'],error=result.get('error'))),flush=True)
        manifest=writer.finish(expected_ids=[r['unit_id'] for r in rows])
    columns=sorted({k for r in results for k in r})
    pq.write_table(pa.Table.from_pylist([{k:r.get(k) for k in columns} for r in results]),args.output/'units.parquet',compression='zstd')
    counts=dict(Counter(r['status'] for r in results));verified=0
    with AugmentationStore(args.output/'labels') as store:
        for uid in store.by_id:
            frame=store.raw(uid);r=frame['metadata']['result']
            if r['status']=='qualified':validate_labels(frame['metadata']['labels'])
            verified+=1
    summary=dict(status='complete' if counts.get('qualified',0)==len(rows) else 'complete_with_explicit_holds',
        attempted=len(rows),counts=counts,by_role={role:dict(Counter(r['status'] for r in results if r['operational_role']==role)) for role in ['train','validation']},
        workers=args.workers,wall_s=time.monotonic()-started,packed_bytes=manifest['compressed_bytes'],
        reader_files=len(manifest['files'])+1,verified_records=verified,zstd_level=1,
        teacher_origins=dict(Counter(r.get('teacher_origin',{}).get('kind','failed') for r in results)),
        duplicate_context_review_pending=sum(not r['ownership_ready'] for r in results),
        job=os.environ['SLURM_JOB_ID'],release_active=False,masif=False,new_apo_labels=False,
        units_sha256=digest(args.output/'units.parquet'),label_manifest_sha256=digest(args.output/'labels/manifest.json'))
    atomic_json(args.output/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
    if args.canary and summary['status']!='complete':raise RuntimeError('Packed canary has explicit holds; inspect before expansion')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['audit-root','output','teacher-python']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--cpus-per-worker',type=int,choices=[1,2],default=1)
    p.add_argument('--canary',type=int);p.add_argument('--require-canary',type=Path);p.add_argument('--resume',action='store_true')
    run(p.parse_args())
