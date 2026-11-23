"""Independent packed-label, teacher-context and pre/post-mask verification."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path

import pyarrow.parquet as pq

from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.sabdab_labels import qualify_labels
from pkatrain.augmentation_store import AugmentationStore
from sabdab_v7_pkai_build import validate_labels
from sabdab_v7_teacher_context import normalise_pdb

LABELS=None;PREPARED=None;PLAN=None


def read(path):return json.loads(Path(path).read_text())


def initialise(labels,prepared,plan):
    global LABELS,PREPARED,PLAN
    LABELS=AugmentationStore(labels);PREPARED=AugmentationStore(prepared);PLAN=plan


def one(uid):
    frame=LABELS.raw(uid);data=frame['metadata'];result=data['result'];plan=PLAN[uid]
    if result['unit_id']!=uid or result['operational_role']!=plan['operational_role']:
        raise ValueError('Teacher source/role identity differs: '+uid)
    if result['ownership_ready']!=plan['ownership_ready'] or result['release_active']:
        raise ValueError('Ownership/activation policy changed: '+uid)
    if result['status']!='qualified':
        return dict(unit_id=uid,status=result['status'],error=result.get('error'))
    labels=data['labels'];validate_labels(labels)
    for state in ['AB','A','B']:
        if {tuple(r[:4]) for r in labels['pkai'][state]}!={tuple(r[:4]) for r in labels['pkai_plus'][state]}:
            raise ValueError('Teacher models disagree on target identities: '+uid)
    prepared=PREPARED.raw(uid);source=prepared['metadata'];meta=source['structure']
    if data['prepared_frame_sha256']!=PREPARED.by_id[uid]['sha256'] or result['native_coordinate_sha256']!=meta['native_coordinate_sha256']:
        raise ValueError('Prepared native context differs: '+uid)
    sites=copy.deepcopy(source['sites']);counts=qualify_labels(sites,labels)
    if counts!=result['counts'] or sites!=data['sites']:raise ValueError('Post-teacher masks differ from frozen source curation: '+uid)
    mask='train_mask_interface' if result['operational_role']=='train' else 'eval_mask_interface'
    if counts.get(mask,0)<1:raise ValueError('Qualified record lacks paired interface targets: '+uid)
    if data['true_termini']!=source['result']['true_termini'] or data['known_termini']!=source['result']['known_termini']:
        raise ValueError('Natural/cropped terminal evidence changed: '+uid)
    key_map={(r['chain'],r['number'],r['insertion']):(r['chain'],r['number'],r['insertion']) for r in meta['residue_map']}
    states={state:normalise_pdb(frame['arrays']['teacher_pdb_'+state].tobytes().decode(),
        data['request']['mappings'][state],key_map) for state in ['AB','A','B']}
    if states['A'].keys()&states['B'].keys() or states['A']|states['B']!=states['AB']:
        raise ValueError('Free teacher atoms differ from biological partner removal: '+uid)
    for state in ['A','B']:
        if {meta['side_by_chain'][k[0]] for k in states[state]}!={state}:
            raise ValueError('Teacher free state splits a biological partner: '+uid)
    return dict(unit_id=uid,status='qualified',role=result['operational_role'],counts=counts,
        ownership_ready=result['ownership_ready'],teacher_origin=result['teacher_origin']['kind'])


def run(args):
    require_compute(threads=args.workers+1)
    summary=read(args.labels/'summary.json');inputs=read(args.labels/'build-inputs.json')
    for path,pin in inputs['inputs_sha256'].items():
        if digest(path)!=pin:raise ValueError('Pinned build input changed: '+path)
    for path,pin in inputs['teacher_input_pins'].items():
        if digest(path)!=pin:raise ValueError('Pinned teacher package/model changed')
    for path,pin in inputs['code_sha256'].items():
        if digest(path)!=pin:raise ValueError('Frozen build code changed')
    if inputs['masif'] or inputs['new_apo_labels'] or inputs['release_active']:raise ValueError('Unexpected annotation scope')
    plan={r['unit_id']:r for r in pq.read_table(args.audit_root/'build-manifest-v2/units.parquet').to_pylist()
        if r['source']=='sabdab' and r['eligible_for_build']}
    if args.allow_canary_subset:
        if not set(inputs['selection'])<=set(plan):raise ValueError('Canary contains records outside the approved candidate inventory')
        plan={uid:plan[uid] for uid in inputs['selection']}
    with AugmentationStore(args.labels/'labels') as store:
        store.verify_files();ids=sorted(store.by_id)
    if set(ids)!=set(plan) or len(ids)!=summary['attempted']:raise ValueError('Packed release differs from complete candidate inventory')
    with ProcessPoolExecutor(args.workers,initializer=initialise,
            initargs=(args.labels/'labels',args.audit_root/'sabdab/prepared',plan)) as pool:
        rows=list(pool.map(one,ids,chunksize=8))
    if Counter(r['status'] for r in rows)!=Counter(summary['counts']):raise ValueError('Final status accounting differs')
    result=dict(status='passed',records=len(rows),counts=dict(Counter(r['status'] for r in rows)),
        by_role={role:sum(r.get('role')==role for r in rows if r['status']=='qualified') for role in ['train','validation']},
        duplicate_context_review_pending=sum(r.get('ownership_ready') is False for r in rows),
        checks=['full packed file/frame hashes and record inventory','both teacher models finite and identical target keys',
            'native coordinate/source identity and source roles','post-teacher masks independently recomputed',
            'natural termini unchanged','exact chemical AB = disjoint biological A union B',
            'model/package and frozen code pins','no MaSIF, apo labels or training activation'],
        summary_sha256=digest(args.labels/'summary.json'),label_manifest_sha256=digest(args.labels/'labels/manifest.json'),
        release_active=False,masif=False,new_apo_labels=False)
    atomic_json(args.labels/'independent-verification.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--audit-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--allow-canary-subset',action='store_true');run(p.parse_args())
