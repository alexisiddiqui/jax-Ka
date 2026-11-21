"""Repack verified zstd frames without re-encoding scientific payloads."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.augmentation_store import decode,fingerprint,FORMAT


def canonical_runs(ids):
    ids=np.asarray(ids)
    if ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>np.iinfo(np.int32).max):
        raise ValueError('invalid or unordered canonical residue selection')
    ids=ids.astype(np.int64)
    if np.any(np.diff(ids)<=0):
        raise ValueError('invalid or unordered canonical residue selection')
    cuts=np.r_[0,np.flatnonzero(np.diff(ids)!=1)+1,len(ids)]
    return [[int(ids[a]),int(ids[b-1])+1] for a,b in zip(cuts[:-1],cuts[1:])]


def check_frame(entry,record,profile):
    m=record['metadata'];a=record['arrays'];receipt=m['receipt'];s=entry['summary']
    if m['parent']!=entry['id'] or any(m[k]!=s[k] for k in ['dataset','cid','split']):raise ValueError('packed source identity/split mismatch')
    if any(receipt.get(k)!=v for k,v in profile.items()):raise ValueError('scientific checkpoint/runtime changed')
    policy=receipt.get('surface_policy','original-v1')
    if policy!=s['surface_policy']:raise ValueError('per-record preparation tag differs from original receipt')
    if receipt.get('status')!='complete':raise ValueError('failed receipt cannot enter training release')
    if receipt.get('inference_backend')=='jax-gpu' and (receipt.get('reference_patches_checked')!=3 or receipt.get('descriptor_max_reference_error',1)>1e-4):
        raise ValueError('missing GPU/TF reference validation')
    n=len(a['vertices']);nr=len(a['canonical_residue_ids']);owner=a['owner'];faces=a['faces']
    if a['vertices'].shape!=(n,3) or a['descriptors'].shape!=(n,80) or a['raw_features'].shape!=(n,4) or a['normals'].shape!=(n,3):raise ValueError('invalid dense feature dimensions')
    if not n or not nr or nr!=s['n'] or owner.shape!=(n,) or owner.dtype.kind not in 'iu' or owner.min()<0 or owner.max()>=nr:raise ValueError('invalid residue/owner correspondence')
    if faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu' or not len(faces) or faces.min()<0 or faces.max()>=n:raise ValueError('invalid face indices')
    if not all(np.isfinite(x).all() for x in a.values()):raise ValueError('nonfinite scientific fields')
    return canonical_runs(a['canonical_residue_ids'])


def shard_task(task):
    index,entries,output,state,profile,signature=task;name='records-%05d.bin'%index;path=Path(output)/name;state=Path(state)
    if state.exists():
        old=json.loads(state.read_text())
        if old['signature']!=signature:raise ValueError('shard resume selection changed')
        if path.exists() and digest(path)==old['sha256']:return old
    pending=path.with_suffix('.pending');rows=[];whole=hashlib.sha256();offset=0;fd=None;source=None
    try:
        with pending.open('wb') as target:
            for entry in entries:
                if source!=entry['path']:
                    if fd is not None:os.close(fd)
                    source=entry['path'];fd=os.open(source,os.O_RDONLY)
                data=os.pread(fd,entry['bytes'],entry['offset'])
                if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('source frame hash mismatch: '+entry['id'])
                record=decode(data,max_output_size=entry['raw_bytes']);runs=check_frame(entry,record,profile)
                row={k:entry[k] for k in ['id','bytes','raw_bytes','sha256']}
                row.update(shard=name,offset=offset,summary=dict(entry['summary'],canonical_runs=runs))
                target.write(data);whole.update(data);offset+=len(data);rows.append(row)
            target.flush();os.fsync(target.fileno())
        pending.replace(path)
    finally:
        if fd is not None:os.close(fd)
    result=dict(shard=name,sha256=whole.hexdigest(),bytes=offset,rows=rows,signature=signature)
    atomic_json(state,result);return result


def choose_pilot(parents,entries):
    lookup={e['id']:e for e in entries};cells=defaultdict(list)
    for row in parents:
        if row['status']!='covered':continue
        antibody=any(u['role']=='Ab' for u in row['required_units']);n=row['canonical_residues']
        size=next(i for i,b in enumerate([150,300,600,float('inf')]) if n<=b)
        current=any(lookup[u['id']]['summary']['surface_policy']!='original-v1' for u in row['required_units'])
        cells[row['dataset'],row['split'],antibody,size,current].append(row)
    chosen={}
    for key,rows in cells.items():
        rows=sorted(rows,key=lambda r:(r['canonical_residues'],r['cid']));row=rows[len(rows)//2];chosen[row['dataset'],row['cid']]=row
    for row in parents:
        if row['status']=='covered' and row['cid'] in ['1x3z','5gjh']:chosen[row['dataset'],row['cid']]=row
    ids={u['id'] for row in chosen.values() for u in row['required_units']}
    return sorted(chosen.values(),key=lambda r:(r['dataset'],r['cid'])),[e for e in entries if e['id'] in ids]


def pack(plan,output,work,workers=8,pilot=False,shard_bytes=2*1024**3):
    plan=Path(plan);output=Path(output);work=Path(work);output.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():raise FileExistsError('immutable release already published')
    selection=json.loads((plan/'selection.json').read_text());coverage=json.loads((plan/'coverage.json').read_text())
    if digest(plan/'coverage.json')!=selection['coverage_sha256']:raise ValueError('coverage changed after source selection')
    for root,sha in selection['source_manifests'].items():
        if digest(Path(root)/'manifest.json')!=sha:raise ValueError('source manifest changed')
    lookup={e['id']:e for e in selection['entries']};parents=coverage['parents'];count_holds=[]
    for row in parents:
        if row['status']=='covered':
            n=sum(lookup[u['id']]['summary']['n'] for u in row['required_units'])
            if n!=row['canonical_residues']:
                row['status']='held';row['reasons'].append('source_surface_residue_count_differs_from_training_assembly')
                count_holds.append(dict(dataset=row['dataset'],cid=row['cid'],surface_residues=n,training_residues=row['canonical_residues']))
    ids={u['id'] for row in parents if row['status']=='covered' for u in row['required_units']};entries=[e for e in selection['entries'] if e['id'] in ids]
    if pilot:parents,entries=choose_pilot(parents,entries);coverage['parents']=parents
    coverage['scope']='size/profile-spread loader pilot' if pilot else 'all currently covered v5 native assemblies'
    # Sort by original shard and offset: sequential source access, compact
    # destination files, and byte-identical zstd frames.
    entries.sort(key=lambda e:(e['path'],e['offset']));groups=[];current=[];size=0
    for e in entries:
        if e['bytes']>shard_bytes:raise ValueError('frame larger than destination shard budget')
        if current and size+e['bytes']>shard_bytes:groups.append(current);current=[];size=0
        current.append(e);size+=e['bytes']
    if current:groups.append(current)
    signature=fingerprint(dict(selection_sha256=digest(plan/'selection.json'),pilot=pilot,shard_bytes=shard_bytes,ids=[e['id'] for e in entries]))
    atomic_json(work/'launch.json',dict(records=len(entries),shards=len(groups),bytes=sum(e['bytes'] for e in entries),signature=signature))
    result=[];started=time.monotonic()
    with ProcessPoolExecutor(workers) as pool:
        futures=[pool.submit(shard_task,(i,g,str(output),str(work/('shard-%05d.json'%i)),selection['scientific_profile'],signature)) for i,g in enumerate(groups)]
        for future in as_completed(futures):
            item=future.result();result.append(item)
            atomic_json(work/'progress.json',dict(status='packing',shards_done=len(result),shards=len(groups),records=sum(len(x['rows']) for x in result),
                bytes=sum(x['bytes'] for x in result),wall_s=time.monotonic()-started))
    rows=[row for item in sorted(result,key=lambda x:x['shard']) for row in item['rows']];by_id={row['id']:row for row in rows}
    if len(by_id)!=len(entries) or set(by_id)!={e['id'] for e in entries}:raise ValueError('packed inventory mismatch')
    # Validate complete canonical partitions from the actual decoded frames,
    # including joined H/L surfaces and PINDER A/B map selections.
    for parent in parents:
        if parent['status']!='covered':continue
        runs=sorted(run for unit in parent['required_units'] for run in by_id[unit['id']]['summary']['canonical_runs'])
        cursor=0
        for lo,hi in runs:
            if lo!=cursor or hi<=lo:raise ValueError('canonical assembly gap/overlap: '+parent['dataset']+'/'+parent['cid'])
            cursor=hi
        if cursor!=parent['canonical_residues']:raise ValueError('canonical assembly extent mismatch')
    atomic_json(output/'index.json',dict(format=FORMAT,records=rows))
    hashes={item['shard']:item['sha256'] for item in result}
    atomic_json(output/'verification.json',dict(passed=True,records_checked=len(rows),raw_bytes=sum(e['raw_bytes'] for e in rows),file_digests=hashes,
        checks=['source compressed frame SHA','full decode and finite dense features','model/runtime and record-specific preparation profile',
            'owner mapping','complete nonoverlapping canonical training assemblies','byte-identical source-frame preservation']))
    atomic_json(output/'coverage.json',coverage)
    summary={}
    for dataset in ['pkpdb','pinder']:
        for split in ['train','val']:
            group=[p for p in parents if p['dataset']==dataset and p['split']==split]
            summary[dataset+'/'+split]=dict(total=len(group),covered=sum(p['status']=='covered' for p in group),held=sum(p['status']!='covered' for p in group))
    atomic_json(output/'coverage-summary.json',dict(coverage=summary,records=len(rows),shards=len(result),frame_bytes=sum(e['bytes'] for e in rows),
        new_count_mismatch_holds=count_holds,profile_counts=dict(Counter(e['summary']['surface_policy'] for e in rows)),
        partial_pool=True,original_membership_changed=False))
    atomic_json(output/'policy.json',dict(name='masif-native-explicit-profile-complete-assembly-v1',
        legacy_review='All selected legacy frames have the full source/geometry/field audit; topology warnings remain explicit.',
        mixed_preparation_profiles=True,training_membership_files_unchanged=True,missing_features='withhold complete assembly; never zero-fill or mix incomplete partners',
        frame_compression='original zstd level 1 frames, copied without re-encoding',source_manifests=selection['source_manifests'],
        audit_results_sha256=selection['audit_results_sha256'],outlier_review_sha256=selection['outlier_review_sha256']))
    shutil.copyfile(plan/'outlier-review.json',output/'outlier-review.json')
    for name in ['index.json','verification.json','coverage.json','coverage-summary.json','policy.json','outlier-review.json']:hashes[name]=digest(output/name)
    manifest=dict(format=FORMAT,verified=True,signature=signature,zstd_level=1,shard_bytes=shard_bytes,records=len(rows),
        compressed_bytes=sum(e['bytes'] for e in rows),raw_bytes=sum(e['raw_bytes'] for e in rows),files=hashes,
        provenance=dict(kind='dense-native-masif-explicit-profile-release-v1',**selection['scientific_profile'],
            preparation_policy='per-record-receipt-and-index',coverage='complete assemblies from available v5 features; partial overall pool',
            source_selection_sha256=digest(plan/'selection.json')))
    atomic_json(output/'manifest.json',manifest)
    atomic_json(work/'status.json',dict(status='complete',records=len(rows),shards=len(result),reader_files=len(hashes)+1,
        release=str(output),manifest_sha256=digest(output/'manifest.json'),wall_s=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True);pack(a.plan,a.output,a.work,a.workers,a.pilot)
