"""Compute future pilot frames privately; atomically publish completed cache links."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import numpy as np


def task(t):
    parent,c,sample,runtime,root,private=t
    from augmentation_surface_filter import frame
    from augmentation_masif_audit import load_views,array_hash
    from augmentation_masif_report import view
    from augmentation_surface_audit import frames
    from pkatrain.augmentation_store import AugmentationStore
    canonical=root/'parents'/hashlib.sha256(parent['parent'].encode()).hexdigest()[:16]/f'c{c:02d}'
    if canonical.exists():return dict(parent=parent['parent'],candidate=c,status='already_started')
    with AugmentationStore(runtime/'results/full/coordinates') as store:record=store.raw(parent['parent'])
    m,a,_,his=load_views(sample,parent);a['anchor_residue']=record['arrays']['anchor_residue']
    ref=view(record,'historical_zero');xyz=ref['coordinates'];centers=ref['anchor_centers']
    old=Path(sample['historical'])/f"{parent['source_index']:03d}"
    prep=json.loads((old/'preparation.json').read_text())
    heavy=np.flatnonzero(np.asarray(prep['is_native'])&np.asarray(prep['is_heavy']))
    moving=np.asarray(np.load(old/f'native/c{c:02d}/min.npy')[heavy],np.float32)
    if any(array_hash(moving)==array_hash(view(record,r)['coordinates']) for r in record['metadata']['roles']):
        return dict(parent=parent['parent'],candidate=c,status='existing_packed_frame')
    ref_frames=frames(xyz.astype(float),a['backbone_atoms']);refca=xyz[a['backbone_atoms'][:,1]]
    ff=frames(moving.astype(float),a['backbone_atoms'])
    targets=np.stack([moving[a['backbone_atoms'][r,1]]+(centers[i]-refca[r])@ref_frames[r].T@ff[r]
                      for i,r in enumerate(a['anchor_residue'])])
    dest=private/canonical.parent.name/canonical.name
    env=json.loads((runtime/'results/environment.json').read_text())
    v,receipt=frame(dest,moving,targets,his,m,a,runtime,env)
    charge=record['metadata']['views'][record['metadata']['roles']['historical_zero']]['receipt']['charge_identity_sha256']
    if receipt['charge_identity_sha256']!=charge:raise ValueError('charges changed in accelerated frame')
    # symlink is atomic and fails if the parent worker has started this frame.
    # Every artifact is complete before publication; no existing path is replaced.
    try:os.symlink(dest.resolve(),canonical);status='published'
    except FileExistsError:status='parent_already_started'
    return dict(parent=parent['parent'],candidate=c,status=status,wall_s=receipt['wall_s'])


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--original-code',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=24)
    args=p.parse_args()
    # Preserve the original cache signatures and exact scientific adapter.
    sys.path.insert(0,str(args.original_code))
    from pkabench.runtime import require_compute,atomic_json
    from augmentation_masif_audit import worker_affinity
    from augmentation_surface_filter import evaluation_ids
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']))
    args.output.mkdir(parents=True,exist_ok=True)
    sample=json.loads((args.runtime/'results/sample.json').read_text());heldout=evaluation_ids(sample['parents'])
    tasks=[]
    for parent in sample['parents']:
        token=hashlib.sha256(parent['parent'].encode()).hexdigest()[:16]
        dest=args.input/'parents'/token
        if parent['parent'] not in heldout or (dest/'result.json').exists():continue
        old=Path(sample['historical'])/f"{parent['source_index']:03d}"
        metrics=json.loads((old/'report.json').read_text())['metrics']
        for metric in metrics:
            c=metric['candidate']
            if c and metric['geometry_valid'] and not (dest/f'c{c:02d}').exists():
                tasks.append((parent,c,sample,args.runtime,args.input,args.output/'private'))
    # Earlier future candidates are completed first, before parent workers reach them.
    tasks.sort(key=lambda t:(t[1],-t[0]['n_residues']))
    queue=multiprocessing.Queue()
    for subset in np.array_split(sorted(os.sched_getaffinity(0)),args.workers):queue.put(list(map(int,subset)))
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_affinity,initargs=(queue,)) as pool:
        for f in as_completed([pool.submit(task,t) for t in tasks]):
            result=f.result();results.append(result);print(json.dumps(result),flush=True)
    atomic_json(args.output/'summary.json',dict(tasks=len(tasks),results=results,published=sum(r['status']=='published' for r in results)))
