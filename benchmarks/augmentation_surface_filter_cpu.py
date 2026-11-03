"""One-core raw surface rebuild parity and throughput against four-core pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import threading
import psutil
import numpy as np
from scipy.spatial import cKDTree
from pkabench.runtime import atomic_json,digest,require_compute
from augmentation_masif_audit import execute_frame,worker_affinity


def canonical_errors(reference, variant):
    # PyMesh can return a different vertex order when thread affinity changes.
    # Require a bijection with identical geometry, topology and per-vertex data.
    if reference['vertices'].shape!=variant['vertices'].shape:raise ValueError('vertex count changed')
    distances,to_reference=cKDTree(reference['vertices']).query(variant['vertices'])
    if distances.max()>1e-6 or len(np.unique(to_reference))!=len(to_reference):raise ValueError('surface geometry changed')
    inverse=np.argsort(to_reference);errors={}
    vertex_fields={'vertices','normals','owner','raw_features','vertex_area'}
    for k in reference.files if hasattr(reference,'files') else reference:
        a=reference[k];b=variant[k]
        if k in vertex_fields:b=b[inverse]
        elif k=='faces':
            aa=np.sort(a,axis=1);bb=np.sort(to_reference[b],axis=1)
            if set(map(tuple,aa))!=set(map(tuple,bb)):raise ValueError('surface topology changed')
            continue
        elif k=='anchor_vertex_ids':
            b=b.copy();valid=b>=0;b[valid]=to_reference[b[valid]]
        if a.dtype.kind=='f':
            errors[k]=float(np.max(np.abs(a-b))) if a.size else 0.
            if not np.allclose(a,b,rtol=1e-6,atol=1e-6):raise ValueError('one-core parity: '+k)
        elif not np.array_equal(a,b):raise ValueError('one-core parity: '+k)
    return errors


def trial(task):
    golden,dest,runtime=map(Path,task[:3]);verify_existing=task[3]
    dest.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(golden/'input.pdb',dest/'input.pdb')
    job=json.loads((golden/'job.json').read_text())
    started=time.monotonic();peak=[0.];stop=threading.Event()
    def sample_memory():
        process=psutil.Process()
        while not stop.is_set():
            rss=0
            for child in [process]+process.children(recursive=True):
                try:rss+=child.memory_info().rss
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            peak[0]=max(peak[0],rss/1024**2);stop.wait(.1)
    sampler=threading.Thread(target=sample_memory,daemon=True);sampler.start()
    try:
        if not verify_existing:execute_frame(runtime,dest,job)
    finally:stop.set();sampler.join()
    wall=time.monotonic()-started
    with np.load(golden/'features.npz') as a,np.load(dest/'features.npz') as b:
        assert set(a.files)==set(b.files)
        errors=canonical_errors(a,b)
    old=json.loads((golden/'receipt.json').read_text());new=json.loads((dest/'receipt.json').read_text())
    assert old['charge_identity_sha256']==new['charge_identity_sha256']
    return dict(source=str(golden),four_core_wall_s=old['wall_s'],one_core_wall_s=new['wall_s'],
                host_wall_s=wall,speed_ratio=new['wall_s']/old['wall_s'],max_errors=errors,
                one_core_peak_rss_MiB=new['peak_rss_MiB'],peak_tree_rss_MiB=peak[0],timings=new['timings'])


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-existing',action='store_true')
    p.add_argument('--heaviest',action='store_true');p.add_argument('--workers',type=int)
    args=p.parse_args();allocated=int(os.environ['SLURM_CPUS_PER_TASK']);workers=args.workers or allocated;require_compute(threads=allocated)
    args.output.mkdir(parents=True,exist_ok=True)
    # Pick small/middle/large completed controls per dataset; deterministic by size.
    available=[]
    for path in (args.input/'parents').glob('*/result.json'):
        if args.verify_existing and not (args.output/path.parent.name/'features.npz').exists():continue
        m=json.loads(path.read_text());available.append((m,path.parent/'rigid-control'))
    tasks=[];selection=[]
    for dataset in ('atlas','pinder','pkpdb'):
        sub=sorted([r for r in available if r[0]['dataset']==dataset],key=lambda r:r[0]['n_residues'])
        if args.heaviest:
            candidates=[(v['timings']['electrostatics_s'],m,path.parent/f"c{v['candidate']:02d}") for m,path in sub for v in m['new_frame_timings']]
            cost,m,source=max(candidates,key=lambda r:r[0])
            selection.append(dict(parent=m['parent'],n_residues=m['n_residues'],dataset=dataset,source=str(source)))
            tasks.append((str(source),str(args.output/source.parent.name/source.name),str(args.runtime),False))
        else:
            for idx in sorted(set([0,len(sub)//2,len(sub)-1])):
                m,source=sub[idx];selection.append(dict(parent=m['parent'],n_residues=m['n_residues'],dataset=dataset))
                tasks.append((str(source),str(args.output/source.parent.name),str(args.runtime),args.verify_existing))
    queue=multiprocessing.Queue()
    for core in sorted(os.sched_getaffinity(0))[:workers]:queue.put([core])
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_affinity,initargs=(queue,)) as pool:
        results=list(pool.map(trial,tasks))
    summary=dict(passed=True,workers=workers,selection=selection,results=results,code_sha256=digest(__file__),
                 median_slowdown=float(np.median([r['speed_ratio'] for r in results])),
                 one_core_worker_mean_s=float(np.mean([r['one_core_wall_s'] for r in results])))
    atomic_json(args.output/'summary.json',summary);print(json.dumps(summary),flush=True)
