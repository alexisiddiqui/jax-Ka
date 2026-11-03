"""Cold end-to-end deferred-APBS benchmark on repaired coordinates; no training."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import numpy as np
import psutil
from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.augmentation_store import AugmentationStore,fingerprint
from augmentation_masif_audit import CONFIG,SEED,load_views,write_pdb,bridge,worker_affinity,array_hash
from augmentation_masif_report import view
from augmentation_surface_audit import fit,rms,frames
from augmentation_surface_filter import measure,bad_fraction,KEYS
from augmentation_surface_filter_cpu import canonical_errors

BASE_FIELDS=('vertices','faces','normals','owner','raw_features','vertex_area','anchor_valid',
             'anchor_vertex_ids','anchor_mapping_distance','anchor_centers')
CHEAP_KEYS=tuple(k for k in KEYS if k!='electrostatics_rmse')


class MemoryMonitor:
    def __init__(self):self.peak=0.;self.phase='geometry';self.peaks={};self.stop=threading.Event()
    def sample(self):
        process=psutil.Process()
        while not self.stop.is_set():
            rss=0
            for child in [process]+process.children(recursive=True):
                try:rss+=child.memory_info().rss
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            mb=rss/1024**2;self.peak=max(self.peak,mb);self.peaks[self.phase]=max(self.peaks.get(self.phase,0.),mb)
            self.stop.wait(.1)
    def __enter__(self):self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join()


def geometry(work,xyz,targets,his,meta,arrays,runtime):
    work.mkdir()
    write_pdb(work/'input.pdb',xyz,meta,arrays)
    job=dict(command='frame',raw_only=True,geometry_only=True,config=CONFIG,seed=SEED,
        signature=fingerprint(dict(xyz=array_hash(xyz),targets=array_hash(targets),phase='geometry',bridge=digest(Path(__file__).with_name('augmentation_masif_bridge.py')))),
        input='/audit/input.pdb',output='/audit/features.npz',receipt='/audit/receipt.json',
        his_states=his,anchor_residue=arrays['anchor_residue'].tolist(),anchor_centers=targets.tolist())
    atomic_json(work/'job.json',job);bridge(runtime,work,'job.json')
    with np.load(work/'features.npz') as z:v={k:z[k].copy() for k in z.files}
    v['coordinates']=xyz
    receipt=json.loads((work/'receipt.json').read_text())
    assert receipt['geometry_only'] and receipt['charge_identity_sha256'] is None
    # Intermediate MSMS files are no longer needed; retain exact H PDB and mesh.
    for child in work.iterdir():
        if child.name not in {'input.pdb','protonated.pdb','features.npz','receipt.json','job.json','job.json.log'}:
            if child.is_dir():shutil.rmtree(child)
            else:child.unlink()
    return v,receipt


def electrostatics(work,v,runtime):
    job=dict(command='electrostatics',protonated='/audit/protonated.pdb',mesh='/audit/features.npz',
        output='/audit/electrostatics.npz',receipt='/audit/electrostatics-receipt.json',
        signature=fingerprint(dict(phase='electrostatics',mesh=digest(work/'features.npz'),pdb=digest(work/'protonated.pdb'),
            bridge=digest(Path(__file__).with_name('augmentation_masif_bridge.py')))))
    atomic_json(work/'electrostatics-job.json',job);bridge(runtime,work,'electrostatics-job.json')
    with np.load(work/'electrostatics.npz') as z:v['raw_features'][:,2]=z['electrostatics']
    receipt=json.loads((work/'electrostatics-receipt.json').read_text())
    for child in work.iterdir():
        if child.name.startswith('potential'):child.unlink()
    return receipt


def check_geometry(gold,new,with_e=False):
    ref={k:gold[k].copy() for k in BASE_FIELDS};var={k:new[k].copy() for k in BASE_FIELDS}
    if not with_e:ref['raw_features'][:,2]=0.;var['raw_features'][:,2]=0.
    return canonical_errors(ref,var)


def task(t):
    parent,sample,runtime,exhaustive,output,limits,allowance=t
    started=time.monotonic();token=hashlib.sha256(parent['parent'].encode()).hexdigest()[:16]
    dest=output/'parents'/token;dest.mkdir(parents=True)
    m,a,_,his=load_views(sample,parent)
    # Saved anchor locations fix correspondence; no exhaustive features enter selection.
    with AugmentationStore(runtime/'results/full/coordinates') as store:record=store.raw(parent['parent'])
    a['anchor_residue']=record['arrays']['anchor_residue']
    oldref=view(record,'historical_zero');refxyz=oldref['coordinates'];centers=oldref['anchor_centers']
    # Reuse frozen input targets, not measured features, so remapping is identical
    # to the exhaustive audit even for its pre-existing amplitude frames.
    cache=runtime/'results/cache'/token
    ref_job=json.loads((cache/record['metadata']['roles']['historical_zero']/'job.json').read_text())
    reference_targets=np.asarray(ref_job['anchor_centers'])
    target_by_hash={}
    for role,fid in record['metadata']['roles'].items():
        key=array_hash(view(record,role)['coordinates'])
        target_by_hash[key]=np.asarray(json.loads((cache/fid/'job.json').read_text())['anchor_centers'])
    old=Path(sample['historical'])/f"{parent['source_index']:03d}"
    prep=json.loads((old/'preparation.json').read_text());heavy=np.flatnonzero(np.asarray(prep['is_native'])&np.asarray(prep['is_heavy']))
    repair=json.loads((old/'report.json').read_text());refs=frames(refxyz.astype(float),a['backbone_atoms']);refca=refxyz[a['backbone_atoms'][:,1]]
    candidates={};geo_receipts=[];e_receipts=[];cheap_rows={};queries=[];selected=[];timings={}
    tmp_root=os.environ.get('SLURM_TMPDIR') or '/tmp'
    with MemoryMonitor() as memory,tempfile.TemporaryDirectory(prefix='ka-staged-'+os.environ['SLURM_JOB_ID']+'-',dir=tmp_root) as scratch:
        scratch=Path(scratch);tick=time.monotonic()
        ref,receipt=geometry(scratch/'zero',refxyz,reference_targets,his,m,a,runtime);geo_receipts.append(receipt)
        memory.phase='electrostatics';receipt=electrostatics(scratch/'zero',ref,runtime);e_receipts.append(receipt)
        refcharge=receipt['charge_identity_sha256'];timings['reference_s']=time.monotonic()-tick
        memory.phase='geometry';tick=time.monotonic();coords={'reference':refca}
        for metric in repair['metrics']:
            c=metric['candidate']
            if not c or not metric['geometry_valid']:continue
            xyz=np.asarray(np.load(old/f'native/c{c:02d}/min.npy')[heavy],np.float32)
            ff=frames(xyz.astype(float),a['backbone_atoms'])
            key=array_hash(xyz)
            if key in target_by_hash:targets=target_by_hash[key]
            else:
                targets=np.asarray(json.loads((exhaustive/'parents'/token/f'c{c:02d}/job.json').read_text())['anchor_centers'])
            v,receipt=geometry(scratch/f'c{c:02d}',xyz,targets,his,m,a,runtime);geo_receipts.append(receipt)
            rows,expected=measure(parent,a,ref,v,f'c{c:02d}')
            for row in rows:row['electrostatics_rmse']=None
            candidates[c]=v;cheap_rows[c]=rows;coords[str(c)]=xyz[a['backbone_atoms'][:,1]]
        timings['candidate_geometry_and_screen_s']=time.monotonic()-tick
        minimum={}
        for c,v in candidates.items():
            if bad_fraction(cheap_rows[c],expected,limits,CHEAP_KEYS)>allowance:continue
            rot,trans=fit(coords[str(c)],refca);minimum[c]=rms(coords[str(c)]@rot+trans,refca)
        tick=time.monotonic();memory.phase='electrostatics';full_rows={}
        while minimum and len(selected)<3:
            c=max(minimum,key=lambda c:(minimum[c],-c));del minimum[c];queries.append(c)
            receipt=electrostatics(scratch/f'c{c:02d}',candidates[c],runtime);e_receipts.append(receipt)
            if receipt['charge_identity_sha256']!=refcharge:raise ValueError('staged candidate charge assignment changed')
            rows,nref=measure(parent,a,ref,candidates[c],f'c{c:02d}');full_rows[c]=rows
            if nref!=expected:raise ValueError('reference anchor cohort changed')
            if bad_fraction(rows,expected,limits)>allowance:continue
            selected.append(c)
            for other in minimum:
                x,y=coords[str(other)],coords[str(c)];rot,trans=fit(x,y)
                minimum[other]=min(minimum[other],rms(x@rot+trans,y))
        timings['deferred_electrostatics_and_selection_s']=time.monotonic()-tick
        measured_wall=time.monotonic()-started
        # Validation occurs after timing; exhaustive outputs cannot inform the cold algorithm.
        memory.phase='validation'
        golden=json.loads((exhaustive/'parents'/token/'result.json').read_text())
        reusable={array_hash(view(record,r)['coordinates']):view(record,r) for r in record['metadata']['roles']}
        max_errors=check_geometry(oldref,ref,True)
        basecharge=record['metadata']['views'][record['metadata']['roles']['historical_zero']]['receipt']['charge_identity_sha256']
        if refcharge!=basecharge:raise ValueError('staged reference charge assignment changed')
        for c,v in candidates.items():
            key=array_hash(v['coordinates'])
            if key in reusable:gold=reusable[key]
            else:
                with np.load(Path(golden['path'])/f'c{c:02d}/features.npz') as z:gold={k:z[k].copy() for k in z.files}
            errors=check_geometry(gold,v,c in queries)
            for k,e in errors.items():max_errors[k]=max(max_errors.get(k,0.),e)
        from augmentation_surface_filter_summary import staged_selection
        gold_policy=staged_selection(golden,limits,allowance)
        if selected!=gold_policy['selected'] or queries!=gold_policy['queried_candidates']:
            raise ValueError('cold deferred-APBS selection differs from exhaustive policy')
        np.savez_compressed(dest/'selected-heavy-coordinates.npz',reference=refxyz,**{str(c):candidates[c]['coordinates'] for c in selected})
    result=dict(parent=parent['parent'],dataset=parent['dataset'],size_bin=parent['size_bin'],n_residues=parent['n_residues'],
        passed=True,selected=selected,queried_candidates=queries,geometry_frames=len(geo_receipts),
        electrostatic_frames=len(e_receipts),wall_s=measured_wall,timings=timings,
        geometry_bridge_wall_s=sum(r['wall_s'] for r in geo_receipts),electrostatic_bridge_wall_s=sum(r['wall_s'] for r in e_receipts),
        peak_tree_rss_MiB=max(memory.peaks.get('geometry',0),memory.peaks.get('electrostatics',0)),memory_by_phase_MiB=memory.peaks,
        canonical_max_errors=max_errors,exhaustive_selection_match=True,validation_s=time.monotonic()-started-measured_wall)
    atomic_json(dest/'result.json',result);return result


def main(args):
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']))
    args.output.mkdir(parents=True,exist_ok=True)
    sample=json.loads((args.runtime/'results/sample.json').read_text())
    calibration=json.loads((args.calibration/'summary.json').read_text());limits=calibration['thresholds'];allowance=calibration['max_bad_patch_fraction']
    ids={r['parent'] for r in calibration['parents']};parents=[r for r in sample['parents'] if r['parent'] in ids]
    assert len(parents)==12
    atomic_json(args.output/'design.json',dict(cold_geometry_and_electrostatics=True,parents=sorted(ids),thresholds=limits,
        allowance=allowance,workers=args.workers,allocated_cpus=int(os.environ['SLURM_CPUS_PER_TASK']),
        code_sha256=digest(__file__),bridge_sha256=digest(Path(__file__).with_name('augmentation_masif_bridge.py')),
        calibration_sha256=digest(args.calibration/'summary.json'),sample_sha256=digest(args.runtime/'results/sample.json')))
    queue=multiprocessing.Queue()
    for subset in np.array_split(sorted(os.sched_getaffinity(0)),args.workers):queue.put(list(map(int,subset)))
    started=time.monotonic();results=[]
    tasks=[(p,sample,args.runtime,args.exhaustive,args.output,limits,allowance) for p in sorted(parents,key=lambda p:-p['n_residues'])]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_affinity,initargs=(queue,)) as pool:
        for f in as_completed([pool.submit(task,t) for t in tasks]):
            result=f.result();results.append(result);print(json.dumps(dict(done=len(results),parent=result['parent'],wall_s=result['wall_s'],selected=result['selected'],electrostatics=result['electrostatic_frames'])),flush=True)
    atomic_json(args.output/'execution.json',dict(passed=True,parents=len(results),elapsed_s=time.monotonic()-started,workers=args.workers,
        results=results,slurm_job=os.environ['SLURM_JOB_ID']))
    from augmentation_surface_staged_report import report
    report(args.output,args.runtime,args.calibration)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('runtime','exhaustive','calibration','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=12);main(p.parse_args())
