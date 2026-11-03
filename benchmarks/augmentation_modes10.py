"""Matched 5/10-mode ANM candidate audit with the frozen 9 A surface gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import subprocess
import shutil
import numpy as np
from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.augmentation_store import AugmentationStore
from augmentation_masif_audit import SEED,load_views,array_hash
from augmentation_masif_report import view,write_csv
from augmentation_surface_audit import fit,rms,frames
from augmentation_surface_filter import measure,bad_fraction,KEYS
from augmentation_surface_staged import geometry,electrostatics,MemoryMonitor,CHEAP_KEYS


def paired_directions(old_vectors,old_values,new_vectors,new_values,seed):
    """Preserve the original five-mode components and add five orthogonal modes."""
    if new_vectors.shape[1]!=10 or old_vectors.shape[1]!=5:raise ValueError('expected five and ten modes')
    singular=np.linalg.svd(old_vectors.T@new_vectors[:,:5],compute_uv=False)
    if singular.min()<.99999:raise ValueError('lowest-five subspace changed; paired comparison requires review')
    extra=new_vectors[:,5:].copy()
    for j in range(5):
        if extra[np.argmax(np.abs(extra[:,j])),j]<0:extra[:,j]*=-1
    vectors=np.concatenate([old_vectors,extra],axis=1)
    if np.max(np.abs(vectors.T@vectors-np.eye(10)))>1e-5:raise ValueError('paired basis is not orthonormal')
    if not np.allclose(old_values,new_values[:5],rtol=1e-5,atol=1e-8):raise ValueError('five-mode eigenvalues changed')
    rng=np.random.default_rng(seed)
    first=rng.normal(size=(8,5))/np.sqrt(old_values)
    last=rng.normal(size=(8,5))/np.sqrt(new_values[5:])
    five=first@old_vectors.T;extra_part=last@extra.T;ten=five+extra_part
    fraction=np.sum(extra_part**2,axis=1)/np.sum(ten**2,axis=1)
    def norm(x):
        x=x.reshape(8,-1,3);return x/np.sqrt(np.mean(np.sum(x*x,axis=-1),axis=-1))[:,None,None]
    return norm(five),norm(ten),fraction,vectors


def run_parent(args):
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']))
    sample=json.loads((args.runtime/'results/sample.json').read_text())
    calibration=json.loads((args.calibration/'summary.json').read_text());limits=calibration['thresholds'];allow=calibration['max_bad_patch_fraction']
    ids={p['parent'] for p in calibration['parents']};parents=sorted([p for p in sample['parents'] if p['parent'] in ids],key=lambda p:p['parent'])
    parent=parents[args.index];index=parent['source_index'];token=hashlib.sha256(parent['parent'].encode()).hexdigest()[:16]
    dest=args.output/'parents'/token
    if args.shard_count>1:dest=dest/f'shard-{args.shard_index}'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():raise ValueError('refusing to replace completed parent')
    # Frozen legacy repair code, prepared systems and source topology are unchanged.
    sys.path.insert(0,str(Path(__file__).with_name('legacy')))
    import paired_core as q
    from stereo_min import add_stereo
    q.ROOT=args.runtime.parent/'native-large';q.DEST=q.ROOT/'units';q.p.ROOT=args.runtime.parent
    started=time.monotonic();prep=q.Prepared(index)
    m,a,_,his=load_views(sample,parent)
    if [prep.atom_names[j] for j in prep.heavy_ids]!=m['atom_names']:raise ValueError('heavy atom mapping changed')
    with AugmentationStore(args.runtime/'results/full/coordinates') as store:record=store.raw(parent['parent'])
    a['anchor_residue']=record['arrays']['anchor_residue'];ref=view(record,'historical_zero')
    original=q.ROOT/'results'/f'{index:03d}'
    with np.load(original/'source.npz') as z:bb=z['backbone'].copy()
    np.testing.assert_allclose(bb,prep.bb,atol=1e-6,rtol=0)
    with np.load(original/'modes.npz') as z:old_vectors=z['vectors'].copy();old_values=z['eigenvalues'].copy()
    from prody import ANM
    tick=time.monotonic();model=ANM();model.buildHessian(bb[:,1],cutoff=15.);model.calcModes(n_modes=10,zeros=False)
    values=model.getEigvals();vectors=model.getEigvecs();anm_s=time.monotonic()-tick
    if len(values)!=10 or np.any(values<=1e-6):raise ValueError('insufficient nonzero modes')
    d5,d10,fraction,paired_basis=paired_directions(old_vectors,old_values,vectors,values,SEED+index)
    # Exact reproduction of every original five-mode input after PDB rounding.
    max_control_error=0.
    for c in range(1,25):
        amp=(.5,1.,2.)[(c-1)//8];direction=(c-1)%8
        predicted=np.round(bb[:,1]+amp*d5[direction],3)
        old_ca=prep.candidate_ca(c)
        max_control_error=max(max_control_error,float(np.max(np.abs(predicted-old_ca))))
    if max_control_error>1e-6:raise ValueError('original five-mode target reconstruction failed')
    np.savez_compressed(dest/'modes.npz',eigenvalues=values,vectors=paired_basis,directions=d10,extra_mode_fraction=fraction)
    # Match each paired candidate's original local anchor offset, including the
    # three amplitude frames that the first audit reused from its earlier run.
    reusable={array_hash(view(record,role)['coordinates']):fid for role,fid in record['metadata']['roles'].items()}
    old_unit=q.DEST/f'{index:03d}';targets5={};old_frames={};old_ca={};old_xyz={};fallback_targets=[]
    ref_frames=frames(ref['coordinates'].astype(float),a['backbone_atoms']);refca=ref['coordinates'][a['backbone_atoms'][:,1]]
    for c in range(1,25):
        xyz=np.asarray(np.load(old_unit/f'native/c{c:02d}/min.npy')[prep.heavy_ids],np.float32)
        old_xyz[c]=xyz
        key=array_hash(xyz)
        path=(args.runtime/'results/cache'/token/reusable[key]/'job.json') if key in reusable else (args.exhaustive/'parents'/token/f'c{c:02d}/job.json')
        old_frames[c]=frames(xyz.astype(float),a['backbone_atoms']);old_ca[c]=xyz[a['backbone_atoms'][:,1]]
        if path.exists():targets5[c]=np.asarray(json.loads(path.read_text())['anchor_centers'])
        else:
            # Geometry-rejected five-mode candidates never had a surface job.
            # Their replacement uses the same fixed-reference local offset.
            targets5[c]=np.stack([old_ca[c][r]+(ref['anchor_centers'][i]-refca[r])@ref_frames[r].T@old_frames[c][r]
                                 for i,r in enumerate(a['anchor_residue'])])
            fallback_targets.append(c)
    mini=q.Minimizer(prep,system_hook=lambda system,p:add_stereo(system,p,1000.))
    metrics=[];patch_rows=[];all_xyz={};surface_failures=[];geo_count=0;e_count=0;old_charge=record['metadata']['views'][record['metadata']['roles']['historical_zero']]['receipt']['charge_identity_sha256']
    tmp_root=os.environ.get('SLURM_TMPDIR') or '/tmp'
    with MemoryMonitor() as memory,tempfile.TemporaryDirectory(prefix='ka-modes10-'+os.environ['SLURM_JOB_ID']+'-',dir=tmp_root) as scratch:
        scratch=Path(scratch)
        for c in range(1,25):
            if args.control_five and c not in (1,9,17):continue
            tick=time.monotonic();amp=(.5,1.,2.)[(c-1)//8];direction=(c-1)%8
            target=np.round(bb[:,1]+amp*(d5 if args.control_five else d10)[direction],3)
            work=scratch/f'c{c:02d}';work.mkdir();memory.phase='repair'
            post,detail=mini.run(prep.translated(target),target,work/'min.pdb')
            quality=q.quality(prep,mini,post)
            if args.shard_count>1 and (c-1)%args.shard_count!=args.shard_index:
                shutil_rmtree(work);continue
            heavy_xyz=np.asarray(post[prep.heavy_ids],np.float32);all_xyz[str(c)]=heavy_xyz
            rot,trans=fit(post[prep.ca_ids],ref['coordinates'][a['backbone_atoms'][:,1]])
            row=dict(parent=parent['parent'],dataset=parent['dataset'],candidate=c,amplitude_A=amp,direction=direction,
                geometry_valid=quality['geometry_valid'],status='geometry_rejected',surface_pass=False,
                pre_repair_target_rmsd_A=rms(target,bb[:,1]),post_repair_ca_rmsd_A=rms(post[prep.ca_ids]@rot+trans,ref['coordinates'][a['backbone_atoms'][:,1]]),
                repair_s=detail['minimisation_total_s'],old_five_heavy_rmsd_A=rms(heavy_xyz,old_xyz[c]),old_five_max_heavy_difference_A=float(np.max(np.abs(heavy_xyz-old_xyz[c]))),cheap_bad_fraction=None,full_bad_fraction=None,
                expected_anchors=None,mapped_anchors=None,electrostatics_run=False,wall_s=None)
            if quality['geometry_valid']:
                ff=frames(heavy_xyz.astype(float),a['backbone_atoms'])
                targets=np.stack([heavy_xyz[a['backbone_atoms'][r,1]]+(targets5[c][i]-old_ca[c][r])@old_frames[c][r].T@ff[r]
                                  for i,r in enumerate(a['anchor_residue'])])
                memory.phase='geometry'
                v=None
                for attempt in range(2):
                    try:
                        v,receipt=geometry(work/'surface',heavy_xyz,targets,his,m,a,args.runtime)
                        break
                    except subprocess.CalledProcessError as exc:
                        failures=dest/'surface-failures';failures.mkdir(exist_ok=True)
                        for log in (work/'surface').glob('*.log'):shutil.copyfile(log,failures/f'c{c:02d}-attempt{attempt}-{log.name}')
                        shutil_rmtree(work/'surface')
                        if attempt==1:surface_failures.append(dict(candidate=c,phase='geometry',error=str(exc)))
                if v is None:
                    row.update(status='surface_tool_failed',wall_s=time.monotonic()-tick)
                    metrics.append(row);shutil_rmtree(work);continue
                geo_count+=1
                rows,expected=measure(parent,a,ref,v,f'c{c:02d}');cheap=bad_fraction(rows,expected,limits,CHEAP_KEYS)
                row.update(cheap_bad_fraction=cheap,expected_anchors=expected,mapped_anchors=len(rows),status='cheap_rejected')
                if cheap<=allow:
                    memory.phase='electrostatics';receipt=electrostatics(work/'surface',v,args.runtime);e_count+=1
                    if receipt['charge_identity_sha256']!=old_charge:raise ValueError('candidate charge inventory changed')
                    rows,nref=measure(parent,a,ref,v,f'c{c:02d}')
                    if nref!=expected:raise ValueError('reference anchor cohort changed')
                    full=bad_fraction(rows,expected,limits);row.update(full_bad_fraction=full,electrostatics_run=True,
                        surface_pass=full<=allow,status='accepted' if full<=allow else 'electrostatic_rejected')
                else:
                    for rr in rows:rr['electrostatics_rmse']=None
                patch_rows+=rows
            row['wall_s']=time.monotonic()-tick;metrics.append(row)
            atomic_json(dest/'checkpoint.json',dict(parent=parent,metrics=metrics,geometry_frames=geo_count,electrostatic_frames=e_count))
            shutil_rmtree(work)
            print(json.dumps(dict(parent=parent['parent'],done=c,amplitude=amp,status=row['status'])),flush=True)
    np.savez_compressed(dest/'coordinates.npz',reference=ref['coordinates'],**all_xyz)
    result=dict(parent=parent['parent'],dataset=parent['dataset'],size_bin=parent['size_bin'],modes=5 if args.control_five else 10,passed=True,
        metrics=metrics,geometry_frames=geo_count,electrostatic_frames=e_count,anm_s=anm_s,
        eigenvalues=values.tolist(),extra_mode_fraction=fraction.tolist(),anchor_fallback_candidates=fallback_targets,surface_tool_failures=surface_failures,shard_index=args.shard_index,shard_count=args.shard_count,five_mode_target_max_error_A=max_control_error,
        wall_s=time.monotonic()-started,peak_worker_tree_rss_MiB=memory.peak,
        code_sha256=digest(__file__),bridge_sha256=digest(Path(__file__).with_name('augmentation_masif_bridge.py')),
        prepared_sha256=digest(old_unit/'prepared.npz'),source_modes_sha256=digest(original/'modes.npz'))
    write_csv(dest/'patches.csv',patch_rows);atomic_json(dest/'result.json',result)
    print(json.dumps(dict(parent=parent['parent'],passed=True,geometry_frames=geo_count,electrostatic_frames=e_count,wall_s=result['wall_s'])),flush=True)


def shutil_rmtree(path):
    import shutil
    shutil.rmtree(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ('runtime','calibration','exhaustive','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--control-five',action='store_true');p.add_argument('--shard-count',type=int,default=1);p.add_argument('--shard-index',type=int,default=0);run_parent(p.parse_args())
