"""Held-out 9 A surface-filter pilot and measured CPU projection; no training."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from pkabench.runtime import atomic_json, digest, require_compute
from pkatrain.augmentation_store import AugmentationStore, fingerprint
from augmentation_masif_audit import (load_views, write_pdb, execute_frame, array_hash,
    worker_affinity, CONFIG, SEED)
from augmentation_masif_report import view, quantile, write_csv
from augmentation_surface_audit import fit, frames, rms
from augmentation_patch_scale import supports, surface_metrics

KEYS=('surface_distance_A','shape_index_rmse','hbond_rmse','electrostatics_rmse','hydropathy_rmse','residue_exposure_TV')


def evaluation_ids(parents):
    # Four spread bins per dataset, two typical and two tail; frozen before controls.
    return {r['parent'] for r in parents if r['size_bin'] in (0,1,3,4) and
            r['sample_kind']==('typical' if r['size_bin'] in (0,3) else 'tail')}


def measure(meta, arrays, ref, moving, role):
    ca=ref['coordinates'][arrays['backbone_atoms'][:,1]]
    mc=moving['coordinates'][arrays['backbone_atoms'][:,1]]
    rp=supports(ref,[9.])
    vp=supports(moving,[9.]);rows=[];valid_reference=0
    for i,rr in rp.items():
        rid=rr[9.][0];members=np.unique(ref['owner'][rid])
        if len(members)<4 or np.linalg.matrix_rank(ca[members]-ca[members].mean(0))<2:continue
        valid_reference+=1
        if i not in vp:continue
        vid=vp[i][9.][0];rot,trans=fit(mc[members],ca[members])
        sm=surface_metrics(ref,moving,rid,vid,rot,trans,len(ca))
        row=dict(parent=meta['parent'],dataset=meta['dataset'],role=role,anchor=i,
            surface_distance_A=sm['distance'],residue_exposure_TV=sm['exposure_tv'])
        row.update({k+'_rmse':float(v) for k,v in zip(('shape_index','hbond','electrostatics','hydropathy'),sm['rmse'])})
        rows.append(row)
    return rows,valid_reference


def frame(work, xyz, centers, his, meta, arrays, runtime, environment):
    work.mkdir(parents=True,exist_ok=True)
    signature=fingerprint(dict(xyz=array_hash(xyz),centers=array_hash(centers),his=his,
        config=CONFIG,environment=environment,bridge_sha256=digest(Path(__file__).with_name('augmentation_masif_bridge.py')),
        host_sha256=digest(__file__),raw_only=True))
    result=work/'features.npz';receipt=work/'receipt.json'
    if not (result.exists() and receipt.exists() and json.loads(receipt.read_text()).get('signature')==signature
            and json.loads(receipt.read_text()).get('npz_sha256')==digest(result)):
        write_pdb(work/'input.pdb',xyz,meta,arrays)
        job=dict(command='frame',raw_only=True,config=CONFIG,signature=signature,input='/audit/input.pdb',
            output='/audit/features.npz',receipt='/audit/receipt.json',his_states=his,
            anchor_residue=arrays['anchor_residue'].tolist(),anchor_centers=centers.tolist(),seed=SEED)
        execute_frame(runtime,work,job)
    with np.load(result) as z:data={k:z[k].copy() for k in z.files}
    data['coordinates']=xyz
    return data,json.loads(receipt.read_text())


def parent_task(task):
    parent,sample,runtime,output,evaluate=task
    started=time.monotonic();token=hashlib.sha256(parent['parent'].encode()).hexdigest()[:16]
    dest=Path(output)/'parents'/token;dest.mkdir(parents=True,exist_ok=True)
    environment=json.loads((runtime/'results/environment.json').read_text())
    with AugmentationStore(runtime/'results/full/coordinates') as store:record=store.raw(parent['parent'])
    m,a,_,his=load_views(sample,parent)
    a['anchor_residue']=record['arrays']['anchor_residue']
    ref=view(record,'historical_zero');xyz=ref['coordinates'];centers=ref['anchor_centers']
    # Verify raw-only preprocessing against the frozen old pipeline on three parents.
    parity=None
    if parent['size_bin']==2 and parent['sample_kind']=='typical':
        rebuilt,receipt=frame(dest/'raw-parity',xyz,centers,his,m,a,runtime,environment)
        errors={k:float(np.max(np.abs(rebuilt[k]-ref[k]))) for k in
            ('vertices','raw_features','vertex_area')}
        if any(e>1e-6 for e in errors.values()) or not np.array_equal(rebuilt['faces'],ref['faces']):
            raise ValueError('raw-only preprocessing differs from frozen frame: '+str(errors))
        parity=errors
    rng=np.random.default_rng(SEED+parent['source_index']);rot=Rotation.random(random_state=rng).as_matrix()
    trans=rng.uniform(-15,15,3)
    control,control_receipt=frame(dest/'rigid-control',np.asarray(xyz@rot+trans,np.float32),
        centers@rot+trans,his,m,a,runtime,environment)
    base_charge=record['metadata']['views'][record['metadata']['roles']['historical_zero']]['receipt']['charge_identity_sha256']
    if control_receipt['charge_identity_sha256']!=base_charge:raise ValueError('control charges changed')
    control_rows,expected=measure(parent,a,ref,control,'rigid_control')
    candidates=[];patches=list(control_rows);coords={};all_receipts=[control_receipt]
    old=Path(sample['historical'])/f"{parent['source_index']:03d}"
    repair=json.loads((old/'report.json').read_text())
    prep=json.loads((old/'preparation.json').read_text())
    heavy=np.flatnonzero(np.asarray(prep['is_native']) & np.asarray(prep['is_heavy']))
    reusable={array_hash(view(record,role)['coordinates']):view(record,role) for role in record['metadata']['roles']}
    timings=[]
    if evaluate:
        ref_frames=frames(xyz.astype(float),a['backbone_atoms']);refca=xyz[a['backbone_atoms'][:,1]]
        for metric in repair['metrics']:
            c=metric['candidate']
            if c==0:continue
            result=dict(parent=parent['parent'],dataset=parent['dataset'],candidate=c,
                amplitude_A=metric['input_amplitude_A'],geometry_valid=metric['geometry_valid'],status='geometry_rejected',
                error=None,expected_anchors=expected,mapped_anchors=0,reused=False,global_ca_rmsd_A=None,wall_s=None,frame_wall_s=None)
            if metric['geometry_valid']:
                start=time.monotonic();moving=np.asarray(np.load(old/f'native/c{c:02d}/min.npy')[heavy],np.float32)
                key=array_hash(moving);reused=key in reusable
                if reused:v=reusable[key];receipt=None
                else:
                    ff=frames(moving.astype(float),a['backbone_atoms'])
                    targets=np.stack([moving[a['backbone_atoms'][r,1]]+(centers[i]-refca[r])@ref_frames[r].T@ff[r]
                                      for i,r in enumerate(a['anchor_residue'])])
                    try:
                        v,receipt=frame(dest/f'c{c:02d}',moving,targets,his,m,a,runtime,environment)
                    except Exception as e:
                        result.update(status='surface_failed',error=str(e));candidates.append(result);continue
                    if receipt['charge_identity_sha256']!=base_charge:raise ValueError('candidate charges changed')
                    all_receipts.append(receipt)
                rr,nref=measure(parent,a,ref,v,f'c{c:02d}');patches+=rr
                assert nref==expected
                rotation,translation=fit(moving[a['backbone_atoms'][:,1]],refca)
                result.update(status='measured',expected_anchors=expected,mapped_anchors=len(rr),reused=reused,
                    global_ca_rmsd_A=rms(moving[a['backbone_atoms'][:,1]]@rotation+translation,refca),
                    wall_s=time.monotonic()-start,frame_wall_s=receipt['wall_s'] if receipt else None)
                coords[str(c)]=moving[a['backbone_atoms'][:,1]]
                if receipt:timings.append(dict(candidate=c,wall_s=receipt['wall_s'],timings=receipt['timings']))
            candidates.append(result)
    if coords:np.savez_compressed(dest/'candidate-ca.npz',reference=xyz[a['backbone_atoms'][:,1]],**coords)
    result=dict(parent=parent['parent'],dataset=parent['dataset'],n_residues=parent['n_residues'],
        size_bin=parent['size_bin'],sample_kind=parent['sample_kind'],evaluation=evaluate,
        expected_anchors=expected,control_mapped=len(control_rows),control_wall_s=control_receipt['wall_s'],
        raw_parity=parity,candidates=candidates,new_frame_timings=timings,patches=patches,
        wall_s=time.monotonic()-started,path=str(dest),
        repair_production_s=repair['production_s'],repair_stage=repair['stage'])
    atomic_json(dest/'result.json',result)
    return result


def thresholds(controls):
    return {k:quantile(controls,k)['p95'] for k in KEYS}


def bad_fraction(rows, expected, limits, keys=KEYS):
    if expected<=0:return 1.
    bad=sum(any(row[k]>limits[k] for k in keys) for row in rows)
    return (bad+expected-len(rows))/expected


def farthest(candidates, coordinates, number=3):
    selected=[];minimum={}
    ref=coordinates['reference']
    for c in candidates:
        xyz=coordinates[str(c)];rot,trans=fit(xyz,ref);minimum[c]=rms(xyz@rot+trans,ref)
    while minimum and len(selected)<number:
        best=max(minimum,key=lambda c:(minimum[c],-c));selected.append(best);del minimum[best]
        for c in minimum:
            x,y=coordinates[str(c)],coordinates[str(best)];rot,trans=fit(x,y)
            minimum[c]=min(minimum[c],rms(x@rot+trans,y))
    return selected


def report(results,output,elapsed,workers):
    calibration=[r for r in results if not r['evaluation']];evaluation=[r for r in results if r['evaluation']]
    controls=[p for r in calibration for p in r['patches'] if p['role']=='rigid_control']
    limits=thresholds(controls)
    control_fractions=[bad_fraction([p for p in r['patches'] if p['role']=='rigid_control'],r['expected_anchors'],limits) for r in calibration]
    allowance=float(np.quantile(control_fractions,.95))
    summaries=[];rows=[];selections={}
    for result in evaluation:
        byrole=defaultdict(list)
        for row in result['patches']:byrole[row['role']].append(row)
        candidates=result['candidates'];accepted=[]
        for c in candidates:
            if c['status']=='measured':
                c['bad_patch_fraction']=bad_fraction(byrole[f"c{c['candidate']:02d}"],result['expected_anchors'],limits)
                c['accepted']=c['bad_patch_fraction']<=allowance
            else:c['accepted']=False;c['bad_patch_fraction']=None
            if c['accepted']:accepted.append(c['candidate'])
            rows.append(c)
        with np.load(Path(result['path'])/'candidate-ca.npz') as coords:
            selected=farthest(accepted,coords)
            pair=[]
            for i,c in enumerate(selected):
                for d in selected[:i]:
                    x,y=coords[str(c)],coords[str(d)];rot,trans=fit(x,y);pair.append(rms(x@rot+trans,y))
        selections[result['parent']]=selected
        summaries.append(dict(parent=result['parent'],dataset=result['dataset'],eligible=len(accepted),selected=selected,
            selected_pairwise_rmsd_A=pair,control_bad_fraction=bad_fraction(byrole['rigid_control'],result['expected_anchors'],limits)))
    write_csv(output/'candidate-filter.csv',rows)
    write_csv(output/'patches.csv',[p for r in results for p in r['patches']])
    counts=json.loads((output.parents[2]/'inventory.json').read_text())['counts']
    populations={d:sum(v for k,v in counts.items() if k.startswith(d+'_')) for d in ('atlas','pinder','pkpdb')}
    means={d:float(np.mean([t['wall_s'] for r in results if r['dataset']==d for t in r['new_frame_timings']] + [r['control_wall_s'] for r in results if r['dataset']==d])) for d in populations}
    # Each frame uses four disjoint CPU cores (upstream PyMesh internals); no GPU.
    total_worker_s=sum(populations[d]*means[d]*25 for d in populations)
    eta={str(cpus):dict(workers=cpus//4,ideal_days=total_worker_s/(cpus//4)/86400,
                        planning_days_with_30pct_margin=total_worker_s/(cpus//4)/86400*1.3) for cpus in (96,240,288,380)}
    summary=dict(calibration_parents=len(calibration),evaluation_parents=len(evaluation),surface_radius_A=9.,
        thresholds=limits,max_bad_patch_fraction=allowance,control_fractions=control_fractions,
        heldout_control_acceptance=sum(r['control_bad_fraction']<=allowance for r in summaries),
        yield_three=sum(len(r['selected'])==3 for r in summaries),yield_two=sum(len(r['selected'])>=2 for r in summaries),
        parents=summaries,populations=populations,mean_surface_frame_wall_s=means,eta_surface_filter=eta,
        pilot_elapsed_s=elapsed,workers=workers,new_surface_frames=sum(len(r['new_frame_timings'])+1+(r['raw_parity'] is not None) for r in results),
        limitations=['Thresholds calibrated on 18 independent rigid rebuild controls; 12 held-out candidate parents',
            'Thirty size/response-enriched proteins, not a random population sample',
            'One random rigid orientation per parent; no repeatability confidence guarantee',
            'Frame ETA includes MSMS and APBS but excludes ANM construction, repair, queueing and transfers',
            '25 surfaces per parent means one repaired zero plus all 24 candidates, without production rigid controls',
            'Full-data scaling assumes four cores per surface worker and measured new-frame costs',
            'Shape/chemistry nearest-vertex measurements include discretisation and anchor mapping noise',
            'No training and no frozen MaSIF inference in this filter'])
    atomic_json(output/'summary.json',summary)
    atomic_json(output/'selections.json',selections)
    text=['Surface-preservation selection pilot','====================================','',
        f"Calibration: {len(calibration)} parents; held-out evaluation: {len(evaluation)} parents.",
        f"Three-view yield: {summary['yield_three']}/{len(evaluation)}; at least two: {summary['yield_two']}/{len(evaluation)}.",
        f"Held-out rigid controls accepted: {summary['heldout_control_acceptance']}/{len(evaluation)}.",
        f"Maximum bad-patch fraction: {allowance:.4f}.",'','Patch thresholds','----------------','']
    text += [f'* {k}: {v:.5f}' for k,v in limits.items()]
    text += ['','Surface screening ETA','---------------------','']+[f"* {c} CPUs: ideal {v['ideal_days']:.2f} days, planning {v['planning_days_with_30pct_margin']:.2f} days." for c,v in eta.items()]
    text += ['','Limitations','-----------','']+['* '+s for s in summary['limitations']]
    (output/'report.rst').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary),flush=True)


def main(args):
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']))
    args.output.mkdir(parents=True,exist_ok=True)
    sample=json.loads((args.runtime/'results/sample.json').read_text());heldout=evaluation_ids(sample['parents'])
    assert len(heldout)==12
    atomic_json(args.output/'design.json',dict(evaluation=sorted(heldout),calibration=sorted(r['parent'] for r in sample['parents'] if r['parent'] not in heldout),
        seed=SEED,radius_A=9.,host_sha256=digest(__file__),bridge_sha256=digest(Path(__file__).with_name('augmentation_masif_bridge.py'))))
    queue=multiprocessing.Queue()
    for subset in np.array_split(sorted(os.sched_getaffinity(0)),args.workers):queue.put(list(map(int,subset)))
    results=[];started=time.monotonic()
    tasks=[(r,sample,args.runtime,args.output,r['parent'] in heldout) for r in sample['parents']]
    tasks.sort(key=lambda t:(not t[4],-t[0]['n_residues']))
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_affinity,initargs=(queue,)) as pool:
        for future in as_completed([pool.submit(parent_task,t) for t in tasks]):
            r=future.result();results.append(r)
            print(json.dumps(dict(done=len(results),parent=r['parent'],evaluated=r['evaluation'],wall_s=r['wall_s'])),flush=True)
    report(results,args.output,time.monotonic()-started,args.workers)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--runtime',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--workers',type=int,default=24)
    main(p.parse_args())
