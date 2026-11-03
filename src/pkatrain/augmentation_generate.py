"""Coulson CPU generation of the five-mode 1/1/2 coordinate release.

Each parent runs in an isolated subprocess and bounded node-local scratch.
Only verified indexed zstd records, checkpoints and aggregate logs persist.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import zstandard
from pkabench.runtime import atomic_json,digest,require_compute
from .augmentation_selection import RECIPE,AMPLITUDES,QUOTAS,POOL_SIZES,candidate_ids,candidate_plan,select_quota
from .augmentation_quality import quality,rmsd,verify_coordinate_record
from .augmentation_store import ReleaseWriter,AugmentationStore,encode,decode,finalize_workers,fingerprint


def parent_id(row):return f"{row['dataset']}/{row['id']}/{row['state']}"


def parent_seed(parent):
    return int.from_bytes(hashlib.sha256(('20261013|'+parent).encode()).digest()[:4],'little')%2147483647


def read_inventory(path):
    with (Path(path)/'inventory.jsonl').open() as f:
        for line in f:yield json.loads(line)


def guard(args):require_compute(threads=1,allow_comp1400=args.allow_comp1400)


def register(args):
    guard(args);args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'inventory-manifest.json').exists():raise FileExistsError('inventory already frozen')
    census=json.loads((args.pilot/'inventory.json').read_text());runtime=args.pilot.parent/'pkabench'
    rows=[];pool_hashes={};missing=[]
    for dataset in ('pkpdb','pinder'):
        for split,suffix in (('train',''),('val','-val')):
            pool=runtime/f'training/pool-v4/{dataset}{suffix}.tsv';pool_hashes[str(pool)]=digest(pool)
            records=list(csv.DictReader(pool.open(),delimiter='\t'))
            lengths=[n for d,s,n in census['units'] if d==dataset and s==split];cursor=0
            for record in records:
                for state in (('assembly',) if dataset=='pkpdb' else ('A','B')):
                    n=int(lengths[cursor]);cursor+=1;cid=record['id']
                    source=(runtime/f'pretraining/pkpdb-v1/structures/{cid[1:3]}/{cid}.cif.gz' if dataset=='pkpdb'
                            else runtime/f'pretraining/pinder-pkai-v1/entries/{cid}/{state}.cif.gz')
                    required=[source]
                    if dataset=='pkpdb':required.append(runtime/f'pretraining/pkpdb-full-v1/entries/{cid}/defects.json')
                    else:required.extend([source.parent/f'map_{state}.json',source.parent/'input_atom_mask.json.gz'])
                    for path in required:
                        if not path.is_file():missing.append(str(path))
                    b=next((i for i,limit in enumerate((150,300,500,750,1500)) if n<=limit),5)
                    rows.append(dict(index=len(rows),dataset=dataset,split=split,id=cid,state=state,n=n,
                        group=record['group'],bin=b,path=str(source),seed=parent_seed(f'{dataset}/{cid}/{state}')))
                if len(rows)%5000==0:print(json.dumps(dict(inventoried=len(rows))),flush=True)
            if cursor!=len(lengths):raise ValueError('pool membership/census length differs')
    counts=dict(Counter(f"{r['dataset']}_{r['split']}" for r in rows))
    if counts!={k:v for k,v in census['counts'].items() if not k.startswith('atlas_')}:raise ValueError('pool counts changed')
    if missing:
        atomic_json(args.output/'missing-inputs.json',dict(missing=sorted(set(missing))))
        raise ValueError(f'{len(set(missing))} required local source files missing')
    if len({parent_id(r) for r in rows})!=len(rows):raise ValueError('duplicate parent inventory')
    # Length-limited workers fit the measured 2 GiB budget; larger units get 4 GiB.
    classes=(('small',args.small_workers,lambda r:r['n']<=500),('large',args.large_workers,lambda r:r['n']>500))
    assignment={};tasks=[];next_worker=0
    for name,workers,eligible in classes:
        bins=[[] for _ in range(workers)];load=np.zeros(workers)
        for row in sorted((r for r in rows if eligible(r)),key=lambda r:(-r['n'],parent_id(r))):
            j=int(np.argmin(load));bins[j].append(row['index']);load[j]+=5+(row['n']/150.)**2
            assignment[row['index']]=next_worker+j
        for j,members in enumerate(bins):tasks.append(dict(worker=next_worker+j,resource_class=name,indices=members))
        next_worker+=workers
    inventory=args.output/'inventory.jsonl'
    with inventory.open('wb') as f:
        for row in rows:f.write((json.dumps(dict(row,worker=assignment[row['index']]),sort_keys=True)+'\n').encode())
    # Canary uses previously audited sources, spread across size and both splits.
    old_sample=json.loads((args.pilot/'native-large/sample.json').read_text())
    old_units=json.loads((args.pilot/'native-large/unit-summary.json').read_text())
    eligible={parent_id(r) for r,u in zip(old_sample,old_units) if u['zero_valid']}
    canary=[]
    for dataset in ('pkpdb','pinder'):
        for b in (0,2,4):
            group=[r for r in rows if r['dataset']==dataset and r['bin']==b and parent_id(r) in eligible]
            group.sort(key=lambda r:(r['split']!=('val' if b==2 else 'train'),parent_id(r)))
            if not group:raise ValueError('no audited canary source')
            canary.append(group[0]['index'])
    atomic_json(args.output/'tasks.json',dict(tasks=tasks,canary_indices=canary))
    manifest=dict(kind='pkpdb-pinder-five-mode-quota-112-v1',recipe=RECIPE,parents=len(rows),counts=counts,
        inventory_sha256=digest(inventory),tasks_sha256=digest(args.output/'tasks.json'),pool_sha256=pool_hashes,
        workers=next_worker,small_workers=args.small_workers,large_workers=args.large_workers,
        source_runtime=str(runtime),source_pilot=str(args.pilot),node_scratch_only=True,
        storage=dict(zstd_level=1,shard_bytes=2*1024**3,file_limit=5000),training_enabled=False,
        seed='SHA256(20261013|dataset/id/state), independent of worker and pool order',
        source_snapshot_sha256=digest(Path(__file__)),legacy_sha256={p.name:digest(p) for p in args.legacy.glob('*.py')})
    atomic_json(args.output/'inventory-manifest.json',manifest)
    print(json.dumps(dict(parents=len(rows),counts=counts,workers=next_worker,canary=canary)),flush=True)


def backend(args,local,row):
    sys.path.insert(0,str(args.legacy))
    import paired_core as q
    import pilot as p
    p.ROOT=args.pilot;p.RUNTIME=args.pilot.parent/'pkabench'
    p.CONFIG=dict(p.CONFIG,seed=row['seed'])
    q.ROOT=local;q.DEST=local/'units'
    return q,p


def produce(args,row):
    """One child, fresh imports/context/model state; all files are in its local cwd."""
    import openmm as mm
    from openmm import app
    from prody import ANM
    from .augmentation_evidence import upstream_mask
    started=time.monotonic();parent=parent_id(row);local=Path.cwd()
    q,p=backend(args,local,row)
    from stereo_min import add_stereo
    t=time.monotonic();bb,names,keys,segments,heavy_source=p.source_atoms(row);source_s=time.monotonic()-t
    if not 4<=len(bb)<=1500:raise ValueError('resolved residue count outside audited 4..1500 range')
    if len(bb)!=row['n']:raise ValueError(f"resolved source length changed: expected {row['n']}, got {len(bb)}")
    source_hash=digest(row['path'])
    t=time.monotonic();anm=ANM();anm.buildHessian(bb[:,1],cutoff=15.);anm.calcModes(n_modes=5,zeros=False)
    values=anm.getEigvals();vectors=anm.getEigvecs()
    if len(values)!=5 or np.any(values<=1e-6):raise ValueError('insufficient nonzero ANM modes')
    coeff=np.random.default_rng(row['seed']).normal(size=(8,5))/np.sqrt(values)
    delta=(coeff@vectors.T).reshape(8,len(bb),3)
    delta/=np.sqrt(np.mean(np.sum(delta**2,axis=-1),axis=-1))[:,None,None]
    anm_s=time.monotonic()-t
    source=local/'results/000';source.mkdir(parents=True)
    dest=local/'units/000';dest.mkdir(parents=True)
    ff=app.ForceField('amber14-all.xml');missing=[]
    for i,name in enumerate(names):
        template=ff._templates['HIE' if name=='HIS' else name]
        required={a.name for a in template.atoms if a.element!=app.element.hydrogen}
        if (i==len(bb)-1 or segments[i]!=segments[i+1]) and np.sum(segments==segments[i])>1:required.add('OXT')
        missing.extend((i,a) for a in sorted(required-set(heavy_source[i])))
    completion=source/'c00';completion.mkdir()
    t=time.monotonic()
    if missing:
        inp=source/'completion-input.pdb';q.write_heavy(inp,[dict(zip(q.BB,b)) for b in bb],names,segments,[])
        p.Reconstructor()(inp,completion/'all.pdb')
    else:q.write_heavy(completion/'all.pdb',heavy_source,names,segments,[])
    completion_s=time.monotonic()-t
    receipt=dict(source=row,anm_s=anm_s)
    original=[{a:np.asarray(x,float) for a,x in residue.items()} for residue in heavy_source]
    q.source=lambda index:(receipt,bb,names,keys,segments,original)
    q.prepare(0,dest);prep=q.Prepared(0)
    mini=q.Minimizer(prep,system_hook=lambda s,p:add_stereo(s,p,1000.))
    heavy=prep.heavy_ids;info=prep.info;owner=prep.residue[heavy].astype(np.int32)
    atom_names=[prep.atom_names[j] for j in heavy];lookup={(int(r),a):j for j,(r,a) in enumerate(zip(owner,atom_names))}
    bbids=np.asarray([[lookup[i,a] for a in q.BB] for i in range(len(bb))],np.int32)
    remap={int(j):i for i,j in enumerate(heavy)}
    bonds=np.asarray([(remap[a.index],remap[b.index]) for a,b in mini.top.bonds() if a.index in remap and b.index in remap],np.int32).reshape(-1,2)
    inserted={tuple(x) for x in info['inserted_atoms']}
    present=np.array([(int(r),a) not in inserted for r,a in zip(owner,atom_names)],bool)
    arrays=dict(native_coordinates=prep.xyz[heavy].astype(np.float32),source_backbone=bb.astype(np.float32),
        atom_residue=owner,source_present=present,completed=~present,backbone_atoms=bbids,
        segments=np.asarray(segments,np.int32),bonds=bonds)
    meta=dict(parent=parent,source=row,selected=[],status='running',atom_names=atom_names,residue_names=names,residue_keys=keys,
        disulfides=info['disulfides'],capped_singletons=info['capped_singletons'],source_sha256=source_hash,
        augmentation_recipe=RECIPE,reference_view='zero_coordinates',training_enabled=False,
        config=dict(recipe=RECIPE,seed=row['seed'],anm_cutoff_A=15.,platform='CPU',max_iterations=100,
            tolerance_kJ_mol_nm=10.,ca_restraint_kJ_mol_nm2=1000.,stereo_k_kJ_mol=1000.,
            forcefield='amber14-all.xml',nonbonded='CutoffNonPeriodic 1 nm vacuum',openmm=mm.__version__,
            completion='conditional cg2all once per parent; retain existing source heavy atoms',
            minimisation='100-step capped cleanup; convergence reported separately'),
        preparation=dict(source_read_s=source_s,anm_s=anm_s,completion_s=completion_s,
            preparation_s=info['preparation_s'],context_s=mini.context_s,inserted_atoms=info['inserted_atoms']))
    if row['dataset']=='pkpdb':meta['source_defects_sha256']=digest(p.RUNTIME/f"pretraining/pkpdb-full-v1/entries/{row['id']}/defects.json")
    coords={};metrics={};failures=[];attempted=[]
    def repair(c):
        attempted.append(c);t=time.monotonic()
        ca=np.round(bb[:,1] if c==0 else bb[:,1]+AMPLITUDES[(c-1)//8]*delta[(c-1)%8],3)
        try:
            post,detail=mini.run(prep.translated(ca),ca,local/'scratch-min.pdb')
            if detail['energy_after_kJ_mol']>detail['energy_before_kJ_mol']+max(1e-3,abs(detail['energy_before_kJ_mol'])*1e-6):raise ValueError('cleanup raised energy')
            if not np.isfinite(detail['final_rms_force_kJ_mol_nm']):raise ValueError('nonfinite forces')
            persisted=post[heavy].astype(np.float32);check=quality(meta,arrays,persisted)
            metrics[c]=dict(candidate=c,input_amplitude_A=0. if c==0 else AMPLITUDES[(c-1)//8],
                **detail,**check,total_s=time.monotonic()-t)
            coords[c]=persisted
            if args.repeat_zero:
                independent=q.quality(prep,mini,post)
                for key in ('geometry_valid','bond_violations','nonbonded_heavy_pairs_below_1p2_A','chirality_flips','new_cis_nonpro'):
                    if independent[key]!=check[key]:raise ValueError('independent full-precision quality differs: '+key)
            return bool(check['geometry_valid'])
        except (ModuleNotFoundError,ImportError):raise
        except Exception as exc:
            failures.append(dict(candidate=c,error=type(exc).__name__+': '+str(exc),elapsed_s=time.monotonic()-t))
            metrics[c]=dict(candidate=c,geometry_valid=False,error=failures[-1]['error']);return False
    repair(0)
    if 0 not in coords:raise ValueError('zero cleanup did not produce finite coordinates')
    arrays['zero_coordinates']=coords[0]
    selected=[]
    if metrics[0]['geometry_valid']:
        for b,(budget,quota) in enumerate(zip(POOL_SIZES,QUOTAS)):
            valid=sum(repair(c) for c in candidate_ids(b,budget))
            for c in candidate_ids(b)[budget:]:
                if valid>=quota:break
                valid+=repair(c)
        plan=candidate_plan(metrics,replace=True)
        selected=select_quota(plan,{c:x[bbids[:,1]] for c,x in coords.items()},coords[0][bbids[:,1]])
    if args.repeat_zero:
        ca=np.round(bb[:,1],3);repeated,_=mini.run(prep.translated(ca),ca,local/'scratch-repeat.pdb')
        difference=float(np.max(np.abs(repeated[heavy].astype(np.float32)-coords[0])))
        if difference>1e-6:raise ValueError('order-dependent zero cleanup')
        meta['zero_repeat_max_difference_A']=difference
    arrays['coordinates']=np.stack([coords[c] for c in selected]) if selected else np.empty((0,len(heavy),3),np.float32)
    quota_met=tuple(sum(1 for c in selected if (c-1)//8==b) for b in range(3))==QUOTAS
    meta.update(selected=selected,quota_met=quota_met,status='usable' if quota_met else 'partial_quota' if selected else 'zero_control_failed',
        zero_quality=metrics[0],selected_quality={str(c):metrics[c] for c in selected},
        selected_pairwise_ca_rmsd_A=[rmsd(coords[c][bbids[:,1]],coords[d][bbids[:,1]]) for i,c in enumerate(selected) for d in selected[i+1:]],
        attempted_candidates=attempted,candidate_quality={str(c):m for c,m in metrics.items()},failures=failures,
        generation_wall_s=time.monotonic()-started,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    evidence,evidence_meta=upstream_mask(dict(metadata=meta,arrays=arrays),None,None)
    arrays.update(evidence);meta['original_input_evidence']=evidence_meta
    verify_coordinate_record(dict(metadata=meta,arrays=arrays))
    return meta,arrays


def child(args):
    guard(args)
    row=json.loads(args.input.read_text());started=time.monotonic()
    try:meta,arrays=produce(args,row)
    except Exception as exc:
        environment=isinstance(exc,(ImportError,ModuleNotFoundError))
        meta=dict(parent=parent_id(row),source=row,selected=[],quota_met=False,
            status='environment_failed' if environment else 'preparation_failed',error=type(exc).__name__+': '+str(exc),
            traceback=traceback.format_exc()[-4000:],reference_view='zero_coordinates',augmentation_recipe=RECIPE,
            generation_wall_s=time.monotonic()-started,training_enabled=False)
        arrays={}
    frame,_=encode(meta,arrays,zstd_level=1);args.result.write_bytes(frame)


def process_parent(args,row,repeat=False):
    with tempfile.TemporaryDirectory(prefix='ka-anm-',dir=os.environ.get('TMPDIR','/tmp')) as temp:
        local=Path(temp);task=local/'task.json';task.write_text(json.dumps(row))
        result=local/'result.zst';log=local/'child.log'
        command=[sys.executable,'-m','pkatrain.augmentation_generate','child','--input',str(task),'--result',str(result),
            '--pilot',str(args.pilot),'--legacy',str(args.legacy),'--output',str(args.output)]
        if repeat:command.append('--repeat-zero')
        if args.allow_comp1400:command.append('--allow-comp1400')
        with log.open('wb') as handle:
            try:completed=subprocess.run(command,cwd=local,stdout=handle,stderr=subprocess.STDOUT,timeout=args.parent_timeout)
            except subprocess.TimeoutExpired:completed=None
        if completed is not None and completed.returncode==0 and result.exists():
            data=result.read_bytes();record=decode(data,max_output_size=zstandard.frame_content_size(data))
            if record['metadata']['status']=='environment_failed':raise RuntimeError(record['metadata']['traceback'])
            return record
        error='parent timeout' if completed is None else f'child exit {completed.returncode}'
        return dict(metadata=dict(parent=parent_id(row),source=row,selected=[],quota_met=False,status='execution_failed',
            error=error,log_tail=log.read_text(errors='replace')[-3000:],reference_view='zero_coordinates',augmentation_recipe=RECIPE,training_enabled=False),arrays={})


def worker(args):
    guard(args);manifest=json.loads((args.output/'inventory-manifest.json').read_text())
    if digest(Path(__file__))!=manifest['source_snapshot_sha256']:raise ValueError('generator source changed since registration')
    if any(digest(args.legacy/name)!=value for name,value in manifest['legacy_sha256'].items()):raise ValueError('legacy preparation source changed')
    if digest(args.output/'inventory.jsonl')!=manifest['inventory_sha256']:raise ValueError('inventory changed')
    tasks=json.loads((args.output/'tasks.json').read_text())
    if digest(args.output/'tasks.json')!=manifest['tasks_sha256']:raise ValueError('worker assignment changed')
    selected_indices=set(tasks['canary_indices'] if args.canary else tasks['tasks'][args.worker]['indices'])
    rows=[r for r in read_inventory(args.output) if r['index'] in selected_indices]
    dest=args.output/'canary' if args.canary else args.output/'workers'/f'{args.worker:04d}'
    provenance=dict(kind='quota-112-cpu-production-canary' if args.canary else 'quota-112-cpu-production-worker',
        recipe=RECIPE,inventory_manifest_sha256=digest(args.output/'inventory-manifest.json'),
        worker=None if args.canary else args.worker,source_snapshot_sha256=manifest['source_snapshot_sha256'],training_enabled=False)
    if (dest/'manifest.json').exists():
        with AugmentationStore(dest) as old:
            if old.manifest['provenance']!=provenance:raise ValueError('published worker provenance differs')
            old.verify_files()
        return
    started=time.monotonic();counts=Counter();prepared=0
    with ReleaseWriter(dest,provenance,zstd_level=1) as writer:
        counts=Counter(r['summary']['status'] for r in writer.rows)
        prepared=sum(r['summary']['status'] in ('usable','partial_quota','zero_control_failed') for r in writer.rows)
        for row in rows:
            parent=parent_id(row)
            if parent in writer.ids:continue
            t=time.monotonic();record=process_parent(args,row,repeat=args.canary);meta,arrays=record['metadata'],record['arrays']
            if meta['parent']!=parent:raise ValueError('child parent identity changed')
            if arrays:prepared+=1
            counts[meta['status']]+=1
            summary=dict(dataset=row['dataset'],split=row['split'],state=row['state'],n=row['n'],status=meta['status'],selected=len(meta['selected']),quota_met=meta['quota_met'])
            writer.add(parent,meta,arrays,summary=summary)
            if len(writer.rows)%16==0:writer.checkpoint()
            if args.canary or len(writer.rows)%50==0:
                progress=dict(worker=args.worker,completed=len(writer.rows),total=len(rows),counts=dict(counts),
                    wall_s=time.monotonic()-started,last_parent=parent,last_s=time.monotonic()-t)
                atomic_json(dest/'progress.json',progress);print(json.dumps(progress),flush=True)
            if sum(counts.values())>=20 and prepared==0:raise RuntimeError('first twenty sources all failed; inspect worker errors before resuming')
        if args.canary:
            if prepared!=len(rows) or counts['usable']<4:raise RuntimeError('canary preparation/quota checks failed')
            if any(not any(r['summary']['dataset']==dataset and r['summary']['quota_met'] for r in writer.rows) for dataset in ('pkpdb','pinder')):
                raise RuntimeError('canary needs a quota-complete parent from each pool')
        result=writer.finish(expected_ids=[parent_id(r) for r in rows])
    atomic_json(dest/'worker-summary.json',dict(passed=True,records=result['records'],compressed_bytes=result['compressed_bytes'],
        counts=dict(Counter(r['summary']['status'] for r in writer.rows)),wall_s=time.monotonic()-started,
        job=os.environ.get('SLURM_JOB_ID'),host=os.environ.get('SLURMD_NODENAME')))


def finalize(args):
    guard(args);m=json.loads((args.output/'inventory-manifest.json').read_text())
    workers=[args.output/'workers'/f'{i:04d}' for i in range(m['workers'])]
    result=finalize_workers(args.output,workers,dict(kind='pkpdb-pinder-quota-112-coordinate-release-v1',recipe=RECIPE,
        inventory_manifest_sha256=digest(args.output/'inventory-manifest.json'),training_enabled=False),
        expected_ids=[parent_id(r) for r in read_inventory(args.output)])
    counts=Counter();selected=0;replacements=0
    with AugmentationStore(args.output) as store:
        for row in store.rows:
            counts[row['summary']['status']]+=1;selected+=row['summary']['selected']
    atomic_json(args.output/'release-summary.json',dict(passed=True,parents=result['records'],selected_views=selected,
        counts=dict(counts),compressed_bytes=result['compressed_bytes'],raw_bytes=result['raw_bytes'],
        zstd_level=result['zstd_level'],shard_bytes=result['shard_bytes'],files=len([p for p in args.output.rglob('*') if p.is_file()]),
        manifest_sha256=digest(args.output/'manifest.json'),training_enabled=False))
    print(json.dumps(dict(passed=True,parents=result['records'],counts=dict(counts),selected_views=selected)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['register','worker','child','finalize'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--legacy',type=Path,default=Path(__file__).resolve().parents[2]/'legacy')
    p.add_argument('--small-workers',type=int,default=160);p.add_argument('--large-workers',type=int,default=64)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--canary',action='store_true')
    p.add_argument('--allow-comp1400',action='store_true');p.add_argument('--parent-timeout',type=int,default=1200)
    p.add_argument('--repeat-zero',action='store_true');p.add_argument('--input',type=Path);p.add_argument('--result',type=Path)
    args=p.parse_args();globals()[args.command](args)


if __name__=='__main__':main()
