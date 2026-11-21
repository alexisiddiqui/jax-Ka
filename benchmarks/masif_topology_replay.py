"""Native and same-cached-mesh MaSIF replays; diagnostic artifacts only."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
from scipy.io import savemat
from scipy.spatial import cKDTree
from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.augmentation_store import AugmentationStore,ReleaseWriter
from masif_pkpdb_recovery import POLICY,membership,write_inventory,source_hashes
from masif_preparation_compare import frame,compare_arrays
from masif_topology_audit import topology
from masif_paper_batch import run_process,matlab_literal


def prepare(r):
    panel=json.loads((r/'topology/panel.json').read_text());runtime=r.parent.parent/'pkabench'
    rows,hashes=membership(runtime);members={row['id']:row for row in rows};units=[]
    for case in panel['cases']:
        entry=case['entry'];old=entry['summary'];m=members[old['cid']]
        units.append(dict(old,unit_policy=POLICY,group=m['group'],expected_chains=int(m['n_chains']),
            source_files_sha256=source_hashes(runtime,old['cid']),recovery_reason='topology_or_preparation_diagnostic',panel_reason=case['panel_reason']))
    write_inventory(r/'topology/native-inventory.json',units,dict(membership_sha256=hashes,unit_policy=POLICY,
        source='topology and terminal-preparation diagnostic panel',source_panel_sha256=digest(r/'topology/panel.json')))
    flagged=[c for c in panel['cases'] if c.get('nonmanifold_edges',0)]
    chosen={}
    ordered=sorted(flagged,key=lambda c:(c['nonmanifold_edge_fraction'],c['id']))
    if ordered:
        for i in sorted(set([0,len(ordered)//4,len(ordered)//2,3*len(ordered)//4,len(ordered)-1])):chosen[ordered[i]['id']]=ordered[i]
        case=max(flagged,key=lambda c:c['nonmanifold_edges']);chosen[case['id']]=case
    controls=[c for c in panel['cases'] if c['id'] in panel['healthy_controls']]
    for case in controls[:2]:chosen[case['id']]=case
    atomic_json(r/'topology/cached-repeat-panel.json',dict(cases=list(chosen.values()),repeats=2,
        limitations=['Cached vertices, normals and raw chemical channels were stored as float32',
            'Curvature/normals/charts are regenerated on exactly those cached vertices/faces',
            'Charge times 3 and hydrophobicity times 4.5 reproduce the saved normalized channels; clipped potentials are not claimed to be recovered']))


def cached_replays(r):
    b=r.parent;paper=b/'masif-paper-runtime-v1';masif=b/'masif-audit-v1';code=Path(__file__).resolve().parent
    out=r/'topology/cached-replays';out.mkdir(exist_ok=False);panel=json.loads((r/'topology/cached-repeat-panel.json').read_text())
    provenance=dict(kind='diagnostic-cached-mesh-replay-v1',source_panel_sha256=digest(r/'topology/cached-repeat-panel.json'),
        bridge_sha256=digest(code/'masif_paper_bridge.py'),paper_runtime_sha256=digest(paper/'manifest.json'))
    reports=[];args=SimpleNamespace(batch_size=1,output=out);started=time.monotonic()
    with ReleaseWriter(out/'fingerprints',provenance,zstd_level=1) as writer:
        for case in panel['cases']:
            old=frame(case['entry']);a=old['arrays']
            for repeat in range(2):
                rid=case['id']+'/cached-repeat-%d'%repeat;log=out/('case-%03d.log'%len(reports));began=time.monotonic()
                row=dict(id=rid,old_id=case['id'],repeat=repeat,status='failed')
                try:
                    with tempfile.TemporaryDirectory(prefix='masif-topology-',dir='/tmp') as directory:
                        work=Path(directory)
                        savemat(work/'surface.mat',dict(vertices=a['vertices'].astype(float),faces=a['faces'],charge=a['raw_features'][:,2].astype(float)*3,
                            hbond=a['raw_features'][:,1].astype(float),hydrophobicity=a['raw_features'][:,3].astype(float)*4.5))
                        np.savez(work/'surface.npz',vertices=a['vertices'],faces=a['faces'],owner=a['owner'])
                        atomic_json(work/'surface-receipt.json',dict(status='complete',id=rid,timings={},surface_policy='cached-mesh-replay-v1'))
                        job=dict(id=rid,work='/audit',host_work=str(work),output='/audit/features.npz',receipt='/audit/receipt.json',
                            canonical_residue_ids=a['canonical_residue_ids'].tolist(),paper_runtime_sha256=digest(paper/'manifest.json'),
                            patch_diagnostics='/audit/patch-diagnostics.json')
                        atomic_json(work/'jobs.json',[job]);env=dict(os.environ,UDOCKER_DIR=str(masif/'udocker'))
                        expression='addpath('+matlab_literal(code)+');masif_paper_matlab('+matlab_literal(work/'jobs.json')+','+matlab_literal(paper)+');'
                        run_process(['/usr/local/MATLAB/R2021b/bin/matlab','-singleCompThread','-batch',expression],args,0,'matlab',log,env)
                        cmd=[str(masif/'setup-env/bin/udocker'),'--quiet','run','--env=LD_LIBRARY_PATH=/compat','--volume='+str(masif/'compat')+':/compat',
                            '--env=OMP_NUM_THREADS=1','--env=OPENBLAS_NUM_THREADS=1','--env=MKL_NUM_THREADS=1','--env=CUDA_VISIBLE_DEVICES=',
                            '--env=PYTHONPATH=/code','--env=PYTHONDONTWRITEBYTECODE=1','--volume='+str(work)+':/audit','--volume='+str(code)+':/code',
                            '--volume='+str(paper)+':/paper-runtime','--volume='+str(b/'masif-checkpoints/active')+':/checkpoint',
                            '--env=MASIF_CHECKPOINT_PREFIX=/checkpoint/model','--volume='+str(r/'spool')+':/gpu-spool',
                            'masif-audit','python','/code/masif_paper_bridge.py','describe','--jobs','/audit/jobs.json','--runtime','/paper-runtime','--gpu-spool','/gpu-spool']
                        run_process(cmd,args,0,'cached_descriptors',log,env)
                        receipt=json.loads((work/'receipt.json').read_text());row['receipt']=receipt
                        if receipt['status']!='complete':raise ValueError(receipt.get('error','cached replay failed'))
                        diag=json.loads((work/'patch-diagnostics.json').read_text());support=np.array(diag.pop('support'))
                        _,distance=topology(a['vertices'],a['faces']);near=np.isfinite(distance)
                        def support_stats(mask):
                            x=support[mask];return dict(centers=len(x),below_three=int((x<3).sum()),median=float(np.median(x)) if len(x) else None,min=int(x.min()) if len(x) else None)
                        row['patch_diagnostics']=dict(diag,near_nonmanifold=support_stats(near),far_from_nonmanifold=support_stats(~near))
                        with np.load(work/'features.npz',allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
                        writer.add(rid,dict(case=case['id'],repeat=repeat,receipt=receipt),arrays,summary=dict(old_id=case['id'],repeat=repeat))
                        row['status']='complete'
                except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
                row['wall_s']=time.monotonic()-began;reports.append(row);writer.checkpoint()
                atomic_json(out/'progress.json',dict(completed=len(reports),total=2*len(panel['cases']),counts=dict(Counter(x['status'] for x in reports)),wall_s=time.monotonic()-started))
                atomic_json(out/'report.json',dict(cases=reports,status='running'))
        writer.finish(expected_ids=[x['id'] for x in reports if x['status']=='complete'])
    atomic_json(out/'report.json',dict(cases=reports,status='complete' if all(x['status']=='complete' for x in reports) else 'held',wall_s=time.monotonic()-started))


def local_comparison(a,b):
    result=compare_arrays(a,b,np.empty((0,3)));_,distance=topology(a['vertices'],a['faces']);near=np.isfinite(distance)
    d,ix=cKDTree(b['vertices']).query(a['vertices']);same=a['canonical_residue_ids'][a['owner']]==b['canonical_residue_ids'][b['owner'][ix]]
    norms=np.linalg.norm(a['descriptors'],axis=1)*np.linalg.norm(b['descriptors'][ix],axis=1);valid=(d<=.5)&same&(norms>1e-12)
    cosine=np.sum(a['descriptors']*b['descriptors'][ix],axis=1)/np.maximum(norms,1e-12)
    for key,mask in [('near_nonmanifold_12A',near),('far_from_nonmanifold_12A',~near)]:
        values=cosine[valid&mask];result[key]=dict(vertices=int(mask.sum()),matched=int(len(values)),
            matched_fraction=float((valid&mask).sum()/mask.sum()) if mask.any() else None,
            median_cosine=float(np.median(values)) if len(values) else None,p05_cosine=float(np.percentile(values,5)) if len(values) else None)
    return result


def report(r):
    panel=json.loads((r/'topology/panel.json').read_text());results=[]
    with AugmentationStore(r/'topology/native-replay/fingerprints') as native,AugmentationStore(r/'topology/cached-replays/fingerprints') as cached:
        for case in panel['cases']:
            rid=case['id'];old=frame(case['entry'])['arrays'];row=dict(id=rid,panel_reason=case['panel_reason'])
            row['cached_topology']=topology(old['vertices'],old['faces'])[0]
            if rid in native.by_id:
                new=native.raw(rid);row['old_to_current_native']=local_comparison(old,new['arrays'])
                row['current_native_topology']=topology(new['arrays']['vertices'],new['arrays']['faces'])[0]
                row['current_native_preparation']=dict(added_terminal_oxygen_canonical_ids=new['metadata']['source']['added_terminal_oxygen_canonical_ids'],
                    mesh_path=new['metadata']['receipt'].get('mesh_path'))
            else:row['native_replay_status']='failed; retained in native report'
            for repeat in range(2):
                key=rid+'/cached-repeat-%d'%repeat
                if key in cached.by_id:row['old_to_cached_repeat_%d'%repeat]=local_comparison(old,cached.raw(key)['arrays'])
            if all(rid+'/cached-repeat-%d'%i in cached.by_id for i in range(2)):
                row['cached_repeat_0_to_1']=local_comparison(cached.raw(rid+'/cached-repeat-0')['arrays'],cached.raw(rid+'/cached-repeat-1')['arrays'])
            results.append(row)
    atomic_json(r/'topology/replay-comparison.json',dict(status='complete',cases=results,
        limitations=['Diagnostic stratified/extreme panel, not a random estimate of all flagged records',
            'Native replays combine current terminal preparation, remeshing and chart variability',
            'Current cleanup does not guarantee a manifold surface: old/new topology is reported, and agreement alone cannot certify a shared defect',
            'Same-cached-mesh repeats isolate chart/curvature recomputation on saved float32 geometry with fixed saved chemical fields',
            'Near/far masks use 12 A edge-graph distance and same-residue nearest matches within 0.5 A; unmatched fractions are reported']))


def main(r):
    r=Path(r);prepare(r);b=r.parent
    cmd=[sys.executable,str(Path(__file__).with_name('masif_annotation_pilot.py')),'--native-unit-inventory',str(r/'topology/native-inventory.json'),
        '--runtime',str(b.parent/'pkabench'),'--masif',str(b/'masif-audit-v1'),'--paper-runtime',str(b/'masif-paper-runtime-v1'),
        '--checkpoint',str(b/'masif-checkpoints/active/model'),'--dataset','pkpdb','--full','--workers','1','--batch-size','4',
        '--gpu-spool',str(r/'spool'),'--output',str(r/'topology/native-replay')]
    result=subprocess.run(cmd)
    if not (r/'topology/native-replay/report.json').exists():raise RuntimeError('native topology pilot did not finish')
    cached_replays(r);report(r)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True);main(a.root)
