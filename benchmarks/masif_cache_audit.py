"""Read-only audit of old pKPDB fingerprints; geometry and policy are separate.

No descriptors or source structures are modified. A valid legacy record is not
declared numerically corrupt merely because the new policy adds terminal OXT.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.spatial import ConvexHull
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.augmentation_store import AugmentationStore,decode
from masif_annotation_pilot import export_native
from masif_surface_quality import complete_terminal_oxygens


def inspect_arrays(a):
    hard=[];review=[];metrics={}
    required={'vertices','faces','normals','owner','raw_features','descriptors','canonical_residue_ids'}
    if not required<=set(a):return ['missing_array_fields'],review,metrics
    if not all(np.isfinite(x).all() for x in a.values()):return ['nonfinite_array'],review,metrics
    v=a['vertices'];f=a['faces'];n=len(v);nr=len(a['canonical_residue_ids'])
    if v.shape!=(n,3) or a['descriptors'].shape!=(n,80) or a['normals'].shape!=(n,3) or a['raw_features'].shape!=(n,4):
        return ['array_dimensions'],review,metrics
    if not n or not nr or a['owner'].shape!=(n,) or a['owner'].min()<0 or a['owner'].max()>=nr:
        return ['owner_mapping'],review,metrics
    if f.ndim!=2 or f.shape[1]!=3 or not len(f) or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=n:
        return ['face_indices'],review,metrics
    tri=v[f].astype(float)
    repeated=(f[:,0]==f[:,1])|(f[:,0]==f[:,2])|(f[:,1]==f[:,2])
    area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    _,counts=np.unique(np.sort(f,axis=1),axis=0,return_counts=True)
    metrics.update(vertices=n,faces=len(f),residues=nr,repeated_index_faces=int(repeated.sum()),
        zero_area_faces=int((area2<=1e-10).sum()),duplicate_faces=int(np.maximum(counts-1,0).sum()))
    for name in ['repeated_index_faces','zero_area_faces','duplicate_faces']:
        if metrics[name]:hard.append(name)
    edges=np.sort(np.concatenate([f[:,:2],f[:,1:],f[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    metrics.update(boundary_edges=int((counts==1).sum()),nonmanifold_edges=int((counts>2).sum()),
                   isolated_vertices=int(n-len(np.unique(f))))
    graph=coo_matrix((np.ones(len(unique)),(unique[:,0],unique[:,1])),shape=(n,n))
    count,labels=connected_components(graph,directed=False);sizes=np.bincount(labels,minlength=count)
    metrics['component_sizes']=sorted(sizes.tolist(),reverse=True)
    metrics['tiny_component_vertices']=int(sizes[sizes<4].sum())
    if metrics['isolated_vertices']:hard.append('isolated_vertices')
    if metrics['tiny_component_vertices']:hard.append('tiny_components')
    # Boundaries can also be introduced by the current cleanup policy; they
    # require review rather than an automatic blanket rebuild decision.
    if metrics['boundary_edges']:review.append('boundary_edges')
    if metrics['nonmanifold_edges']:review.append('nonmanifold_edges')
    return hard,review,metrics


def extent_metrics(vertices,atoms):
    bbox=((atoms<vertices.min(axis=0)-2.)|(atoms>vertices.max(axis=0)+2.)).any(axis=1)
    hull=ConvexHull(vertices)
    outside=0
    for start in range(0,len(atoms),256):
        distance=atoms[start:start+256]@hull.equations[:,:3].T+hull.equations[:,3]
        outside+=int((distance.max(axis=1)>2.).sum())
    return dict(source_heavy_atoms=len(atoms),atoms_outside_bbox_plus_2A=int(bbox.sum()),
                atoms_outside_convex_hull_plus_2A=outside)


def initialize_worker(runtime,cpus):
    global RUNTIME,LOCAL
    RUNTIME=Path(runtime);LOCAL=tempfile.TemporaryDirectory(prefix='masif-cache-audit-',dir='/tmp')
    # Cache actual source parses across sibling chains; values are unchanged.
    import pkatrain.production_sidechains as side
    import pkatrain.production_inputs as inputs
    side._pkpdb_atoms=lru_cache(maxsize=1)(side._pkpdb_atoms)
    inputs._residues=lru_cache(maxsize=1)(inputs._residues)
    # Keep each worker single threaded for numeric kernels, with two CPU slots.
    identity=int(__import__('multiprocessing').current_process()._identity[-1])-1
    index=identity%(len(cpus)//2);os.sched_setaffinity(0,set(cpus[2*index:2*index+2]))


def inspect_parent(task):
    cid,entries,role_info=task;rows=[]
    try:
        from pkatrain.production_sidechains import _pkpdb_atoms
        from pkatrain.production_inputs import _residues
        bb,atoms,identity=_pkpdb_atoms(RUNTIME,cid);other,chain,other_identity=_residues('pkpdb',RUNTIME,cid)
        if not np.array_equal(identity,other_identity) or not np.allclose(bb,other,atol=.001,rtol=0):
            raise ValueError('native training source order differs')
        regroup=[]
        if role_info and role_info['status']=='antibody_only':
            from masif_pkpdb_abag import contact_components
            groups,_=contact_components(atoms,chain,len(role_info['role_evidence']))
            regroup=[g for g in groups if len(g)>1]
    except Exception as e:
        return [dict(id=x['id'],cid=cid,status='source_unverified',error=str(e)) for x in entries]
    for entry in entries:
        row=dict(id=entry['id'],cid=cid,status='audited',hard_rebuild_reasons=[],review_reasons=[],
                 superseded_by_grouped_abag=bool(entry['superseded_by_grouped_abag']),
                 intact_antibody_regrouping=regroup)
        try:
            with Path(entry['path']).open('rb') as stream:
                stream.seek(entry['offset']);frame=stream.read(entry['bytes'])
            if hashlib.sha256(frame).hexdigest()!=entry['sha256']:raise ValueError('compressed frame hash mismatch')
            record=decode(frame,max_output_size=entry['raw_bytes']);a=record['arrays'];m=record['metadata']
            if m['parent']!=entry['id']:raise ValueError('record identity mismatch')
            hard,review,metrics=inspect_arrays(a);row['hard_rebuild_reasons']+=hard;row['review_reasons']+=review;row['mesh']=metrics
            if m['receipt']['checkpoint_sha256']!=entry['checkpoint_sha256']:row['hard_rebuild_reasons'].append('model_profile_mismatch')
            item=entry['summary'];canonical=np.flatnonzero(chain==item['chain_index']).astype(np.int32)
            if not np.array_equal(canonical,a['canonical_residue_ids']):row['hard_rebuild_reasons'].append('canonical_mapping')
            original=Path(LOCAL.name)/'input.pdb';src=export_native(RUNTIME,item,original,surface_policy=None)
            if src['pdb_sha256']!=m['source']['pdb_sha256']:row['hard_rebuild_reasons'].append('native_input_PDB_hash')
            points=np.asarray([p for i in canonical for name,p in atoms[int(i)].items() if name[0] not in ('H','D')],float)
            if not hard:
                metrics.update(extent_metrics(a['vertices'],points))
                if metrics['atoms_outside_bbox_plus_2A']:row['hard_rebuild_reasons'].append('source_bbox_coverage')
                if metrics['atoms_outside_convex_hull_plus_2A']:row['hard_rebuild_reasons'].append('source_convex_hull_coverage')
            selected=[atoms[int(i)] for i in canonical];backbone=bb[canonical];ch=chain[canonical]
            segments=np.zeros(len(canonical),np.int32)
            for i in range(1,len(canonical)):
                segments[i]=segments[i-1]+int(ch[i]!=ch[i-1] or np.linalg.norm(backbone[i-1,2]-backbone[i,0])>2.)
            try:
                _,added=complete_terminal_oxygens(selected,segments)
                row['new_policy_added_OXT_canonical_ids']=[int(canonical[i]) for i in added]
                row['preparation_compatibility']='requires_terminal_policy_review' if added else 'no_terminal_addition_required'
            except ValueError as e:
                row['preparation_compatibility']='terminal_geometry_unresolved';row['terminal_error']=str(e)
            row['hard_rebuild_reasons']=sorted(set(row['hard_rebuild_reasons']))
        except Exception as e:
            row['status']='audit_error';row['error']=str(e)
        rows.append(row)
    return rows


def catalog(old,grouped,output):
    old=Path(old);grouped=Path(grouped);parents=json.loads((grouped/'inventory/inventory.json').read_text())['parents']
    roles={r['id']:r for r in parents};replacement=set()
    for u in json.loads((grouped/'inventory/inventory.json').read_text())['units']:
        if u['split']=='train':replacement.update('pkpdb/'+u['cid']+'/chain-%04d'%i for i in u['chain_indices'])
    sources=sorted((old/'release/nodes').glob('*/fingerprints'))+[old/'release/recovery/queue-v1/fingerprints']
    entries=[];seen=set();manifests={}
    for source in sources:
        with AugmentationStore(source) as store:
            manifests[str(source)]=digest(source/'manifest.json')
            checkpoint=store.manifest['provenance']['checkpoint_sha256']
            for r in store.rows:
                if r['id'] in seen:raise ValueError('duplicate successful legacy record')
                seen.add(r['id']);entries.append(dict(r,path=str(source/r['shard']),
                    checkpoint_sha256=checkpoint,superseded_by_grouped_abag=r['id'] in replacement))
    if len(entries)!=100938:raise ValueError('legacy success inventory differs from completion audit')
    atomic_json(output/'catalog.json',dict(entries=entries,source_manifests=manifests,roles=roles,
        scope='all 100938 numerically successful native pKPDB training chain records; no source/cache mutation'))


def audit(runtime,root,workers):
    root=Path(root);data=json.loads((root/'catalog.json').read_text());path=root/'records.jsonl';done=set();position=0
    if path.exists():
        with path.open('rb') as stream:
            for line in stream:
                try:done.add(json.loads(line)['id']);position=stream.tell()
                except ValueError:break
        with path.open('r+b') as stream:stream.truncate(position)
    groups=defaultdict(list)
    for row in data['entries']:
        if row['id'] not in done:groups[row['summary']['cid']].append(row)
    tasks=[(cid,rows,data['roles'].get(cid)) for cid,rows in sorted(groups.items())]
    began=time.monotonic();cpus=sorted(os.sched_getaffinity(0));count=len(done)
    with path.open('a') as stream,ProcessPoolExecutor(workers,initializer=initialize_worker,initargs=(runtime,cpus)) as pool:
        for result in pool.map(inspect_parent,tasks,chunksize=8):
            for row in result:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');count+=1
            stream.flush()
            if count%200<8:atomic_json(root/'progress.json',dict(audited=count,total=len(data['entries']),wall_s=time.monotonic()-began))
    rows=[json.loads(line) for line in path.read_text().splitlines()];ids={r['id'] for r in rows}
    if ids!={r['id'] for r in data['entries']} or len(ids)!=len(rows):raise ValueError('audit coverage incomplete/duplicated')
    redo=[r for r in rows if r.get('hard_rebuild_reasons')];review=[r for r in rows if r.get('review_reasons')]
    summary=dict(status='audit_complete',records=len(rows),statuses=dict(Counter(r['status'] for r in rows)),
        hard_rebuild_records=len(redo),hard_rebuild_reasons=dict(Counter(reason for r in redo for reason in r['hard_rebuild_reasons'])),
        topology_review_records=len(review),preparation_compatibility=dict(Counter(r.get('preparation_compatibility','unverified') for r in rows)),
        superseded_chain_records=sum(r.get('superseded_by_grouped_abag',False) for r in rows),
        additional_intact_antibody_parent_ids=sorted({r['cid'] for r in rows if r.get('intact_antibody_regrouping')}),
        wall_s=time.monotonic()-began,catalog_sha256=digest(root/'catalog.json'),results_sha256=digest(path),
        limitations=['Convex-hull/source-extent checks detect exterior surface loss, not every missing concave patch',
                    'Full raw-MSMS coverage is not reconstructed for every legacy record',
                    'Policy mismatch alone is not counted as numerical corruption; equivalence needs paired preparation controls',
                    'Boundary/nonmanifold edge counts are review flags because the current cleanup also permits boundaries'])
    atomic_json(root/'summary.json',summary);atomic_json(root/'hard-rebuild.json',redo)
    atomic_json(root/'review.json',review);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['catalog','audit'])
    p.add_argument('--old',type=Path);p.add_argument('--grouped',type=Path);p.add_argument('--runtime',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True)
    a.output.mkdir(parents=True,exist_ok=True)
    if a.command=='catalog':catalog(a.old,a.grouped,a.output)
    else:audit(a.runtime,a.output,a.workers)
