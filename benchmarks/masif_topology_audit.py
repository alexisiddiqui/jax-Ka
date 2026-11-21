"""Topology census and deterministic replay panel from immutable cached meshes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from pkabench.runtime import atomic_json,digest,require_compute
from masif_preparation_compare import frame


def topology(vertices,faces,radius=12.):
    edges=np.sort(np.concatenate([faces[:,:2],faces[:,1:],faces[:,[2,0]]]),axis=1)
    edges,count=np.unique(edges,axis=0,return_counts=True)
    bad=edges[count>2];boundary=edges[count==1];seeds=np.unique(bad)
    # Mirror the author's directed-edge selection: geodesic_dists_graph
    # transposes unique directed face edges, then retains row < column.
    # All edges survive on consistently oriented closed meshes; boundary or
    # inconsistent orientation can remove edges from the chart graph.
    directed=np.concatenate([faces[:,:2],faces[:,1:],faces[:,[2,0]]])
    selected=np.unique(np.sort(directed[directed[:,0]>directed[:,1]],axis=1),axis=0)
    lengths=np.linalg.norm(vertices[selected[:,0]]-vertices[selected[:,1]],axis=1)
    graph=coo_matrix((np.r_[lengths,lengths],(np.r_[selected[:,0],selected[:,1]],np.r_[selected[:,1],selected[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
    distance=(dijkstra(graph,indices=seeds,directed=False,min_only=True,limit=radius) if len(seeds) else np.full(len(vertices),np.inf))
    result=dict(vertices=len(vertices),faces=len(faces),edges=len(edges),nonmanifold_edges=len(bad),boundary_edges=len(boundary),
        max_edge_face_incidence=int(count.max()),nonmanifold_edge_fraction=float(len(bad)/len(edges)),
        author_graph_missing_undirected_edges=int(len(edges)-len(selected)),nonmanifold_vertices=len(seeds),vertices_within_12A_graph=int(np.isfinite(distance).sum()),
        fraction_within_12A_graph=float(np.isfinite(distance).mean()))
    return result,distance


def inspect(entry):
    try:
        a=frame(entry)['arrays'];metrics,_=topology(a['vertices'],a['faces'])
        return dict(id=entry['id'],status='complete',n=entry['summary']['n'],**metrics)
    except Exception as exc:return dict(id=entry['id'],status='failed',error=str(exc))


def select_panel(rows,records,catalog):
    safe={r['id'] for r in records if r['status']=='audited' and not r.get('hard_rebuild_reasons') and r.get('preparation_compatibility')!='terminal_geometry_unresolved' and not r.get('intact_antibody_regrouping') and not r.get('superseded_by_grouped_abag')}
    eligible=sorted([r for r in rows if r['status']=='complete' and r['id'] in safe and r['nonmanifold_edges']],key=lambda r:(r['nonmanifold_edge_fraction'],r['id']))
    selected={}
    # Four severity quartiles crossed with four size bins. Include maxima
    # separately; the panel is diagnostic, not a population error estimate.
    ranks={r['id']:i/max(1,len(eligible)-1) for i,r in enumerate(eligible)}
    for severity in range(4):
        for size,(lo,hi) in enumerate([(0,150),(150,300),(300,600),(600,10**9)]):
            options=[r for r in eligible if lo<r['n']<=hi and min(3,int(ranks[r['id']]*4))==severity]
            if options:
                row=options[len(options)//2];selected[row['id']]=dict(row,panel_reason='severity-%d-size-%d'%(severity,size))
    for metric in ['nonmanifold_edges','nonmanifold_edge_fraction','fraction_within_12A_graph']:
        if eligible:
            row=max(eligible,key=lambda r:(r[metric],r['id']));selected[row['id']]=dict(row,panel_reason='maximum_'+metric)
    clean=sorted([r for r in records if r['id'] in safe and not r.get('review_reasons')],key=lambda r:r['id'])
    controls=[]
    for lo,hi in [(0,150),(150,300),(300,600),(600,10**9)]:
        options=[r for r in clean if lo<catalog[r['id']]['summary']['n']<=hi]
        if options:
            row=options[len(options)//2];selected[row['id']]=dict(id=row['id'],panel_reason='healthy_size_control',n=catalog[row['id']]['summary']['n']);controls.append(row['id'])
    # Explicitly stress preparation policy where the first 42-case comparison
    # was least representative: many newly exposed fragment termini.
    terminal=sorted([r for r in records if r['id'] in safe and r.get('new_policy_added_OXT_canonical_ids')],
        key=lambda r:(-len(r['new_policy_added_OXT_canonical_ids']),r['id']))
    for r in terminal[:6]:selected.setdefault(r['id'],dict(id=r['id'],n=catalog[r['id']]['summary']['n'],panel_reason='fragmented_terminal_stress'))
    return [dict(row,entry=catalog[rid]) for rid,row in sorted(selected.items())],controls


def run(audit,output,workers):
    audit=Path(audit);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    catalog={r['id']:r for r in json.loads((audit/'full/catalog.json').read_text())['entries']}
    records=[json.loads(line) for line in (audit/'full/records.jsonl').read_text().splitlines()]
    candidates=[catalog[r['id']] for r in records if r.get('review_reasons')];rows=[];start=time.monotonic()
    with ProcessPoolExecutor(workers) as pool:
        for row in pool.map(inspect,candidates,chunksize=8):
            rows.append(row)
            if len(rows)%100==0:atomic_json(output/'progress.json',dict(audited=len(rows),total=len(candidates),wall_s=time.monotonic()-start))
    atomic_json(output/'census.json',dict(records=rows,status='complete' if all(r['status']=='complete' for r in rows) else 'held',
        source_audit_sha256=digest(audit/'full/records.jsonl'),wall_s=time.monotonic()-start,
        locality='12 Angstrom shortest edge-path distance; matches author chart graph radius, not a Euclidean ball'))
    if any(r['status']!='complete' for r in rows):raise ValueError('topology census has unresolved errors')
    panel,controls=select_panel(rows,records,catalog)
    atomic_json(output/'panel.json',dict(cases=panel,healthy_controls=controls,
        design='severity quartile x residue-size bin, extrema, healthy controls, fragmented-terminal stress cases',
        source_census_sha256=digest(output/'census.json')))
    metrics={}
    for key in ['nonmanifold_edges','nonmanifold_edge_fraction','fraction_within_12A_graph']:
        values=[r[key] for r in rows];metrics[key]={str(q):float(np.percentile(values,q)) for q in [0,25,50,75,95,99,100]}
    atomic_json(output/'summary.json',dict(status='census_complete',records=len(rows),panel_cases=len(panel),metrics=metrics,
        original_confirmed_geometry_failures=sum(bool(r.get('hard_rebuild_reasons')) for r in records if r.get('review_reasons')),
        limitations=['Affected-neighbourhood fraction is a locality diagnostic, not the fraction of incorrect fingerprints',
                    'Native replay can change remeshing and terminal preparation; cached-mesh repeats are needed to separate numerical variation']))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True);run(a.audit,a.output,a.workers)
