"""Corresponding-patch, feature, and discrimination measurements for frozen MaSIF."""
import csv
import json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.augmentation_store import AugmentationStore
from augmentation_surface_audit import fit,rms


def view(record,role):
    prefix=record['metadata']['roles'][role]+'_'
    return {k[len(prefix):]:v for k,v in record['arrays'].items() if k.startswith(prefix)}


def weights(rows):
    frames=defaultdict(set);counts=Counter((r['parent'],r['role']) for r in rows)
    for r in rows:frames[r['parent']].add(r['role'])
    return np.array([1/(len(frames)*len(frames[r['parent']])*counts[r['parent'],r['role']]) for r in rows])


def quantile(rows,key):
    rows=[r for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    if not rows:return None
    x=np.array([r[key] for r in rows]);w=weights(rows);order=np.argsort(x)
    return dict(zip(('p05','median','p95'),map(float,np.interp([.05,.5,.95],np.cumsum(w[order])-.5*w[order],x[order]))))


def bootstrap_margin(rows,key):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r.get(key) is not None:groups[r['parent']][r.get('role','one')].append(r[key])
    if not groups:return None
    values=np.array([np.mean([np.median(v) for v in views.values()]) for views in groups.values()]);rng=np.random.default_rng(20261013)
    samples=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
    return dict(proteins=len(values),mean_parent_view_median=float(values.mean()),
                bootstrap_95_interval=list(map(float,np.quantile(samples,[.025,.975]))))


def collect(records):
    gallery=[]
    for record in records:
        m=record['metadata'];a=record['arrays'];zero=view(record,'zero')
        for i in np.flatnonzero(zero['anchor_valid']):
            ids=zero['patch_indices'][i][zero['mask'][i].astype(bool)]
            gallery.append(dict(parent=m['parent'],group=m['source_group'],anchor=int(i),descriptor=zero['descriptors'][i],
                chemistry=zero['features'][i,zero['mask'][i].astype(bool),2:].mean(0),support=set(map(int,ids))))
    rows=[];coverage=[];controls=[]
    for record in records:
        m,a=record['metadata'],record['arrays'];bb=a['backbone_atoms'];owner=a['atom_residue']
        comparisons=[('repair_only','native','zero'),('historical_zero_drift','zero','historical_zero')]
        comparisons += [('selected_vs_zero','zero',r) for r in m['roles'] if r.startswith('selected_')]
        comparisons += [('selected_vs_native','native',r) for r in m['roles'] if r.startswith('selected_')]
        comparisons += [(r+'_vs_zero','historical_zero',r) for r in m['roles'] if r.startswith('amplitude_')]
        comparisons += [('rigid_rebuild_control','zero','rigid_zero')] if 'rigid_zero' in m['roles'] else []
        comparisons += [('seed_repeat_control','zero','repeat_zero')] if 'repeat_zero' in m['roles'] else []
        for role in m['roles']:
            v=view(record,role)
            coverage.append(dict(parent=m['parent'],dataset=m['dataset'],role=role,total=len(v['anchor_valid']),mapped=int(v['anchor_valid'].sum())))
        zero=view(record,'zero')
        if 'rigid_mesh_descriptor_distances' in zero:
            controls.append(dict(parent=m['parent'],kind='rigid_fixed_mesh',median_distance=float(np.median(zero['rigid_mesh_descriptor_distances'])),p95_distance=float(np.quantile(zero['rigid_mesh_descriptor_distances'],.95))))
        for group,refrole,role in comparisons:
            ref,v=view(record,refrole),view(record,role)
            ca=ref['coordinates'][bb[:,1]];moving=v['coordinates'][bb[:,1]]
            gr,gt=fit(moving,ca);global_rms=rms(moving@gr+gt,ca)
            d1=cKDTree(v['vertices']@gr+gt).query(ref['vertices'])[0]
            d2=cKDTree(ref['vertices']).query(v['vertices']@gr+gt)[0]
            whole_distance=float(.5*(np.average(d1,weights=ref['vertex_area'])+np.average(d2,weights=v['vertex_area'])))
            common=np.flatnonzero(ref['anchor_valid']&v['anchor_valid'])
            for i in common:
                rid=ref['patch_indices'][i][ref['mask'][i].astype(bool)]
                vid=v['patch_indices'][i][v['mask'][i].astype(bool)]
                members=np.unique(ref['owner'][rid])
                if len(members)<4 or np.linalg.matrix_rank(ca[members]-ca[members].mean(0))<2:continue
                lr,lt=fit(moving[members],ca[members])
                rp=ref['vertices'][rid];vp=v['vertices'][vid]@lr+lt
                rd,ri=cKDTree(vp).query(rp);vd,vi=cKDTree(rp).query(vp)
                rw,vw=ref['vertex_area'][rid],v['vertex_area'][vid]
                fref=ref['features'][i,ref['mask'][i].astype(bool)]
                fvar=v['features'][i,v['mask'][i].astype(bool)]
                feature_rmse=np.sqrt(.5*(np.average((fref-fvar[ri])**2,axis=0,weights=rw)+np.average((fvar-fref[vi])**2,axis=0,weights=vw)))
                descriptor=ref['descriptors'][i];query=v['descriptors'][i]
                dist=float(np.linalg.norm(query-descriptor));den=float(np.linalg.norm(query)*np.linalg.norm(descriptor))
                row=dict(parent=m['parent'],dataset=m['dataset'],split=m['split'],sample_kind=m['sample_kind'],size_bin=m['size_bin'],
                    completed_fraction=float(a['completed'].mean()),
                    group=group,reference=refrole,role=role,anchor=int(a['anchor_residue'][i]),
                    anchor_key=json.dumps(m['residue_keys'][int(a['anchor_residue'][i])]),
                    descriptor_L2=dist,descriptor_cosine=float(np.dot(query,descriptor)/den) if den else None,
                    global_ca_rmsd_A=global_rms,global_molecular_surface_distance_A=whole_distance,
                    local_ca_rmsd_A=rms(moving[members]@lr+lt,ca[members]),
                    local_heavy_rmsd_A=rms(v['coordinates'][np.isin(owner,members)]@lr+lt,ref['coordinates'][np.isin(owner,members)]),
                    local_molecular_surface_distance_A=float(.5*(np.average(rd,weights=rw)+np.average(vd,weights=vw))),
                    surface_point_coverage_below_2p5_A=float(.5*(np.average(rd<=2.5,weights=rw)+np.average(vd<=2.5,weights=vw))),
                    effective_geodesic_radius_A=float(ref['rho'][i].max()),
                    patch_vertices=int(len(rid)))
                for k,val in zip(('shape_index','curvature','hbond','electrostatics','hydropathy'),feature_rmse):row[k+'_rmse']=float(val)
                # Gallery is repaired zero; identity distances above use the
                # requested reference independently. Retrieval metrics are
                # explicitly always against the zero gallery.
                candidates=[];same=[];other=[]
                own=next((g for g in gallery if g['parent']==m['parent'] and g['anchor']==i),None)
                gallery_query=descriptor if group=='repair_only' else query
                row['zero_gallery_query_role']=refrole if group=='repair_only' else role
                positive=float(np.linalg.norm(gallery_query-zero['descriptors'][i])) if own else None
                if own:
                    for g in gallery:
                        if g['parent']==m['parent']:
                            overlap=len(g['support']&own['support'])/max(1,min(len(g['support']),len(own['support'])))
                            if overlap>.1:continue
                            same.append(float(np.linalg.norm(gallery_query-g['descriptor'])))
                        elif g['group']!=m['source_group']:
                            other.append(float(np.linalg.norm(gallery_query-g['descriptor'])))
                            candidates.append(g)
                row['zero_gallery_corresponding_distance']=positive
                row['same_protein_negative_median']=float(np.median(same)) if same else None
                row['other_protein_negative_median']=float(np.median(other)) if other else None
                row['zero_gallery_rank']=1+sum(d<positive for d in same+other) if positive is not None else None
                row['same_protein_margin']=row['same_protein_negative_median']-positive if same and positive is not None else None
                row['other_protein_margin']=row['other_protein_negative_median']-positive if other and positive is not None else None
                chemical=np.mean(fref[:,2:],0)
                candidates.sort(key=lambda g:(float(np.linalg.norm(g['chemistry']-chemical)),g['parent'],g['anchor']))
                hard=[float(np.linalg.norm(gallery_query-g['descriptor'])) for g in candidates[:32]]
                row['chemistry_matched_negative_median']=float(np.median(hard)) if hard else None
                row['chemistry_matched_margin']=row['chemistry_matched_negative_median']-positive if hard and positive is not None else None
                rows.append(row)
    return rows,coverage,controls


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def report(args):
    require_compute(threads=1)
    dest=args.output/('smoke' if args.smoke else 'full')
    records=[];failed=[]
    with AugmentationStore(dest/'coordinates') as store:
        for row in store.rows:
            record=store.raw(row['id'])
            if record['metadata']['status']=='complete':records.append(record)
            else:failed.append(record['metadata'])
    rows,coverage,controls=collect(records)
    write_csv(dest/'patches.csv',rows);write_csv(dest/'mapping.csv',coverage)
    groups={}
    for group in sorted({r['group'] for r in rows}):
        rr=[r for r in rows if r['group']==group]
        metrics={k:quantile(rr,k) for k in rr[0] if isinstance(rr[0][k],(int,float)) and k not in {'anchor','size_bin'}}
        groups[group]=dict(parents=len({r['parent'] for r in rr}),patches=len(rr),metrics=metrics,
            margins={k:bootstrap_margin(rr,k) for k in ('same_protein_margin','other_protein_margin','chemistry_matched_margin')})
        for name in ('local_molecular_surface_distance_A','global_ca_rmsd_A'):
            pp=defaultdict(list)
            for r in rr:pp[r['parent']].append(r)
            x=[np.mean([r[name] for r in p]) for p in pp.values()];y=[np.mean([r['descriptor_L2'] for r in p]) for p in pp.values()]
            corr=float(spearmanr(x,y).statistic) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else None
            groups[group][name+'_spearman_with_descriptor']=corr if corr is None or np.isfinite(corr) else None
    passed=len(records)==(3 if args.smoke else 30) and not failed and bool(rows)
    summary=dict(passed=passed,planned_parents=3 if args.smoke else 30,completed_parents=len(records),failures=failed,
        groups=groups,controls=controls,mapping=dict(total=sum(r['total'] for r in coverage),mapped=sum(r['mapped'] for r in coverage)),
        aggregation='equal parent, equal view within parent, equal anchor within view; protein bootstrap 2000 draws',
        limitations=['30-parent sample enriched for strongly changed patches, not a population estimate',
            'Frozen MaSIF complementarity fingerprints are not a validated patch-identity classifier',
            'MaSIF 12 A geodesic patches retain at most 200 vertices; effective radii are reported',
            'PARSE charges and source histidine inventories are fixed; no pH titration',
            'Selected-center upstream fast-MDS construction uses deterministic per-center seeds',
            'Conclusions concern MaSIF representation, not jax-Ka performance'])
    summary['by_dataset']={}
    summary['by_sample_kind']={}
    summary['by_completion']={}
    for group in groups:
        rr=[r for r in rows if r['group']==group]
        for field,output in [('dataset','by_dataset'),('sample_kind','by_sample_kind')]:
            summary[output][group]={label:dict(parents=len({r['parent'] for r in rr if r[field]==label}),
                descriptor_L2=quantile([r for r in rr if r[field]==label],'descriptor_L2'),
                local_surface_distance_A=quantile([r for r in rr if r[field]==label],'local_molecular_surface_distance_A'))
                for label in sorted({r[field] for r in rr})}
        summary['by_completion'][group]={label:dict(parents=len({r['parent'] for r in rr if (r['completed_fraction']>0)==has}),
                descriptor_L2=quantile([r for r in rr if (r['completed_fraction']>0)==has],'descriptor_L2'))
                for label,has in [('no_completed_atoms',False),('some_completed_atoms',True)]}
    plotrows=list(rows)
    if not args.smoke and (args.output/'smoke/summary.json').exists():
        previous=json.loads((args.output/'smoke/summary.json').read_text())
        summary['smoke_preprocessing_controls']=dict(parents=previous['completed_parents'],
            fixed_mesh=previous['controls'],groups={k:previous['groups'][k] for k in ('rigid_rebuild_control','seed_repeat_control')},
            scope='three typical middle-bin parents; not a population-wide noise estimate',
            source_sha256=digest(args.output/'smoke/summary.json'))
        with (args.output/'smoke/patches.csv').open() as f:
            for row in csv.DictReader(f):
                if row['group'] in {'rigid_rebuild_control','seed_repeat_control'}:
                    for k in ('descriptor_L2','global_ca_rmsd_A','local_molecular_surface_distance_A','chemistry_matched_margin'):
                        row[k]=float(row[k]) if row[k] else None
                    plotrows.append(row)
    examples=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        candidates=[r for r in rows if r['dataset']==dataset and r['group']=='selected_vs_zero' and r['chemistry_matched_negative_median']]
        if not candidates:continue
        ratio=lambda r:r['zero_gallery_corresponding_distance']/r['chemistry_matched_negative_median']
        median=float(np.median([ratio(r) for r in candidates]))
        typical=min(candidates,key=lambda r:abs(ratio(r)-median))
        worst=max(candidates,key=ratio)
        examples.extend([dict(typical,example_kind='typical'),dict(worst,example_kind='largest_relative_change')])
    atomic_json(dest/'examples.json',examples)
    atomic_json(dest/'summary.json',summary)
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
        for group in ('repair_only','amplitude_0.5_vs_zero','amplitude_1_vs_zero','amplitude_2_vs_zero','selected_vs_zero','selected_vs_native','seed_repeat_control','rigid_rebuild_control'):
            rr=[r for r in plotrows if r['group']==group]
            if not rr:continue
            values=np.array([r['descriptor_L2'] for r in rr]);ww=weights(rr);order=np.argsort(values)
            label=group+(' (3 controls)' if not args.smoke and group in {'rigid_rebuild_control','seed_repeat_control'} else '')
            axes[0,0].plot(values[order],np.cumsum(ww[order]),label=label,
                           linestyle='--' if group in {'rigid_rebuild_control','seed_repeat_control'} else '-')
            byparent=defaultdict(list)
            for r in rr:byparent[r['parent']].append(r)
            axes[0,1].scatter([np.mean([r['global_ca_rmsd_A'] for r in p]) for p in byparent.values()],
                             [np.mean([r['descriptor_L2'] for r in p]) for p in byparent.values()],s=12,label=group)
            axes[1,0].scatter([np.mean([r['local_molecular_surface_distance_A'] for r in p]) for p in byparent.values()],
                             [np.mean([r['descriptor_L2'] for r in p]) for p in byparent.values()],s=12,label=group)
            q=[r for r in rr if r['chemistry_matched_margin'] is not None]
            if q:
                x=np.array([r['chemistry_matched_margin'] for r in q]);w=weights(q);order=np.argsort(x)
                axes[1,1].plot(x[order],np.cumsum(w[order]),label=group)
        axes[0,0].set(xlabel='Frozen fingerprint L2 distance',ylabel='Fraction, equal parent weight')
        axes[0,1].set(xlabel='Global CA RMSD (A)',ylabel='Mean fingerprint distance')
        axes[1,0].set(xlabel='Locally aligned molecular surface distance (A)',ylabel='Mean fingerprint distance')
        axes[1,1].set(xlabel='Chemistry-matched negative minus corresponding distance',ylabel='Fraction');axes[1,1].axvline(0,color='k',lw=.8)
        for ax in axes.flat:ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=7);fig.suptitle('Frozen MaSIF: global versus local changes; no training')
        fig.savefig(dest/'masif-global-vs-local.png',dpi=180);fig.savefig(dest/'masif-global-vs-local.svg');plt.close(fig)
    text=['Frozen MaSIF surface audit','==========================','',f"Completed {len(records)} of {summary['planned_parents']} parents. Passing execution: {passed}.",'']
    for group,data in groups.items():
        text.extend([group,'-'*len(group),''])
        for key in ('descriptor_L2','descriptor_cosine','local_molecular_surface_distance_A','hbond_rmse','electrostatics_rmse','hydropathy_rmse','zero_gallery_rank'):
            q=data['metrics'].get(key)
            if q:text.append(f"* {key}: median {q['median']:.4f}, p95 {q['p95']:.4f}.")
        text.append('')
    text.extend(['Limitations','-----------','']+['* '+s for s in summary['limitations']])
    if 'smoke_preprocessing_controls' in summary:
        text.extend(['','Preprocessing controls','----------------------','',
            'Rigid-rebuild and alternate-seed controls come from three typical middle-bin proteins. They are plotted separately; their variability is not a population-wide noise estimate.'])
    text.extend(['','Representative patches are recorded with original residue keys in examples.json.'])
    (dest/'report.rst').write_text('\n'.join(text)+'\n')
    atomic_json(dest/'report-provenance.json',dict(code_sha256=digest(__file__),input_manifest_sha256=digest(dest/'coordinates/manifest.json'),patch_csv_sha256=digest(dest/'patches.csv') if rows else None))
    print(json.dumps(dict(passed=passed,completed=len(records),failed=len(failed),patches=len(rows),output=str(dest))),flush=True)
    if not passed:raise RuntimeError('audit execution failed; see summary.json for per-parent diagnostics')
