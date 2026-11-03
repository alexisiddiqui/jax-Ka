"""Saved 500-parent quota audit, level-1 coordinate release and host loader smoke.

Never generates new repairs, activates training, or changes source stores.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from pkabench.runtime import atomic_json,digest,require_compute
from pkatrain.augmentation_selection import AMPLITUDES,QUOTAS,RECIPE,candidate_plan,select_quota
from pkatrain.augmentation_quality import rmsd,verify_coordinate_record
from pkatrain.augmentation_store import AugmentationStore,ReleaseWriter,verify_records


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def quantiles(values,weights):
    if not values:return None
    x=np.asarray(values,float);w=np.asarray(weights,float);ix=np.argsort(x)
    return dict(zip(('p05','median','p95'),map(float,np.interp([.05,.5,.95],(np.cumsum(w[ix])-.5*w[ix])/w.sum(),x[ix]))))


def cost(report,attempted,selection_s):
    """Retain fixed ANM/prep/context; replace candidate and selection components.

    The old production_s excluded successful zero cleanup, so explicitly add it.
    Graph costs are removed: this projection generates coordinates only.
    """
    if report['status']!='complete':return report.get('production_s',report['wall_s'])
    positive=[m for m in report['metrics'] if m['candidate']>0]
    failures=[f for f in report['failures'] if isinstance(f.get('candidate'),int) and f['candidate']>0]
    fixed=report['production_s']-sum(m['total_s']+m.get('graph_s',0) for m in positive)-sum(f.get('elapsed_s',0) for f in failures)-report['selection_s']
    if fixed<0:raise ValueError('timing accounting inconsistent')
    zero=next((m['total_s'] for m in report['metrics'] if m['candidate']==0),0.)
    return fixed+zero+sum(m['total_s'] for m in positive if m['candidate'] in attempted)+sum(f.get('elapsed_s',0) for f in failures if f['candidate'] in attempted)+selection_s


def audit(args):
    require_compute(threads=1);started=time.monotonic();args.output.mkdir(parents=True,exist_ok=True)
    pilot=args.pilot/'native-large'
    units=json.loads((pilot/'unit-summary.json').read_text());population=json.loads((pilot/'population.json').read_text())
    rows=[];selected_rows=[];pair_rows=[];selections=[];report_hashes={}
    with AugmentationStore(args.base/'coordinates') as base:
        base.verify_files()
        for number,unit in enumerate(units,1):
            old=pilot/'units'/f"{unit['index']:03d}";report=json.loads((old/'report.json').read_text())
            source=report['source'];parent=f"{source['dataset']}/{source['id']}/{source['state']}"
            report_hashes[parent]=digest(old/'report.json')
            metrics={m['candidate']:m for m in report['metrics']};eligible=report['status']=='complete' and metrics.get(0,{}).get('geometry_valid',False)
            initial=candidate_plan(metrics);refill=candidate_plan(metrics,replace=True)
            if not eligible:
                initial.update(attempted=[],pools=[[],[],[]],counts=[0,0,0],quota_met=False,selected_count=0,replacements=0)
                refill=dict(initial)
            selected_initial=[];selected_refill=[];selection_s=0.;global_values={};min_distances={}
            if eligible:
                record=base.raw(parent);meta,arrays=record['metadata'],record['arrays']
                info=json.loads((old/'preparation.json').read_text())
                heavy=np.flatnonzero(np.asarray(info['is_native'])&np.asarray(info['is_heavy']))
                if [info['atom_names'][j] for j in heavy]!=meta['atom_names']:raise ValueError('atom correspondence changed')
                caids=arrays['backbone_atoms'][:,1]
                zero=arrays['zero_coordinates'][caids].astype(float)
                xyz={c:np.load(old/f'native/c{c:02d}/min.npy')[heavy][caids].astype(float)
                     for c in set(sum(refill['pools'],[]))}
                t=time.monotonic();selected_initial=select_quota(initial,xyz,zero);selection_s=time.monotonic()-t
                t=time.monotonic();selected_refill=select_quota(refill,xyz,zero);refill_selection_s=time.monotonic()-t
                for label,selected in [('initial',selected_initial),('refill',selected_refill)]:
                    values=[rmsd(xyz[c],zero) for c in selected]
                    pairs=[rmsd(xyz[c],xyz[d]) for i,c in enumerate(selected) for d in selected[i+1:]]
                    global_values[label]=values;min_distances[label]=min(pairs) if pairs else None
                    for c,value in zip(selected,values):selected_rows.append(dict(parent=parent,dataset=source['dataset'],split=source['split'],size_bin=source['bin'],weight=unit['weight']/len(selected),parent_weight=unit['weight'],policy=label,candidate=c,amplitude_A=metrics[c]['input_amplitude_A'],post_repair_ca_rmsd_A=value))
                    for value in pairs:pair_rows.append(dict(parent=parent,dataset=source['dataset'],policy=label,weight=unit['weight']/len(pairs),pairwise_ca_rmsd_A=value))
            else:refill_selection_s=0.
            result=dict(index=unit['index'],parent=parent,dataset=source['dataset'],split=source['split'],size_bin=source['bin'],
                stratum=unit['stratum'],weight=unit['weight'],status=report['status'],zero_valid=eligible,
                initial_quota_met=initial['quota_met'],initial_selected=len(selected_initial),
                initial_counts=initial['counts'],initial_selected_ids=selected_initial,
                refill_quota_met=refill['quota_met'],refill_selected=len(selected_refill),
                refill_counts=refill['counts'],refill_selected_ids=selected_refill,
                replacement_candidates=refill['replacements'],attempted=refill['attempted'],
                initial_cost_s=cost(report,initial['attempted'],selection_s),
                refill_cost_s=cost(report,refill['attempted'],refill_selection_s),
                initial_min_pairwise_rmsd_A=min_distances.get('initial'),refill_min_pairwise_rmsd_A=min_distances.get('refill'),
                peak_rss_GiB=unit['peak_rss_GiB'])
            rows.append(result);selections.append(dict(parent=parent,index=unit['index'],selected=selected_refill,source=source,
                quota_met=refill['quota_met'],attempted=refill['attempted'],initial_selected=selected_initial,counts=refill['counts']))
            if number%50==0:print(json.dumps(dict(audited=number,total=len(units))),flush=True)
    cohorts={}
    for dataset in ('all','pkpdb','pinder','atlas'):
        sub=[r for r in rows if dataset=='all' or r['dataset']==dataset];total=sum(r['weight'] for r in sub)
        item=dict(parents=len(sub),complete=sum(r['status']=='complete' for r in sub),zero_valid=sum(r['zero_valid'] for r in sub),
            initial_four=sum(r['initial_quota_met'] for r in sub),refill_four=sum(r['refill_quota_met'] for r in sub),
            initial_at_least_three=sum(r['initial_selected']>=3 for r in sub),refill_at_least_three=sum(r['refill_selected']>=3 for r in sub),
            initial_weighted_four_fraction=sum(r['weight']*r['initial_quota_met'] for r in sub)/total,
            refill_weighted_four_fraction=sum(r['weight']*r['refill_quota_met'] for r in sub)/total,
            parents_needing_replacements=sum(r['replacement_candidates']>0 for r in sub),replacement_candidates=sum(r['replacement_candidates'] for r in sub),
            refill_count_histogram=dict(Counter(r['refill_selected'] for r in sub)),
            projected_units=total,projected_candidate_cpu_seconds=sum(r['weight']*r['refill_cost_s'] for r in sub),
            eta={str(c):dict(initial_ideal_h=sum(r['weight']*r['initial_cost_s'] for r in sub)/c/3600,
                refill_ideal_h=sum(r['weight']*r['refill_cost_s'] for r in sub)/c/3600,
                refill_planning_h_30pct_margin=sum(r['weight']*r['refill_cost_s'] for r in sub)/c/3600*1.3) for c in (240,288,380)})
        cohorts[dataset]=item
    distribution={}
    for policy in ('initial','refill'):
        distribution[policy]={}
        for amplitude in AMPLITUDES:
            sub=[r for r in selected_rows if r['policy']==policy and r['amplitude_A']==amplitude]
            # Each protein receives equal stratum weight within an amplitude.
            counts=Counter(r['parent'] for r in sub);uw={r['parent']:r['weight'] for r in rows}
            distribution[policy][str(amplitude)]=quantiles([r['post_repair_ca_rmsd_A'] for r in sub],[uw[r['parent']]/counts[r['parent']] for r in sub])
    summary=dict(recipe=RECIPE,parents=len(rows),population_units=population['total'],datasets=cohorts,
        post_repair_ca_rmsd_A=distribution,pairwise_ca_rmsd_A={p:quantiles([r['pairwise_ca_rmsd_A'] for r in pair_rows if r['policy']==p],[r['weight'] for r in pair_rows if r['policy']==p]) for p in ('initial','refill')},
        wall_s=time.monotonic()-started,limitations=[
            'Candidate and runtime policies replay saved 24-candidate results; not a fresh eight-candidate end-to-end generation run',
            'Full population and ETA use original dataset/split/length-bin stratum weights; failed source/preparation and zero controls remain in denominators',
            'Includes measured ANM construction, source completion/preparation, context, successful zero cleanup, attempted repair/geometry costs and selection; excludes graph work',
            'Replacement only within the same amplitude, at most eight directions per bin; unsatisfied quotas are explicit, never backfilled across bins',
            'CPU scaling is ideal with a separate 30 percent planning allowance; queueing, transfers, shared-filesystem contention and memory-qualified node concurrency remain outside this estimate',
            'Reference for selecting and comparing repaired structures is the repaired zero; raw native coordinates remain for provenance and independent geometry checks',
            'No local surface-preservation gate, APBS, fresh minimisation or model training'])
    atomic_json(args.output/'audit-summary.json',summary);atomic_json(args.output/'selection.json',dict(recipe=RECIPE,parents=selections))
    atomic_json(args.output/'audit-provenance.json',dict(script_sha256=digest(__file__),selection_module_sha256=digest(Path(__file__).parents[1]/'src/pkatrain/augmentation_selection.py') if (Path(__file__).parents[1]/'src/pkatrain/augmentation_selection.py').exists() else digest(__import__('pkatrain.augmentation_selection',fromlist=['x']).__file__),
        sample_sha256=digest(pilot/'sample.json'),unit_summary_sha256=digest(pilot/'unit-summary.json'),population_sha256=digest(pilot/'population.json'),
        source_manifest_sha256=digest(args.base/'coordinates/manifest.json'),report_sha256=report_hashes))
    write_csv(args.output/'parents.csv',rows);write_csv(args.output/'selected.csv',selected_rows);write_csv(args.output/'pairwise.csv',pair_rows)
    plots(args.output,selected_rows,pair_rows)
    print(json.dumps(cohorts),flush=True)


def plots(output,selected,pairs):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for amp in AMPLITUDES:
        sub=[r for r in selected if r['policy']=='refill' and r['amplitude_A']==amp]
        # Parent weights within the amplitude; the two high-amplitude views do not double its curve weight.
        counts=Counter(r['parent'] for r in sub)
        x=np.array([r['post_repair_ca_rmsd_A'] for r in sub]);w=np.array([r['parent_weight']/counts[r['parent']] for r in sub]);ix=np.argsort(x)
        axes[0].plot(x[ix],np.cumsum(w[ix])/w.sum(),label=f'{amp:g} Å target')
    sub=[r for r in pairs if r['policy']=='refill'];x=np.array([r['pairwise_ca_rmsd_A'] for r in sub]);w=np.array([r['weight'] for r in sub]);ix=np.argsort(x)
    axes[1].plot(x[ix],np.cumsum(w[ix])/w.sum())
    axes[0].set(xlabel='Post-repair aligned CA RMSD vs repaired zero (Å)',ylabel='Population-weighted cumulative fraction');axes[0].legend()
    axes[1].set(xlabel='Pairwise aligned CA RMSD between retained views (Å)',ylabel='Population-weighted cumulative fraction')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Saved 500-parent pilot: five modes, 1/1/2 amplitude quotas with bounded replacement')
    fig.savefig(output/'rmsd-distributions.png',dpi=170);fig.savefig(output/'rmsd-distributions.svg');plt.close(fig)


def pack(args):
    require_compute(threads=1);started=time.monotonic();selection=json.loads((args.output/'selection.json').read_text())
    from pkatrain.augmentation_evidence import CoordinateInputs
    audited=json.loads((args.output/'audit-provenance.json').read_text())
    if audited['source_manifest_sha256']!=digest(args.base/'coordinates/manifest.json'):raise ValueError('audited source changed')
    if selection['recipe']!=RECIPE:raise ValueError('recipe differs from audited selection')
    provenance=dict(kind='five-mode-quota-112-coordinate-readiness-v1',recipe=RECIPE,training_enabled=False,
        source_manifest_sha256=digest(args.base/'coordinates/manifest.json'),selection_sha256=digest(args.output/'selection.json'),
        source_evidence_manifest_sha256=digest(args.base/'evidence/manifest.json'),original_input_evidence='embedded in each coordinate record; no per-parent evidence files',
        reference_view='zero_coordinates',source_native_preserved_for_geometry_and_provenance=True,
        source='saved 24-candidate pilot; new quota selection, no regenerated structures')
    dest=args.output/'coordinates';timings=[]
    if (dest/'manifest.json').exists():
        with AugmentationStore(dest) as published:
            if published.manifest['provenance']!=provenance:raise ValueError('published release provenance differs')
            published.verify_files();manifest=published.manifest
        if (args.output/'packaging.json').exists():
            previous=json.loads((args.output/'packaging.json').read_text())
            if previous['manifest_sha256']!=digest(dest/'manifest.json'):raise ValueError('packaging verification belongs to a different release')
            print(json.dumps(dict(passed=True,published_release_reused=True,parents=manifest['records'])),flush=True)
            return
        return packaging_summary(args,manifest,timings,started)
    with CoordinateInputs(args.base/'coordinates',args.base/'evidence') as base,ReleaseWriter(dest,provenance,zstd_level=1) as writer:
        for number,item in enumerate(selection['parents'],1):
            if item['parent'] in writer.ids:continue
            t=time.monotonic();record=base.raw(item['parent']);meta=dict(record['metadata']);arrays=dict(record['arrays'])
            selected=item['selected'];old=args.pilot/f"native-large/units/{item['index']:03d}"
            report=json.loads((old/'report.json').read_text());metrics={m['candidate']:m for m in report['metrics']}
            if digest(old/'report.json')!=audited['report_sha256'][item['parent']]:raise ValueError('repair report changed since audit')
            meta.update(selected=selected,augmentation_recipe=RECIPE,quota_met=item['quota_met'],
                reference_view='zero_coordinates',attempted_candidates=item['attempted'],
                source_report_selected=meta['selected'],status='usable' if item['quota_met'] else 'partial_quota' if selected else meta['status'])
            if arrays:
                info=json.loads((old/'preparation.json').read_text());heavy=np.flatnonzero(np.asarray(info['is_native'])&np.asarray(info['is_heavy']))
                arrays['coordinates']=np.stack([np.load(old/f'native/c{c:02d}/min.npy')[heavy] for c in selected]).astype(np.float32) if selected else np.empty((0,len(heavy),3),np.float32)
                ca=arrays['backbone_atoms'][:,1]
                meta['selected_quality']={str(c):metrics[c] for c in selected}
                meta['selected_pairwise_ca_rmsd_A']=[rmsd(arrays['coordinates'][i,ca],arrays['coordinates'][j,ca]) for i in range(len(selected)) for j in range(i+1,len(selected))]
                verify_coordinate_record(dict(metadata=meta,arrays=arrays))
            row=writer.add(item['parent'],meta,arrays,summary=dict(dataset=item['source']['dataset'],split=item['source']['split'],state=item['source']['state'],n=item['source']['n'],selected=len(selected),status=meta['status'],quota_met=item['quota_met']))
            timings.append(dict(parent=item['parent'],wall_s=time.monotonic()-t,compressed_bytes=row['bytes'],raw_bytes=row['raw_bytes']))
            if number%50==0:print(json.dumps(dict(packed=number,total=len(selection['parents']))),flush=True)
        manifest=writer.finish(expected_ids=[r['parent'] for r in selection['parents']])
    return packaging_summary(args,manifest,timings,started)


def packaging_summary(args,manifest,timings,started):
    dest=args.output/'coordinates'
    frames=0
    def check(record,row):
        nonlocal frames
        frames+=verify_coordinate_record(record)
        if record['metadata']['reference_view']!='zero_coordinates':raise ValueError('reference is not consistently repaired')
        if record['arrays']:
            for key in ('original_input_present','original_input_known'):
                if record['arrays'][key].shape!=record['arrays']['atom_residue'].shape or record['arrays'][key].dtype!=np.bool_:
                    raise ValueError('original-input atom evidence differs')
        if record['metadata']['quota_met']:
            counts=Counter((c-1)//8 for c in record['metadata']['selected'])
            if tuple(counts[b] for b in range(3))!=QUOTAS:raise ValueError('persisted quota differs')
    with AugmentationStore(dest) as store:
        verification=verify_records(dest,store.rows,check_record=check,expected_files=manifest['files'])
        all_rows=store.rows
    units={r['parent']:r for r in csv.DictReader((args.output/'parents.csv').open())}
    projected=sum(float(units[r['id']]['weight'])*r['bytes'] for r in all_rows)
    ownfiles=[p for p in dest.rglob('*') if p.is_file()]
    result=dict(passed=True,parents=manifest['records'],frames_checked=frames,zstd_level=manifest['zstd_level'],shard_bytes=manifest['shard_bytes'],
        pilot_compressed_bytes=manifest['compressed_bytes'],pilot_raw_bytes=manifest['raw_bytes'],
        projected_full_coordinate_compressed_bytes=projected,projected_data_shards=math.ceil(projected/manifest['shard_bytes']),
        consolidated_release_file_count=math.ceil(projected/manifest['shard_bytes'])+5,
        illustrative_380_worker_data_shard_upper_bound=math.ceil(projected/manifest['shard_bytes'])+379,
        illustrative_380_worker_release_files=380*8+math.ceil(projected/manifest['shard_bytes'])+379+5,
        current_release_files=len(ownfiles),current_release_bytes=sum(p.stat().st_size for p in ownfiles),
        writer_and_independent_verification_wall_s=time.monotonic()-started,writer_unit_timings=timings,
        measured_writer_records=len(timings),resumed_or_published_records=manifest['records']-len(timings),
        manifest_sha256=digest(dest/'manifest.json'),reference='zero_coordinates; same capped native+min repair as candidates',
        original_atom_evidence_embedded=True,
        limits='coordinate payload and original-input atom masks; final indexes, local staging, transfers and graph stores need separate allowance')
    atomic_json(args.output/'packaging.json',result);print(json.dumps({k:v for k,v in result.items() if k!='writer_unit_timings'}),flush=True)


def graphs(args):
    require_compute(threads=1)
    from pkatrain.augmentation_graphs import INPUT_FIELDS,free_graph,pair_inputs
    from pkatrain.augmentation_pilot import verify_graphs
    from pkanet.graph import sidechain_features
    selected=[];alignment=[];bound_checks=0
    dest=args.output/'graphs'
    with AugmentationStore(args.output/'coordinates') as coordinates,AugmentationStore(args.base/'graphs') as old:
        for dataset in ('pkpdb','pinder','atlas'):
            for sizebin in range(5):
                preferred='val' if sizebin in (1,3) else 'train'
                eligible=[r for r in coordinates.rows if r['summary']['quota_met'] and r['summary']['dataset']==dataset
                          and coordinates.raw(r['id'])['metadata']['source']['bin']==sizebin]
                eligible.sort(key=lambda r:(r['summary']['split']!=preferred,r['id']))
                if eligible:selected.append(eligible[0]['id'])
        provenance=dict(kind='quota-112-graph-loader-smoke-v1',coordinate_manifest_sha256=digest(args.output/'coordinates/manifest.json'),
            canonical_source_graph_manifest_sha256=digest(args.base/'graphs/manifest.json'),training_enabled=False,
            reference='native bound branch retained; free reference is repaired zero; all four free augmentations use same repair',parents=selected)
        with ReleaseWriter(dest,provenance,zstd_level=1) as writer:
            for parent in selected:
                record=coordinates.raw(parent);meta,arrays=record['metadata'],record['arrays']
                source=old.raw(parent+'/native');original=source['arrays'];mapping=np.asarray(source['metadata']['canonical_node_mapping'],int)
                bound={k:original[k][0].copy() for k in INPUT_FIELDS};table=bound['nodes'][:,-14:]
                bb=table[:,:12].reshape(-1,4,3).copy();chain=table[:,-2].astype(int);sequence=table[:,-1].astype(int)
                bound['nodes']=bound['nodes'][:,:-14].copy()
                if np.max(np.abs(bb[mapping]-arrays['source_backbone']))>.002:raise ValueError('canonical mapping changed')
                partners=np.zeros(len(bb),int) if meta['source']['dataset']=='pkpdb' else chain
                zero_bb=None
                for c,xyz in [(0,arrays['zero_coordinates'])]+list(zip(meta['selected'],arrays['coordinates'])):
                    moved=bb.copy();moved[mapping]=xyz[arrays['backbone_atoms']]
                    if c==0:zero_bb=moved
                    free=free_graph(bound,moved,chain,partners)
                    payload=pair_inputs(bound,free,bb,moved,chain,sequence)
                    # Recompute changed-partner sidechains; the untouched partner retains its native features.
                    atoms=[{} for _ in meta['residue_names']]
                    for r,name,x in zip(arrays['atom_residue'],meta['atom_names'],xyz):atoms[int(r)][name]=x
                    free_sc=original['sidechains'][1].copy();free_sc[mapping]=sidechain_features(moved[mapping],atoms).astype(np.float16)
                    payload['sidechains']=np.stack((original['sidechains'][0],free_sc))
                    for key in INPUT_FIELDS+('sidechains',):
                        expected=original[key][0];actual=payload[key][(0,*(slice(0,n) for n in expected.shape))]
                        if not np.array_equal(actual,expected):raise ValueError('native bound inputs changed: '+key)
                    if not np.array_equal(payload['nodes'][1,mapping,-14:-2],xyz[arrays['backbone_atoms']].reshape(len(mapping),12)):
                        raise ValueError('free geometry was not rebuilt from persisted coordinates')
                    bound_checks+=1
                    dims=[len(bb),payload['neighbors'].shape[-1],payload['query_residue'].shape[-1],payload['site_residue'].shape[-1],payload['site_neighbors'].shape[-1]]
                    gm=dict(source_parent=parent,dataset=meta['source']['dataset'],split=meta['source']['split'],kind='native' if c==0 else 'augmented',
                        candidate=c,reference_view='repaired_zero',canonical_node_mapping=mapping.tolist(),dims=dims,geometry_labels_reused=False)
                    writer.add(parent+('/native' if c==0 else f'/c{c:02d}'),gm,payload,summary=gm)
                alignment.append(dict(parent=parent,canonical_mapping=True,free_zero_reference=True,selected=meta['selected']))
                print(json.dumps(dict(graph_parent=parent,records=len(writer.rows))),flush=True)
            writer.finish(expected_ids=[parent+suffix for parent in selected for suffix in ['/native']+[f'/c{c:02d}' for c in coordinates.raw(parent)['metadata']['selected']]])
    verify_graphs(dest)
    atomic_json(args.output/'graph-smoke.json',dict(passed=True,parents=len(selected),records=bound_checks,
        bound_arrays_identical_to_canonical_source=True,free_reference='repaired zero',free_geometry_and_sidechains_rebuilt=True,
        geometry_labels_reused=False,alignment=alignment,manifest_sha256=digest(dest/'manifest.json'),
        scope='15 dataset/length-spread parents including train and validation; no training activation or full graph release'))


def reads(args):
    require_compute(threads=2)
    from pkatrain.augmentation_preflight import read_benchmark
    report=read_benchmark(args.output/'graphs',args.output/'loader-smoke.json',workers=(1,2),prefetch=(2,),steps=12,repeats=3,batch=2)
    print(json.dumps(dict(measurements=len(report['results']),peak_rss_GiB=report['peak_rss_GiB'])),flush=True)


def report(args):
    require_compute(threads=1)
    audit=json.loads((args.output/'audit-summary.json').read_text())
    packaging=json.loads((args.output/'packaging.json').read_text())
    graphs=json.loads((args.output/'graph-smoke.json').read_text())
    reads=json.loads((args.output/'loader-smoke.json').read_text())
    inventory=json.loads((args.pilot/'atlas-source-inventory.json').read_text())
    inactive={r['id'] for r in json.loads((args.pilot/'atlas-inactive-records.json').read_text())}
    active=[r for r in inventory if r['id'] not in inactive]
    archives=sorted({r['archive']['archive'] for r in active})
    missing=[p for p in archives if not Path(p).exists()]
    source_staging=dict(active_atlas_complexes=len(active),active_atlas_partner_units=2*len(active),recorded_archives=len(archives),
        recorded_archive_paths_present_on_this_host=len(archives)-len(missing),recorded_archive_paths_missing=len(missing),
        compressed_active_member_bytes=sum(r['archive']['bytes'] for r in active),missing_recorded_archive_paths=missing,
        inventory_sha256=digest(args.pilot/'atlas-source-inventory.json'),
        scope='recorded source inventory paths checked on Coulson; alternate relocated copies are not assumed or searched; member-byte sum excludes tar framing and indexes')
    if not packaging['passed'] or not graphs['passed']:raise ValueError('verification failed')
    if packaging['zstd_level']!=1 or packaging['shard_bytes']!=2*1024**3:raise ValueError('storage layout differs')
    if reads['manifest_sha256']!=graphs['manifest_sha256']:raise ValueError('loader measured a different graph release')
    controls={};throughput=[]
    for row in reads['results']:
        key=row['bucket'],row['kind'];value=row['checksum'],row['materialised_bytes']
        if key in controls and value!=controls[key]:raise ValueError('loader worker/repeat outputs differ')
        controls[key]=value
    for bucket in sorted({r['bucket'] for r in reads['results']},key=int):
        for workers in sorted({r['workers'] for r in reads['results']}):
            pair={}
            for kind in ('native','augmented'):
                sub=[r for r in reads['results'] if r['bucket']==bucket and r['workers']==workers and r['kind']==kind]
                pair[kind]=float(np.median([r['structures_per_second'] for r in sub]))
            throughput.append(dict(bucket=int(bucket),workers=workers,reference_structures_per_s=pair['native'],
                augmented_structures_per_s=pair['augmented'],augmented_vs_reference_ratio=pair['augmented']/pair['native']))
    final=dict(pilot_checks_passed=True,recipe=RECIPE,population_units=audit['population_units'],yield_and_eta=audit['datasets'],
        post_repair_ca_rmsd_A=audit['post_repair_ca_rmsd_A']['refill'],pairwise_ca_rmsd_A=audit['pairwise_ca_rmsd_A']['refill'],
        packaging={k:v for k,v in packaging.items() if k!='writer_unit_timings'},graph_smoke=graphs,
        loader=dict(materialisation_and_checksum_checks_passed=True,peak_rss_GiB=reads['peak_rss_GiB'],measurements=len(reads['results']),throughput=throughput,
            scope='host-only warm-working-set smoke on precomputed graph records; not full-dataset random I/O, GPU throughput or on-the-fly graph construction'),
        production_started=False,training_changed=False,source_staging=source_staging,
        full_dataset_source_staging_ready=not missing,limitations=audit['limitations'])
    atomic_json(args.output/'readiness.json',final)
    lines=['Five-mode 1/1/2 recipe readiness','================================','',
        'All authorised pilot checks passed. No full-dataset generation or training was started.','',
        'Yield (all 500 sampled parents; source/preparation failures remain in denominators)','-------------------------------------------------------------------------------','',
        'Dataset | Initial eight-candidate four-view yield | Bounded-refill four-view yield | At least three after refill','']
    for dataset,s in audit['datasets'].items():lines.append(f"{dataset} | {s['initial_four']}/{s['parents']} | {s['refill_four']}/{s['parents']} | {s['refill_at_least_three']}/{s['parents']}")
    lines+=['','Full population runtime projection','----------------------------------','',
        'Includes ANM construction, source completion/preparation, successful zero cleanup and attempted repair/geometry work. It excludes full graph generation, queueing, transfers, contention and this independent retrospective audit.','']
    for cpus,e in audit['datasets']['all']['eta'].items():lines.append(f"* {cpus} CPU workers: ideal {e['refill_ideal_h']:.2f} h; planning with 30 percent allowance {e['refill_planning_h_30pct_margin']:.2f} h.")
    lines+=['','Coordinate release','------------------','',
        f"zstd level 1; 2 GiB shards; {packaging['parents']} indexed parent records in {packaging['current_release_files']} files.",
        f"Projected full coordinate payload including original-input atom masks: {packaging['projected_full_coordinate_compressed_bytes']/1e9:.2f} GB ({packaging['projected_full_coordinate_compressed_bytes']/1024**3:.2f} GiB).",
        f"Consolidated release allowance: {packaging['consolidated_release_file_count']} files; illustrative conservative 380-worker allowance: {packaging['illustrative_380_worker_release_files']} files, below the 5000-file per-release cap.",
        'Local staging, transfer partials and a future full graph release require separate space and file budgets. This is not a statement of effective user/project quota.','',
        'Loader smoke','------------','',
        f"{graphs['parents']} dataset/size-spread parents; {graphs['records']} graph records. Native bound inputs preserved; free reference uses repaired zero; all four free augmentation inputs and sidechain features rebuilt.",
        'Same materialised outputs/checksums across loader worker settings and repeats. Native in the host loader report means the repaired-zero free reference, with the original bound branch.',
        f"Peak host loader RSS: {reads['peak_rss_GiB']:.2f} GiB. Read timings use a small, uncontrolled warm working set and do not establish training GPU wait or cold full-dataset I/O performance.",'',
        'Bucket | Workers | Reference structures/s | Augmented structures/s | Ratio','']
    for r in throughput:lines.append(f"{r['bucket']} | {r['workers']} | {r['reference_structures_per_s']:.1f} | {r['augmented_structures_per_s']:.1f} | {r['augmented_vs_reference_ratio']:.3f}")
    lines+=['','Source staging for a full Coulson run','------------------------------------','',
        f"{len(active)} active ATLAS complexes ({2*len(active)} partner units). Of {len(archives)} recorded full source archive paths, {len(missing)} are absent on this host.",
        f"Active compressed source members total {source_staging['compressed_active_member_bytes']/1e9:.2f} GB; this excludes archive framing/indexes and is not a measured transfer ETA.",
        'The saved pilot bundle is local. A full run needs the recorded ATLAS inputs staged or an explicitly verified local inventory remapping before launch. The CPU ETA above excludes source transfer.']
    lines+=['','Limitations','-----------','']+['* '+s for s in audit['limitations']]
    (args.output/'report.rst').write_text('\n'.join(lines)+'\n')
    atomic_json(args.output/'provenance.json',dict(script_sha256=digest(__file__),source_files={name:digest(args.output/name) for name in
        ('audit-summary.json','audit-provenance.json','selection.json','packaging.json','graph-smoke.json','loader-smoke.json')}))
    print(json.dumps(dict(pilot_checks_passed=True,output=str(args.output))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['audit','pack','graphs','reads','report'])
    for name in ('pilot','output','base'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();globals()[args.command](args)


if __name__=='__main__':main()
