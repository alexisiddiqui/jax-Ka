"""Population-weighted ETA and held-out surface-filter decisions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from pkabench.runtime import atomic_json,digest,require_compute
from augmentation_surface_filter import report, farthest, bad_fraction, KEYS
from augmentation_surface_audit import fit, rms
from augmentation_masif_report import weights


def staged_selection(result, limits, allowance):
    # Reject cheaply whenever non-electrostatic features already exceed the full
    # allowance. Then query electrostatics in greedy diversity order until three
    # passing candidates are found. This exactly matches exhaustive selection.
    cheap_keys=tuple(k for k in KEYS if k!='electrostatics_rmse')
    byrole={f"c{c['candidate']:02d}":[p for p in result['patches'] if p['role']==f"c{c['candidate']:02d}"] for c in result['candidates']}
    eligible=[];full=[]
    for c in result['candidates']:
        if c['status']!='measured':continue
        rows=byrole[f"c{c['candidate']:02d}"]
        if bad_fraction(rows,result['expected_anchors'],limits,cheap_keys)<=allowance:eligible.append(c['candidate'])
        if bad_fraction(rows,result['expected_anchors'],limits)<=allowance:full.append(c['candidate'])
    queries=[];selected=[]
    with np.load(Path(result['path'])/'candidate-ca.npz') as coords:
        minimum={}
        for c in eligible:
            x,y=coords[str(c)],coords['reference'];rot,trans=fit(x,y);minimum[c]=rms(x@rot+trans,y)
        while minimum and len(selected)<3:
            best=max(minimum,key=lambda c:(minimum[c],-c));del minimum[best];queries.append(best)
            if best not in full:continue
            selected.append(best)
            for c in minimum:
                x,y=coords[str(c)],coords[str(best)];rot,trans=fit(x,y)
                minimum[c]=min(minimum[c],rms(x@rot+trans,y))
        if selected!=farthest(full,coords):raise ValueError('staged selection changed full-filter decisions')
    return dict(parent=result['parent'],dataset=result['dataset'],size_bin=result['size_bin'],
                cheap_eligible=len(eligible),electrostatic_candidate_queries=len(queries),selected=selected,
                queried_candidates=queries)


def plot(results, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected={r['parent']:set(r['selected']) for r in summary['parents']}
    byparent={r['parent']:r for r in results if r['evaluation']}
    decisions=[]
    import csv
    with (output/'candidate-filter.csv').open() as f: decisions=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for amplitude in (.5,1.,2.):
        values=[float(r['bad_patch_fraction']) for r in decisions if r['bad_patch_fraction'] and float(r['amplitude_A'])==amplitude]
        axes[0,0].hist(values,bins=np.linspace(0,1,21),histtype='step',label=f'{amplitude:g} Å ANM')
    axes[0,0].axvline(summary['max_bad_patch_fraction'],color='k',linestyle='--',label='Calibrated allowance')
    for dataset in ('atlas','pinder','pkpdb'):
        pp=[r for r in summary['parents'] if r['dataset']==dataset]
        axes[0,1].scatter([byparent[r['parent']]['n_residues'] for r in pp],[r['eligible'] for r in pp],label=dataset)
    axes[0,1].axhline(3,color='k',linestyle='--')
    baseline={}
    for parent,r in byparent.items():
        with np.load(Path(r['path'])/'candidate-ca.npz') as coords:
            baseline[parent]=set(farthest([c['candidate'] for c in r['candidates'] if c['status']=='measured'],coords))
    # Restrict before/after comparisons to parents that actually yield three.
    comparable={p for p in selected if len(selected[p])==3}
    for label, choices in [('Global diversity only',baseline),('Surface filter + global diversity',selected)]:
        values=[];surfaces=[]
        for parent in comparable:
            r=byparent[parent]
            values.extend(dict(parent=parent,role='selected',value=c['global_ca_rmsd_A']) for c in r['candidates'] if c['candidate'] in choices[parent])
            roles={f'c{c:02d}' for c in choices[parent]}
            surfaces.extend(dict(v,value=v['surface_distance_A']) for v in r['patches'] if v['role'] in roles)
        for ax,rr in [(axes[1,0],values),(axes[1,1],surfaces)]:
            if rr:
                x=np.array([v['value'] for v in rr]);w=weights(rr);ix=np.argsort(x)
                ax.plot(x[ix],np.cumsum(w[ix]),label=label)
    axes[0,0].set(xlabel='Flagged or unmapped fraction of 9 Å patches',ylabel='Candidate count')
    axes[0,1].set(xlabel='Protein residues',ylabel='Candidates passing surface filter')
    axes[1,0].set(xlabel='Selected aligned global CA RMSD (Å)',ylabel='Fraction, equal parent weight')
    axes[1,1].set(xlabel='Locally aligned surface distance (Å)',ylabel='Fraction, equal parent weight')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f"Surface-filter pilot: 12 held-out parents; matched before/after comparison on {len(comparable)} yielding three views")
    fig.savefig(output/'surface-filter-pilot.png',dpi=180);fig.savefig(output/'surface-filter-pilot.svg');plt.close(fig)


def main(args):
    require_compute(threads=1)
    results=[json.loads(p.read_text()) for p in sorted((args.input/'parents').glob('*/result.json'))]
    if len(results)!=30:raise ValueError('all 30 parents must complete')
    args.output.mkdir(parents=True,exist_ok=True)
    report(results,args.output,max(r['wall_s'] for r in results),24)
    summary=json.loads((args.output/'summary.json').read_text())
    pilot=args.input.parents[2]
    population=json.loads((pilot/'native-large/population.json').read_text())
    strata={}
    for key,count in population['strata'].items():
        dataset,split,b=key.split('|');cell=dataset+'|'+b
        strata[cell]=strata.get(cell,0)+count
    frame_cost={};apbs_cost={}
    for cell,count in strata.items():
        dataset,b=cell.split('|');parents=[r for r in results if r['dataset']==dataset and r['size_bin']==int(b)]
        assert len(parents)==2
        # Equal protein weight, rather than treating 24 repeated frames as 24 proteins.
        frame_cost[cell]=float(np.mean([np.mean([r['control_wall_s']]+[v['wall_s'] for v in r['new_frame_timings']]) for r in parents]))
        # Raw-only bridge receipts record surface/electrostatic sub-stage costs.
        costs=[]
        for r in parents:
            controls=json.loads((Path(r['path'])/'rigid-control/receipt.json').read_text())
            costs.append(np.mean([controls['timings']['electrostatics_s']]+[v['timings']['electrostatics_s'] for v in r['new_frame_timings']]))
        apbs_cost[cell]=float(np.mean(costs))
    total_frames=sum(strata.values())*25
    surface_worker_s=sum(strata[c]*25*frame_cost[c] for c in strata)
    apbs_worker_s=sum(strata[c]*25*apbs_cost[c] for c in strata)
    baseline=json.loads((pilot/'native-large/unit-summary.json').read_text())
    base_cpu_s=sum(r['weight']*r['production_s'] for r in baseline)
    one=json.loads((args.cpu/'summary.json').read_text())
    assert one['passed']
    ratios=[r['speed_ratio'] for r in one['results']]
    slowdown=float(np.mean(ratios))
    eta={}
    for cpus in (96,240,288,380):
        surface_days=surface_worker_s*slowdown/cpus/86400
        coordinate_days=base_cpu_s/cpus/86400
        eta[str(cpus)]=dict(one_core_workers=cpus,surface_screening_ideal_days=surface_days,
            coordinate_generation_including_ANM_ideal_days=coordinate_days,
            sequential_total_ideal_days=surface_days+coordinate_days,
            sequential_total_planning_days_30pct_margin=(surface_days+coordinate_days)*1.3,
            original_four_core_surface_ideal_days=surface_worker_s/(cpus//4)/86400)
    staged=[staged_selection(r,summary['thresholds'],summary['max_bad_patch_fraction']) for r in results if r['evaluation']]
    # Middle-bin query count is inferred from the two adjacent measured bins.
    # Costs retain all 25 cheap geometry surfaces as a conservative denominator.
    query_counts={}
    for cell in strata:
        dataset,b=cell.split('|');b=int(b)
        observed=[r['electrostatic_candidate_queries'] for r in staged if r['dataset']==dataset and r['size_bin']==b]
        if not observed:observed=[r['electrostatic_candidate_queries'] for r in staged if r['dataset']==dataset and r['size_bin'] in (1,3)]
        query_counts[cell]=float(np.mean(observed))
    staged_worker_s=sum(strata[c]*(25*(frame_cost[c]-apbs_cost[c])+(1+query_counts[c])*apbs_cost[c]) for c in strata)
    staged_eta={str(cpus):dict(surface_filter_ideal_days=staged_worker_s*slowdown/cpus/86400,
        including_coordinate_generation_planning_days_30pct_margin=(staged_worker_s*slowdown+base_cpu_s)/cpus/86400*1.3) for cpus in (96,240,288,380)}
    summary['staged_filter_simulation']=dict(selections_match_exhaustive=True,parents=staged,
        eta=staged_eta,query_count_by_length_bin=query_counts,
        scope='Selection simulated from all measured candidates; timing sums measured stages, not a new end-to-end staged benchmark')
    summary['eta_surface_filter_unweighted_initial']=summary.pop('eta_surface_filter')
    summary['population_weighted_eta']=eta
    summary['frame_cost_by_dataset_length_bin_s']=frame_cost
    summary['population_by_dataset_length_bin']=strata
    summary['full_surface_frames']=total_frames
    summary['electrostatics_fraction_of_surface_cost']=apbs_worker_s/surface_worker_s
    summary['one_core_profile']=dict(controls=len(ratios),mean_slowdown=slowdown,
        median_slowdown=float(np.median(ratios)),range=list(map(float,[min(ratios),max(ratios)])),
        max_peak_rss_MiB=max(r['one_core_peak_rss_MiB'] for r in one['results']),passed=True,
        source=str(args.cpu/'summary.json'))
    if args.memory:
        memory=json.loads((args.memory/'summary.json').read_text())
        assert memory['passed']
        peak=max(r['peak_tree_rss_MiB'] for r in memory['results'])
        cores_per_worker=max(1,int(np.ceil(peak*1.25/(2048*.9))))
        memory_eta={}
        for cpus in (96,240,288,380):
            workers=cpus//cores_per_worker
            # When four cores must be reserved for memory, use the measured
            # four-core computation instead of leaving three cores idle.
            active_cores=4 if cores_per_worker>=4 else 1
            layout_slowdown=1. if active_cores==4 else slowdown
            memory_eta[str(cpus)]=dict(workers=workers,reserved_cores_per_worker=cores_per_worker,
                active_cores_per_worker=active_cores,
                exhaustive_ideal_days=(surface_worker_s*layout_slowdown/workers+base_cpu_s/cpus)/86400,
                exhaustive_planning_days_30pct_margin=(surface_worker_s*layout_slowdown/workers+base_cpu_s/cpus)/86400*1.3,
                staged_simulation_ideal_days=(staged_worker_s*layout_slowdown/workers+base_cpu_s/cpus)/86400,
                staged_simulation_planning_days_30pct_margin=(staged_worker_s*layout_slowdown/workers+base_cpu_s/cpus)/86400*1.3)
        summary['memory_safe_eta']=memory_eta
        summary['memory_profile']=dict(source=str(args.memory/'summary.json'),max_tree_rss_MiB=peak,
            worker_margin=.25,node_memory_reserve=.1,reserved_cores_per_worker=cores_per_worker,
            allocation_memory_per_core_MiB=2048,scope='Three heaviest measured candidate electrostatics frames; no claim to bound all future proteins')
        summary['limitations'].append('One-worker-per-CPU figures are compute-only ceilings; memory-safe ETA reserves enough 2 GB core allocations for measured process-tree peaks')
    summary['limitations'] += ['Final ETA weights dataset and recorded-length bins, pooling train/validation; only two enriched parent examples per cell',
        'One-core ETA extrapolates nine matched controls to production candidates; concurrent hundreds-worker I/O scaling remains unmeasured',
        'Existing 500-parent coordinate projection includes source reading/completion, ANM, hydrogen/system preparation, repair and geometry selection',
        'Staged cost is a simulation using cached measurements; chemistry can be deferred but CPU/memory staging overhead is not benchmarked',
        '30% margin is a planning allowance, not a statistical confidence interval; no full release has started']
    plot(results,summary,args.output)
    atomic_json(args.output/'summary.json',summary)
    with (args.output/'report.rst').open('a') as f:
        f.write('\nPopulation-weighted final ETA\n-----------------------------\n\n')
        for c,e in eta.items():f.write(f"* {c} one-core workers: surface {e['surface_screening_ideal_days']:.2f} days; coordinates incl. ANM {e['coordinate_generation_including_ANM_ideal_days']:.2f} days; combined planning {e['sequential_total_planning_days_30pct_margin']:.2f} days.\n")
        for c,e in staged_eta.items():f.write(f"* Simulated two-stage filter, {c} CPUs: combined planning {e['including_coordinate_generation_planning_days_30pct_margin']:.2f} days.\n")
        if args.memory:
            for c,e in memory_eta.items():f.write(f"* Memory-safe {c} CPUs / {e['workers']} workers: exhaustive planning {e['exhaustive_planning_days_30pct_margin']:.2f} days; staged simulated planning {e['staged_simulation_planning_days_30pct_margin']:.2f} days.\n")
        f.write(f"\nAPBS accounts for {100*summary['electrostatics_fraction_of_surface_cost']:.1f}% of measured surface processing.\n")
    atomic_json(args.output/'provenance.json',dict(code_sha256=digest(__file__),decision_code_sha256=digest(Path(__file__).with_name('augmentation_surface_filter.py')),
        design_sha256=digest(args.input/'design.json'),population_sha256=digest(pilot/'native-large/population.json'),
        baseline_sha256=digest(pilot/'native-large/unit-summary.json'),one_core_sha256=digest(args.cpu/'summary.json'),memory_sha256=digest(args.memory/'summary.json') if args.memory else None,
        parent_results_sha256={r['parent']:digest(Path(r['path'])/'result.json') for r in results}))
    print(json.dumps(dict(yield_three=summary['yield_three'],yield_two=summary['yield_two'],evaluation=summary['evaluation_parents'],eta=eta)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('input','cpu','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--memory',type=Path)
    main(p.parse_args())
