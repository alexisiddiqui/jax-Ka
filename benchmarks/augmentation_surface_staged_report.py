"""Measured staged-filter throughput, parity, memory and population ETA."""
import json
from pathlib import Path
import numpy as np
from pkabench.runtime import atomic_json,digest


def report(output,runtime,calibration):
    execution=json.loads((output/'execution.json').read_text());results=execution['results']
    population=json.loads((runtime.parent/'native-large/population.json').read_text())
    counts={}
    for key,count in population['strata'].items():
        dataset,split,b=key.split('|');cell=dataset+'|'+b;counts[cell]=counts.get(cell,0)+count
    costs={};interpolated=[]
    for cell in counts:
        dataset,b=cell.split('|');b=int(b)
        matches=[r for r in results if r['dataset']==dataset and r['size_bin']==b]
        if not matches:
            lower=next(r for r in results if r['dataset']==dataset and r['size_bin']==1)
            upper=next(r for r in results if r['dataset']==dataset and r['size_bin']==3)
            costs[cell]=.5*(lower['wall_s']+upper['wall_s']);interpolated.append(cell)
        else:costs[cell]=float(np.mean([r['wall_s'] for r in matches]))
    surface_worker_s=sum(counts[c]*costs[c] for c in counts)
    baseline=json.loads((runtime.parent/'native-large/unit-summary.json').read_text())
    coordinate_cpu_s=sum(r['weight']*r['production_s'] for r in baseline)
    peak=max(r['peak_tree_rss_MiB'] for r in results)
    reserved=max(4,int(np.ceil(peak*1.25/(2048*.9))))
    etas={}
    for cpus in (240,288,380):
        workers=cpus//reserved
        ideal=(surface_worker_s/workers+coordinate_cpu_s/cpus)/86400
        etas[str(cpus)]=dict(workers=workers,reserved_cores_per_worker=reserved,
            staged_filter_ideal_days=surface_worker_s/workers/86400,
            coordinate_generation_including_ANM_ideal_days=coordinate_cpu_s/cpus/86400,
            combined_ideal_days=ideal,combined_planning_days_with_30pct_margin=ideal*1.3)
    old=json.loads((calibration/'summary.json').read_text())
    summary=dict(passed=execution['passed'],parents=len(results),elapsed_s=execution['elapsed_s'],
        selected_count_histogram={str(n):sum(len(r['selected'])==n for r in results) for n in (1,2,3)},
        exhaustive_selections_match=all(r['exhaustive_selection_match'] for r in results),
        geometry_frames=sum(r['geometry_frames'] for r in results),electrostatic_frames=sum(r['electrostatic_frames'] for r in results),
        full_surface_frames_without_deferral=sum(r['geometry_frames'] for r in results),
        peak_worker_tree_rss_MiB=peak,eta=etas,weighted_parent_filter_s=surface_worker_s/sum(counts.values()),
        measured_cost_s_by_dataset_length_bin=costs,interpolated_bins=interpolated,
        prior_simulated_planning_days={c:old['memory_safe_eta'][c]['staged_simulation_planning_days_30pct_margin'] for c in etas},
        results=results,limitations=['12 held-out, size/response-enriched proteins; no population yield estimate',
            'Geometry and electrostatics rebuilt cold; no feature cache reused for decisions',
            'Uses fixed input anchor targets from the exhaustive pilot for a matched comparison',
            'Benchmark starts from existing repaired coordinates; ANM and repair costs added from earlier 500-parent baseline',
            'Middle length-bin costs interpolated between bins 1 and 3; only one protein per measured dataset/length cell',
            'ETA assumes measured four-core speed, 2 GB/core, 25% worker memory margin and 10% node reserve',
            '30% planning allowance is not a confidence interval; queues and data transfers excluded',
            'High-concurrency production I/O, packed output, and pipeline overlap are not benchmarked'])
    atomic_json(output/'summary.json',summary)
    atomic_json(output/'provenance.json',dict(report_code_sha256=digest(__file__),execution_sha256=digest(output/'execution.json'),
        design_sha256=digest(output/'design.json'),population_sha256=digest(runtime.parent/'native-large/population.json'),
        coordinate_baseline_sha256=digest(runtime.parent/'native-large/unit-summary.json')))
    text=['Measured deferred-APBS surface filter','====================================','',
        f"Completed {len(results)} parents in {execution['elapsed_s']/60:.2f} minutes.",
        f"Geometry frames: {summary['geometry_frames']}; electrostatics frames: {summary['electrostatic_frames']}.",
        f"All selections match exhaustive filtering: {summary['exhaustive_selections_match']}.",
        f"Peak worker process-tree RSS: {peak:.0f} MiB; {reserved} reserved cores per worker.",'','Full-run ETA','------------','']
    text += [f"* {c} CPUs / {e['workers']} workers: ideal {e['combined_ideal_days']:.2f} days, planning {e['combined_planning_days_with_30pct_margin']:.2f} days." for c,e in etas.items()]
    text += ['','Limitations','-----------','']+['* '+v for v in summary['limitations']]
    (output/'report.rst').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for dataset in ('atlas','pinder','pkpdb'):
        rr=sorted([r for r in results if r['dataset']==dataset],key=lambda r:r['n_residues'])
        axes[0].plot([r['n_residues'] for r in rr],[r['wall_s']/60 for r in rr],marker='o',label=dataset)
        axes[1].scatter([r['n_residues'] for r in rr],[r['peak_tree_rss_MiB']/1024 for r in rr],label=dataset)
    axes[0].set(xlabel='Protein residues',ylabel='Cold staged-filter time (minutes)')
    axes[1].set(xlabel='Protein residues',ylabel='Peak worker process-tree RSS (GiB)')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'staged-runtime.png',dpi=180);fig.savefig(output/'staged-runtime.svg');plt.close(fig)
    print(json.dumps(dict(passed=True,parents=len(results),elapsed_s=execution['elapsed_s'],electrostatics=summary['electrostatic_frames'],eta=etas)),flush=True)
