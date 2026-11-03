"""Paired parent-level comparison of 5/10-mode surface preservation rates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from pkabench.runtime import require_compute,atomic_json,digest
from augmentation_surface_audit import fit,rms
from augmentation_masif_report import write_csv,quantile


def report(args):
    require_compute(threads=1)
    for parent in (args.output/'parents').iterdir():
        if (parent/'result.json').exists():continue
        shards=sorted(parent.glob('shard-*/result.json'))
        if len(shards)!=4:continue
        parts=[json.loads(p.read_text()) for p in shards]
        metrics=sorted([m for part in parts for m in part['metrics']],key=lambda m:m['candidate'])
        if [m['candidate'] for m in metrics]!=list(range(1,25)):raise ValueError('incomplete or overlapping candidate shards')
        coordinates={};reference=None;direction_ref=None
        for path in shards:
            with np.load(path.parent/'modes.npz') as z:
                if direction_ref is None:direction_ref=z['directions'].copy()
                else:np.testing.assert_allclose(z['directions'],direction_ref,atol=1e-6,rtol=0)
            with np.load(path.parent/'coordinates.npz') as z:
                if reference is None:reference=z['reference'].copy()
                else:np.testing.assert_array_equal(z['reference'],reference)
                coordinates.update({k:z[k].copy() for k in z.files if k!='reference'})
        with np.load(parent/'modes.npz') as original:np.testing.assert_allclose(original['directions'],direction_ref,atol=1e-6,rtol=0)
        np.savez_compressed(parent/'coordinates.npz',reference=reference,**coordinates)
        merged=dict(parts[0],metrics=metrics,geometry_frames=sum(p['geometry_frames'] for p in parts),
            electrostatic_frames=sum(p['electrostatic_frames'] for p in parts),
            surface_tool_failures=[f for part in parts for f in part.get('surface_tool_failures',[])],
            shard_count=4,merged_shard_sha256={str(p.relative_to(parent)):digest(p) for p in shards})
        atomic_json(parent/'result.json',merged)
    results=[json.loads(p.read_text()) for p in sorted((args.output/'parents').glob('*/result.json'))]
    if len(results)!=12 or not all(r['passed'] for r in results):raise ValueError('all 12 parents must finish')
    with (args.baseline/'candidate-filter.csv').open() as f:old=list(csv.DictReader(f))
    amplitudes=(.5,1.,2.);new=[m for r in results for m in r['metrics']]
    if len(new)!=288 or len(old)!=288:raise ValueError('expected 288 candidates in each mode-count arm')
    rows=[];rng=np.random.default_rng(20261022)
    for amplitude in amplitudes:
        five=[r for r in old if float(r['amplitude_A'])==amplitude];ten=[r for r in new if r['amplitude_A']==amplitude]
        pp=[]
        for parent in sorted(r['parent'] for r in results):
            a=[r['accepted']=='True' for r in five if r['parent']==parent]
            b=[r['surface_pass'] for r in ten if r['parent']==parent]
            if len(a)!=8 or len(b)!=8:raise ValueError('paired cohort changed')
            pp.append(np.mean(b)-np.mean(a))
        pp=np.asarray(pp);bootstrap=pp[rng.integers(0,12,(5000,12))].mean(1)*100
        rows.append(dict(amplitude_A=amplitude,five_modes_pass=sum(r['accepted']=='True' for r in five),
            ten_modes_pass=sum(r['surface_pass'] for r in ten),candidates_per_arm=96,
            five_modes_geometry_pass=sum(r['geometry_valid']=='True' for r in five),
            ten_modes_geometry_pass=sum(r['geometry_valid'] for r in ten),
            five_modes_pass_percent=100*np.mean([r['accepted']=='True' for r in five]),
            ten_modes_pass_percent=100*np.mean([r['surface_pass'] for r in ten]),
            ten_modes_surface_tool_failures=sum(r['status']=='surface_tool_failed' for r in ten),
            ten_modes_pass_percent_upper_if_tool_failures_pass=100*np.mean([r['surface_pass'] or r['status']=='surface_tool_failed' for r in ten]),
            difference_percentage_points=float(pp.mean()*100),paired_parent_bootstrap_95_CI_percentage_points=np.quantile(bootstrap,[.025,.975]).tolist()))
    # Actual aligned pairwise diversity of geometry-valid conformers at each amplitude.
    diversity=[]
    for result in results:
        token=hashlib.sha256(result['parent'].encode()).hexdigest()[:16]
        with np.load(args.output/'parents'/token/'coordinates.npz') as z:ten={k:z[k].copy() for k in z.files}
        with np.load(args.exhaustive/'parents'/token/'candidate-ca.npz') as z:five={k:z[k].copy() for k in z.files}
        index=next(r['source_index'] for r in json.loads((args.runtime/'results/sample.json').read_text())['parents'] if r['parent']==result['parent'])
        info=json.loads((args.runtime.parent/f'native-large/units/{index:03d}/preparation.json').read_text())
        heavy=np.flatnonzero(np.asarray(info['is_native'])&np.asarray(info['is_heavy']))
        local={int(v):i for i,v in enumerate(heavy)};caids=np.array([local[int(v)] for v in info['ca_ids']])
        for amplitude in amplitudes:
            start=int(amplitudes.index(amplitude))*8+1
            for label,coords in [('5',five),('10',ten)]:
                valid=({int(r['candidate']) for r in old if r['parent']==result['parent'] and r['geometry_valid']=='True'} if label=='5' else {r['candidate'] for r in result['metrics'] if r['geometry_valid']})
                ids=[c for c in range(start,start+8) if c in valid];dd=[]
                for i,c in enumerate(ids):
                    for d in ids[:i]:
                        x,y=coords[str(c)],coords[str(d)]
                        if label=='10':x,y=x[caids],y[caids]
                        rotation,translation=fit(x,y);dd.append(rms(x@rotation+translation,y))
                if dd:diversity.append(dict(parent=result['parent'],modes=int(label),amplitude_A=amplitude,median_pairwise_ca_rmsd_A=float(np.median(dd))))
    diversity_summary={str(a):{str(n):float(np.median([r['median_pairwise_ca_rmsd_A'] for r in diversity if r['amplitude_A']==a and r['modes']==n])) for n in (5,10)} for a in amplitudes}
    columns=sorted({k for r in new for k in r});new=[{k:r.get(k) for k in columns} for r in new]
    write_csv(args.output/'candidates.csv',new);write_csv(args.output/'diversity.csv',diversity)
    summary=dict(passed=True,parents=12,table=rows,geometry_frames=sum(r['geometry_frames'] for r in results),
        electrostatic_frames=sum(r['electrostatic_frames'] for r in results),median_parent_pairwise_ca_rmsd_A=diversity_summary,
        max_five_mode_target_reconstruction_error_A=max(r['five_mode_target_max_error_A'] for r in results),
        ten_mode_anm_build_and_solve_s=dict(median=float(np.median([r['anm_s'] for r in results])),max=max(r['anm_s'] for r in results)),
        mean_added_five_mode_fraction_before_normalization=float(np.mean([x for r in results for x in r['extra_mode_fraction']])),
        limitations=['Same 12 enriched proteins and frozen 18-parent calibrated 9 A surface gate; not a population estimate',
            'Eight paired directions per protein and amplitude; only 12 independent protein clusters',
            'Paired-parent bootstrap does not include uncertainty from hardware/cleanup sensitivity measured by the five-mode controls',
            'Original five-mode random components retained; five higher components added, then target RMSD renormalized',
            'Native transport, 100-step stereo-restrained CPU minimization, geometry gates and source charge inventories unchanged',
            'Non-electrostatic rejection is definitive; APBS evaluated for every candidate still capable of passing, with no three-view early stop',
            'General CPU nodes used for ten-mode arm because AMD nodes unavailable; runtime is not a matched hardware speed comparison',
            'External surface-tool failures, if any, are counted separately and cannot be interpreted as physical surface rejection',
            'Fixed-reference anchor offsets used where the five-mode candidate had failed geometry and had no surface job'])
    if args.control:
        controls=[json.loads(p.read_text()) for p in sorted((args.control/'parents').glob('*/result.json'))]
        if len(controls)!=12 or sum(len(r['metrics']) for r in controls)!=36:raise ValueError('all 36 same-hardware controls must finish')
        lookup={(r['parent'],int(r['candidate'])):r['accepted']=='True' for r in old}
        new_lookup={(r['parent'],r['candidate']):r['surface_pass'] for r in new}
        cm=[m for r in controls for m in r['metrics']]
        summary['same_hardware_five_mode_controls']=dict(candidates=len(cm),
            classification_agreement=sum(r['surface_pass']==lookup[r['parent'],r['candidate']] for r in cm),
            median_heavy_coordinate_rmsd_A=float(np.median([r['old_five_heavy_rmsd_A'] for r in cm])),
            max_heavy_coordinate_difference_A=max(r['old_five_max_heavy_difference_A'] for r in cm),
            by_amplitude={str(a):dict(original_pass=sum(lookup[r['parent'],r['candidate']] for r in cm if r['amplitude_A']==a),
                repeated_pass=sum(r['surface_pass'] for r in cm if r['amplitude_A']==a),ten_mode_matched_pass=sum(new_lookup[r['parent'],r['candidate']] for r in cm if r['amplitude_A']==a),candidates=12) for a in amplitudes})
        summary['limitations'].append('Same-hardware controls repeat one direction per amplitude; capped minimization and mesh sensitivity can change individual gate decisions')
    continuous={5:[],10:[]};parent_ids={r['parent'] for r in results}
    with (args.baseline/'patches.csv').open() as f:
        for r in csv.DictReader(f):
            if r['parent'] in parent_ids and r['role'].startswith('c'):continuous[5].append(r)
    for parent in (args.output/'parents').iterdir():
        files=sorted(parent.glob('shard-*/patches.csv')) if (parent/'shard-0').exists() else [parent/'patches.csv']
        for path in files:
            if path.exists():
                with path.open() as f:continuous[10].extend(csv.DictReader(f))
    summary['continuous_local_surface_metrics']={}
    for amplitude in amplitudes:
        summary['continuous_local_surface_metrics'][str(amplitude)]={}
        for mode,rr in continuous.items():
            selected=[]
            for r in rr:
                candidate=int(r['role'][1:]);amp=amplitudes[(candidate-1)//8]
                if amp!=amplitude:continue
                selected.append(dict(r,**{k:float(r[k]) for k in ('surface_distance_A','hbond_rmse','hydropathy_rmse','residue_exposure_TV')}))
            summary['continuous_local_surface_metrics'][str(amplitude)][str(mode)]={k:quantile(selected,k) for k in ('surface_distance_A','hbond_rmse','hydropathy_rmse','residue_exposure_TV')}
    atomic_json(args.output/'summary.json',summary)
    atomic_json(args.output/'provenance.json',dict(code_sha256=digest(__file__),baseline_sha256=digest(args.baseline/'candidate-filter.csv'),
        calibration_sha256=digest(args.baseline/'summary.json'),parent_result_sha256={r['parent']:digest(args.output/'parents'/hashlib.sha256(r['parent'].encode()).hexdigest()[:16]/'result.json') for r in results}))
    text=['Five versus ten ANM modes','=========================','','Same RMSD amplitudes, paired random coefficients, repair and frozen surface thresholds.','']
    for row in rows:
        text.append(f"* {row['amplitude_A']:g} A: five modes {row['five_modes_pass']}/96 ({row['five_modes_pass_percent']:.1f}%); ten modes {row['ten_modes_pass']}/96 ({row['ten_modes_pass_percent']:.1f}%). Unevaluable surface cases: {row['ten_modes_surface_tool_failures']}. Paired difference {row['difference_percentage_points']:+.1f} percentage points; parent-bootstrap CI {row['paired_parent_bootstrap_95_CI_percentage_points']}.")
    if args.control:text += ['', 'Same-hardware control agreement: '+str(summary['same_hardware_five_mode_controls']['classification_agreement'])+'/36.', '']
    text += ['','Limitations','-----------','']+['* '+s for s in summary['limitations']]
    (args.output/'report.rst').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    xx=np.arange(3)
    axes[0].bar(xx-.18,[r['five_modes_pass_percent'] for r in rows],width=.36,label='5 modes')
    axes[0].bar(xx+.18,[r['ten_modes_pass_percent'] for r in rows],width=.36,label='10 modes')
    axes[0].set(xticks=xx,xticklabels=['0.5','1','2'],xlabel='Target CA RMSD (Å)',ylabel='Candidates passing fixed surface gate (%)',ylim=(0,100))
    for n in (5,10):axes[1].plot(amplitudes,[diversity_summary[str(a)][str(n)] for a in amplitudes],marker='o',label=f'{n} modes')
    axes[1].set(xlabel='Target CA RMSD (Å)',ylabel='Median within-parent pairwise CA RMSD (Å)')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(args.output/'modes5-vs10.png',dpi=180);fig.savefig(args.output/'modes5-vs10.svg');plt.close(fig)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('output','baseline','exhaustive','runtime'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--control',type=Path)
    report(p.parse_args())
