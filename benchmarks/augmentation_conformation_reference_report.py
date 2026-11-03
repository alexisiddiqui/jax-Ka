"""Parent-balanced descriptive comparison; no learned model or surface pass gate."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np
from pkabench.runtime import atomic_json,digest,require_compute
from augmentation_conformation_reference import write_csv
from augmentation_surface_audit import checks

GROUPS=[('repair_only',0.,'Native → repaired zero','#999999'),('anm_vs_zero',.5,'0.5 Å ANM vs repaired zero','#2474b4'),
        ('anm_vs_zero',1.,'1 Å ANM vs repaired zero','#e28b20'),('anm_vs_zero',2.,'2 Å ANM vs repaired zero','#c94a45'),
        ('experimental',0.,'Experimental apo/holo (raw)','#171717')]
METRICS=['global_ca_rmsd_A','patch_ca_after_global_rmsd_A','patch_ca_after_local_rmsd_A',
         'global_surface_distance_A','local_surface_distance_A','exposed_composition_TV',
         'atom_exposure_change','local_chemistry_mismatch_fraction']


def read(path):
    with path.open() as stream:rows=list(csv.DictReader(stream))
    for row in rows:
        for key in set(row)-{'parent','group'}:row[key]=float(row[key])
    return rows


def balanced(rows,key):
    """Equal protein, then conformer, then patch weight."""
    counts=Counter((r['parent'],r['candidate']) for r in rows)
    conformers=Counter(p for p,c in counts)
    parents=len(conformers)
    values=np.array([r[key] for r in rows],float)
    weights=np.array([1/(parents*conformers[r['parent']]*counts[r['parent'],r['candidate']]) for r in rows])
    ids=np.argsort(values,kind='stable')
    return values[ids],weights[ids]


def quantile(rows,key):
    x,w=balanced(rows,key)
    if not len(x):return None
    return dict(zip(('p05','median','p95'),map(float,np.interp([.05,.5,.95],np.cumsum(w)-.5*w,x))))


def report(args):
    require_compute(threads=1);started=time.monotonic()
    sample=json.loads((args.output/'sample.json').read_text())
    if not sample['parents']:raise ValueError('no comparable experimental pairs')
    paths=[args.output/'parents'/f'{i:02d}' for i in range(len(sample['parents']))]
    results=[json.loads((p/'result.json').read_text()) for p in paths]
    if not all(r['passed'] and r['sample_sha256']==digest(args.output/'sample.json') for r in results):
        raise ValueError('incomplete/stale parent results')
    checked=0
    for unit,result in zip(sample['parents'],results):
        if unit['parent']!=result['parent']:raise ValueError('parent order changed')
        old=args.output.parent/'native-large/units'/f"{unit['index']:03d}"
        for filename,key in [('report.json','input_report_sha256'),('preparation.json','input_preparation_sha256'),('prepared.npz','input_prepared_sha256')]:
            if digest(old/filename)!=result[key]:raise ValueError('source changed: '+str(old/filename))
        for c,expected in result['candidate_coordinate_sha256'].items():
            if digest(old/f'native/c{int(c):02d}/min.npy')!=expected:raise ValueError('candidate changed')
            checked+=1
        for role,path in unit['paths'].items():
            if digest(path)!=unit['downloads'][role]['sha256']:raise ValueError('download changed')
    patches=[r for p in paths for r in read(p/'patches.csv')]
    wholes=[r for p in paths for r in read(p/'structures.csv')]
    for row in patches+wholes:
        if any(not np.isfinite(v) for k,v in row.items() if k not in ('parent','group')):
            raise ValueError('nonfinite measured feature')
    raw_patches=patches
    common_patches={};coverage=[]
    for unit,path in zip(sample['parents'],paths):
        pp=[r for r in raw_patches if r['parent']==unit['parent']]
        for radius in (6.,9.,12.):
            expected={r['anchor'] for r in pp if r['radius_A']==radius and r['group']=='experimental'}
            common=expected.copy();losses=[]
            for group,candidate in {(r['group'],r['candidate']) for r in pp}:
                actual={r['anchor'] for r in pp if r['radius_A']==radius and r['group']==group and r['candidate']==candidate}
                common &= actual
                if expected!=actual:losses.append(dict(group=group,candidate=candidate,missing_anchors=sorted(expected-actual)))
            if not common:raise ValueError('no shared measurable surface patches')
            common_patches[unit['parent'],radius]=common
            coverage.append(dict(parent=unit['parent'],radius_A=radius,experimental_anchors=len(expected),
                common_anchors=len(common),removed_anchors=sorted(expected-common),frame_losses=losses))
    patches=[r for r in raw_patches if r['anchor'] in common_patches[r['parent'],r['radius_A']]]
    rows=[];comparisons=[];rng=np.random.default_rng(20261022)
    for radius in (6.,9.,12.):
        experimental=[r for r in patches if r['group']=='experimental' and r['radius_A']==radius]
        for group,amplitude,label,color in GROUPS+[
            ('anm_vs_native',a,f'{a:g} Å ANM including repair','') for a in (.5,1.,2.)]:
            sub=[r for r in patches if r['group']==group and r['amplitude_A']==amplitude and r['radius_A']==radius]
            if not sub:continue
            rows.append(dict(group=group,amplitude_A=amplitude,radius_A=radius,
                parents=len({r['parent'] for r in sub}),frames=len({(r['parent'],r['candidate']) for r in sub}),
                patches=len(sub),metrics={key:quantile(sub,key) for key in METRICS}))
            if group=='anm_vs_zero':
                comparison=dict(amplitude_A=amplitude,radius_A=radius,metrics={})
                for key in ('local_surface_distance_A','exposed_composition_TV','atom_exposure_change'):
                    eq=quantile(experimental,key);x,w=balanced(sub,key)
                    # Descriptive overlap, not a biological acceptance criterion.
                    comparison['metrics'][key]=dict(experimental_p05=eq['p05'],experimental_p95=eq['p95'],
                        fraction_below_experimental_p05=float(w[x<eq['p05']].sum()),
                        fraction_within_experimental_p05_p95=float(w[(x>=eq['p05'])&(x<=eq['p95'])].sum()),
                        fraction_above_experimental_p95=float(w[x>eq['p95']].sum()))
                # Paired parent means: candidate/patch multiplicity never inflates n.
                changes=[];paired=[]
                for parent in sorted({r['parent'] for r in experimental}):
                    a=[r for r in sub if r['parent']==parent]
                    b=[r for r in experimental if r['parent']==parent]
                    if a and b:
                        av,aw=balanced(a,'local_surface_distance_A');bv,bw=balanced(b,'local_surface_distance_A')
                        am,bm=float(av@aw),float(bv@bw)
                        changes.append(am-bm)
                        paired.append(dict(parent=parent,anm_mean=am,experimental_mean=bm))
                d=np.asarray(changes)
                bootstrap=d[rng.integers(0,len(d),(5000,len(d)))].mean(1)
                comparison.update(paired_parent_mean_surface_difference_A=float(d.mean()),
                    paired_parent_bootstrap_95_CI_A=np.quantile(bootstrap,[.025,.975]).tolist(),paired=paired)
                comparisons.append(comparison)
    sensitivity=[]
    for p in paths:
        if (p/'sensitivity.csv').exists():sensitivity+=read(p/'sensitivity.csv')
    lookup={(r['parent'],r['group'],r['candidate'],r['radius_A'],r['anchor']):r for r in patches}
    resolution={}
    for key in ('local_surface_distance_A','exposed_composition_TV','atom_exposure_change'):
        changes=[abs(r[key]-lookup[r['parent'],r['group'],r['candidate'],r['radius_A'],r['anchor']][key]) for r in sensitivity
                 if (r['parent'],r['group'],r['candidate'],r['radius_A'],r['anchor']) in lookup]
        resolution[key]=dict(comparisons=len(changes),median_abs_change=float(np.median(changes)),p95_abs_change=float(np.quantile(changes,.95))) if changes else None
    numerical=checks()
    summary=dict(parents=len(results),independent_proteins=len({r['protein'] for r in sample['parents']}),
        requested_pairs=sample['requested_pairs'],available_links=sample['available_links'],
        excluded=sample['exclusions'],sampling=sample['selection'],table=rows,comparisons=comparisons,
        numerical_controls=numerical,resolution_256_vs_512=resolution,
        verified_coordinate_files=checked,identical_patch_coverage_across_arms=True,patch_coverage=coverage,
        geometry_candidates=sum(len(r['geometry']) for r in results),
        geometry_valid_candidates=sum(c['geometry_valid'] for r in results for c in r['geometry']),
        parent_wall_s=[r['wall_s'] for r in results],wall_s=time.monotonic()-started,
        limitations=[
            'Small paired PINDER subset of the saved pilot; not population-representative or a biological validity certification',
            'Experimental apo means independently deposited unbound form assigned by PINDER, not guaranteed ligand-free',
            '100% identity over aligned resolved residues with >=90% coverage of each; unmatched-atom and chain-break neighbourhoods masked',
            'Whole isolated protein surfaces retain all atoms for occlusion; binding partner removed from holo for the same environment',
            '6/9/12 A Euclidean radii about fixed surface anchors; distinct from earlier geodesic SES patches',
            'Solvent-accessible surface with residue-class chemistry and exposed amino-acid composition; no electrostatic potential',
            'Experimental raw structures include crystal, resolution, side-chain and preparation differences, not just motion',
            'Only well-resolved comparable neighbourhoods are analysed; highly flexible unresolved loops may be underrepresented',
            'Summary distributions use the common measurable anchor intersection across all geometry-valid views; raw rows and any disappearing small-patch samples are retained separately',
            'ANM vs zero removes shared source-repair baseline; ANM vs native and repair-only are also reported to expose this difference',
            'Paired experimental structures use one conformational difference per protein, not a complete equilibrium ensemble',
            'Overlap of these summary metrics does not establish realistic atomic motions or usefulness to a learned representation',
            'No surface-preservation rejection threshold is applied; geometry-valid candidates at all three amplitudes retained'])
    atomic_json(args.output/'summary.json',summary)
    write_csv(args.output/'patches.csv',patches);write_csv(args.output/'raw-patches.csv',raw_patches);write_csv(args.output/'structures.csv',wholes)
    atomic_json(args.output/'provenance.json',dict(sample_sha256=digest(args.output/'sample.json'),
        report_sha256=digest(__file__),surface_sha256=digest(Path(__file__).with_name('augmentation_surface_audit.py')),
        parent_result_sha256={r['parent']:digest(p/'result.json') for r,p in zip(results,paths)}))
    table=['Experimental conformations versus ANM','=====================================','',
        f"{len(results)} independent matched proteins; five modes, eight saved directions per amplitude.",
        'Surfaces: whole-protein solvent-accessible spheres; nested Euclidean patch radii. No learned model or surface pass gate.','',
        'Parent-balanced distribution medians','-----------------------------------','',
        'Radius | View | Global CA RMSD | Local CA RMSD | Local surface distance | Composition TV | Exposure change','']
    for row in rows:
        m=row['metrics'];label=row['group'] if row['group']!='anm_vs_zero' else f"ANM {row['amplitude_A']:g} A"
        table.append(f"{row['radius_A']:g} A | {label} | "+' | '.join(f"{m[k]['median']:.4f}" for k in
            ('global_ca_rmsd_A','patch_ca_after_local_rmsd_A','local_surface_distance_A','exposed_composition_TV','atom_exposure_change')))
    nine={(r['group'],r['amplitude_A']):r['metrics'] for r in rows if r['radius_A']==9.}
    table+=['','Repair attribution at 9 A','-------------------------','',
        'Experimental pairs compare raw deposited conformations. ANM-versus-zero compares two repaired states. These have different preparation baselines; the full native-to-candidate comparisons above must also be consulted.',
        f"Native-to-zero repair alone has median local surface distance {nine['repair_only',0.]['local_surface_distance_A']['median']:.4f} A; experimental raw apo/holo pairs have {nine['experimental',0.]['local_surface_distance_A']['median']:.4f} A.",
        'For 0.5/1/2 A targets, median total candidate-versus-native local surface distances are '+
        '/'.join(f"{nine['anm_vs_native',a]['local_surface_distance_A']['median']:.4f}" for a in (.5,1.,2.))+' A. Distances are not additive: do not subtract distribution medians to estimate an ANM contribution.',
        'The larger ANM targets add more variation between consistently repaired views. Similarity of a few surface statistics is descriptive evidence, not a biological acceptance filter.',
        '','Sampling sensitivity (256 versus 512 points per atom)','-----------------------------------------------------','',json.dumps(resolution,indent=2),
            '','Limitations','-----------','']+['* '+s for s in summary['limitations']]
    (args.output/'report.rst').write_text('\n'.join(table)+'\n')
    plots(args.output,patches,wholes)
    print(json.dumps(dict(parents=len(results),geometry_valid=summary['geometry_valid_candidates'],output=str(args.output))),flush=True)


def plots(output,patches,wholes):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for column,radius in enumerate((6.,9.,12.)):
        for group,amp,label,color in GROUPS:
            sub=[r for r in patches if r['group']==group and r['amplitude_A']==amp and r['radius_A']==radius]
            for row,key in enumerate(('local_surface_distance_A','exposed_composition_TV')):
                if not sub:continue
                x,w=balanced(sub,key);axes[row,column].plot(x,np.cumsum(w),color=color,label=label,lw=2 if group=='experimental' else 1.5)
        axes[0,column].set(title=f'{radius:g} Å surface patch radius',xlabel='Locally aligned SAS distance (Å)',ylabel='Parent-balanced cumulative fraction')
        axes[1,column].set(xlabel='Change in exposed amino-acid composition (TV)',ylabel='Parent-balanced cumulative fraction')
    axes[0,0].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_ylim(0,1)
    fig.suptitle('Five-mode ANM versus independent experimental conformations\nSame proteins and patch anchors; geometry-screened candidates; no surface rejection')
    fig.savefig(output/'surface-distributions.png',dpi=170);fig.savefig(output/'surface-distributions.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for group,amp,label,color in GROUPS:
        ws=[r for r in wholes if r['group']==group and r['amplitude_A']==amp]
        x,w=balanced(ws,'global_ca_rmsd_A');axes[0].plot(x,np.cumsum(w),label=label,color=color)
        sub=[r for r in patches if r['group']==group and r['amplitude_A']==amp and r['radius_A']==9.]
        for parent in sorted({r['parent'] for r in sub}):
            pp=[r for r in sub if r['parent']==parent]
            axes[1].scatter(np.mean([r['global_surface_distance_A'] for r in pp]),np.mean([r['local_surface_distance_A'] for r in pp]),color=color,s=22,alpha=.7)
    axes[0].set(xlabel='Global aligned CA RMSD (Å)',ylabel='Parent-balanced cumulative fraction');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='9 Å patch SAS distance after global alignment (Å)',ylabel='After local patch alignment (Å)')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(output/'global-vs-local.png',dpi=170);fig.savefig(output/'global-vs-local.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for ax,key,title in zip(axes,('patch_ca_after_local_rmsd_A','local_surface_distance_A','exposed_composition_TV'),
                            ('Local CA RMSD (Å)','Local SAS distance (Å)','Exposed composition change (TV)')):
        for group,amp,label,color in GROUPS:
            sub=[r for r in patches if r['radius_A']==9. and r['group']==group and r['amplitude_A']==amp]
            x,w=balanced(sub,key);ax.plot(x,np.cumsum(w),color=color,label=label,lw=2 if group=='experimental' else 1.5)
            if group=='anm_vs_zero':
                native=[r for r in patches if r['radius_A']==9. and r['group']=='anm_vs_native' and r['amplitude_A']==amp]
                x,w=balanced(native,key);ax.plot(x,np.cumsum(w),color=color,ls='--',alpha=.8,label=f'{amp:g} Å ANM vs native')
        ax.set(xlabel=title,ylabel='Parent-balanced cumulative fraction',ylim=(0,1));ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('9 Å patches: distinguish ANM variation from preparation changes\nSolid ANM curves compare repaired states; dashed curves include the source repair')
    fig.savefig(output/'repair-attribution.png',dpi=170);fig.savefig(output/'repair-attribution.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);report(p.parse_args())
