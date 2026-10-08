"""Export radial diagnostics without changing teacher environments or labels."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from .runtime import atomic_json, digest, require_compute


def prepare(out):
    from .schema import read_table
    require_compute(); out=Path(out); dest=out/'plots'; dest.mkdir(exist_ok=True)
    manifest=json.loads((out/'manifest.json').read_text()); metadata={}; sites={}; duplicate=set()
    for case in manifest['cases']:
        cid=case['complex_id']; seen=set()
        sites[cid]={tuple(str(s[k]) for k in ('chain','resnum','icode','group')):s for s in read_table(out/cid/'baseline/sites.parquet')}
        for name,v in case['variants'].items():
            spec=v['perturbation']
            if not spec: continue
            keys=tuple(sorted(tuple(r['key']) for r in spec['residues']))
            if keys in seen: duplicate.add((cid,name))
            seen.add(keys)
            metadata[cid,name]=spec
    with (out/'site_errors.csv').open() as f: rows=list(csv.DictReader(f))
    enriched=[]
    for row in rows:
        if row['kind']=='repeat' or (row['complex_id'],row['variant']) in duplicate: continue
        site=sites[row['complex_id']][tuple(row[k] for k in ('chain','resnum','icode','group'))]
        row.update(retained_residue_delta_sasa_A2=site['residue_delta_sasa'],
            retained_functional_delta_sasa_A2=site['functional_delta_sasa'])
        enriched.append(row)
    with (dest/'plot_data.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(enriched[0])); writer.writeheader(); writer.writerows(enriched)
    atomic_json(dest/'provenance.json',{'source_sha256':digest(out/'site_errors.csv'),
        'duplicate_perturbations_removed':sorted(duplicate),'script_sha256':digest(Path(__file__)),
        'distance':'Nearest intact-reference deleted heavy atom to retained target functional-group atoms, Angstrom.',
        'error':'Absolute error in paired delta-pKa relative to intact reference; failures excluded from error distributions, included in structural count curves.',
        'sasa':'Deleted-residue bound SASA / SASA of the same isolated residue, probe 1.4 A, 1000 points; per-residue fractions in manifest. Target residue and functional-group free-minus-bound SASA joined from baseline sites. Absolute bound/free SASA was not saved in these tables.',
        'counts':'Site-deletion observations, not unique native sites. Identical deleted-residue sets within each reference deduplicated. Radius r excludes distance < r. Already deleted sites and artificial terminal targets are outside these counts.'})
    runtime=Path(os.environ['PKABENCH_RUNTIME']); env=runtime/'envs/radial-plots'; python=env/'bin/python'
    uv='/home/coulson/oc/lina4225/_runtime/BioFeaturisers/cuda-86/toolchain/bin/uv'
    if not python.exists():
        subprocess.run([uv,'venv','--python',sys.executable,str(env)],check=True)
        subprocess.run([uv,'pip','install','--python',str(python),'matplotlib==3.10.7'],check=True)
        lock=subprocess.check_output([uv,'pip','freeze','--python',str(python)],text=True)
        (runtime/'manifests/radial-plots.requirements.lock').write_text(lock)
    subprocess.run([str(python),'-m','pkabench.radial_plots',str(dest)],check=True)


def render(dest):
    require_compute()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    dest=Path(dest)
    with (dest/'plot_data.csv').open() as f: rows=list(csv.DictReader(f))
    classes=[('Terminal deletions',{'terminal'}),('Buried deletions',{'buried','buried_charged'})]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
    def save(fig,name):
        for suffix in ('png','pdf','svg'): fig.savefig(dest/f'{name}.{suffix}',bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.3),sharey=True,layout='constrained')
    bins=np.arange(0,85,5); summaries=[]
    for ax,(title,kinds) in zip(axes,classes):
        rr=[r for r in rows if r['kind'] in kinds and r['status']=='ok']
        x=np.array([float(r['distance']) for r in rr]); y=np.abs([float(r['delta_pka_error']) for r in rr])
        # Log-spaced error bins preserve the dense small-error distribution and large outliers.
        mesh=ax.hist2d(x,np.maximum(y,1e-5),bins=[np.arange(0,82,2),np.geomspace(1e-5,4,65)],norm=LogNorm(),cmap='Blues')
        centres=[]; med=[]; p95=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            v=y[(x>=lo)&(x<hi)]
            if not len(v): continue
            centres.append((lo+hi)/2); med.append(np.median(v)); p95.append(np.quantile(v,.95))
            summaries.append({'class':title,'lo_A':lo,'hi_A':hi,'n':len(v),'median':med[-1],'p95':p95[-1],'max':v.max()})
        ax.plot(centres,med,color='#e07a18',lw=2,label='Median (5 Å bins)')
        ax.plot(centres,p95,color='#a8203b',lw=2,label='95th percentile')
        ax.axhline(.1,color='0.4',ls=':',lw=1,label='0.1 pKa unit')
        ax.set(yscale='log',xlim=(0,80),ylim=(1e-5,4),xlabel='Distance from deleted atoms (Å)',title=f'{title}\n{len(rr):,} paired observations; {int(sum(x>80))} beyond 80 Å')
        ax.legend(fontsize=8,loc='upper right'); fig.colorbar(mesh[3],ax=ax,label='Observations per bin',shrink=.8)
    axes[0].set_ylabel('Absolute ΔpKa error (pKa units)')
    fig.suptitle('Where does deletion-induced error occur?',fontsize=17)
    fig.supxlabel('23 reference complexes • repeated sites across deletions • zero errors plotted at 10⁻⁵ • no fitted confidence interval',fontsize=9)
    save(fig,'error_vs_distance')
    with (dest/'error_distance_bins.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,layout='constrained'); countrows=[]
    radii=np.arange(0,41,.5)
    for j,(title,kinds) in enumerate(classes):
        rr=[r for r in rows if r['kind'] in kinds]
        for length in sorted({int(r['deleted_residues']) for r in rr}):
            subset=[r for r in rr if int(r['deleted_residues'])==length]
            distance=np.array([float(r['distance']) for r in subset]); paired=np.array([r['status']=='ok' for r in subset])
            allcounts=np.array([(distance<radius).sum() for radius in radii]); goodcounts=np.array([((distance<radius)&paired).sum() for radius in radii])
            line=axes[0,j].plot(radii,goodcounts,label=f'{length} residues (N={paired.sum():,} paired)')[0]
            axes[0,j].plot(radii,allcounts,color=line.get_color(),ls=':',alpha=.65)
            axes[1,j].plot(radii,100*goodcounts/max(1,paired.sum()),color=line.get_color(),label=f'{length} residues')
            for radius,alln,goodn in zip(radii,allcounts,goodcounts): countrows.append({'class':title,'deleted_residues':length,'radius_A':radius,'eligible_observations':len(subset),'paired_observations':int(paired.sum()),'excluded_eligible':int(alln),'excluded_paired':int(goodn),'retained_paired':int(paired.sum()-goodn)})
        axes[0,j].set_title(title); axes[0,j].legend(fontsize=9)
        axes[1,j].set(xlabel='Exclusion radius around deleted atoms (Å)',ylim=(0,100))
        for ax in axes[:,j]: ax.grid(alpha=.18)
    axes[0,0].set_ylabel('Number of site–deletion observations excluded')
    axes[1,0].set_ylabel('Paired observations excluded (%)')
    fig.suptitle('How much supervision does a radius exclude?',fontsize=17)
    fig.supxlabel('Solid: sites with valid reference + perturbed AB/free predictions. Dotted: all structurally eligible sites.\nCounts include repeated native sites across deletions; identical deletion sets are counted once per reference.',fontsize=9)
    save(fig,'sites_excluded_vs_radius')
    with (dest/'exclusion_counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(countrows[0])); w.writeheader(); w.writerows(countrows)
    print(json.dumps({'plots':str(dest),'rows':len(rows),'status':'complete'}))


if __name__=='__main__': render(sys.argv[1])
