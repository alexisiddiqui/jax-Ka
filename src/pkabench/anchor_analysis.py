"""Observable flank-distance diagnostics; no new pKa calculations or mask changes."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from .runtime import require_compute, atomic_json, digest


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def prepare(out):
    require_compute()
    import numpy as np
    import biotite.structure as struc
    from .prep import read_cif
    from .annotate import SITE_ATOMS
    out=Path(out); dest=out/'anchor-analysis-v2'; dest.mkdir(exist_ok=False)
    manifest=json.loads((out/'manifest.json').read_text()); geometry={}; provenance=[]
    for case in manifest['cases']:
        cid=case['complex_id']; source=out/cid/'baseline/structures'/cid/'AB.cif'
        atoms=read_cif(source); starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
        residues={}; chains={}
        for s,e in zip(starts[:-1],starts[1:]):
            r=atoms[s:e]; key=(str(r.chain_id[0]),int(r.res_id[0]),str(r.ins_code[0]).strip())
            assert key not in residues
            residues[key]=r; chains.setdefault(key[0],[]).append(key)
        seen=set()
        for name,v in case['variants'].items():
            spec=v['perturbation']
            if not spec: continue
            removed={tuple(r['key']) for r in spec['residues']}; signature=tuple(sorted(removed))
            if signature in seen: continue
            seen.add(signature); chain=next(iter(removed))[0]; keys=chains[chain]
            ix=sorted(keys.index(k) for k in removed)
            assert ix==list(range(ix[0],ix[-1]+1)), 'Noncontiguous deletion'
            flanks=[keys[i] for i in (ix[0]-1,ix[-1]+1) if 0<=i<len(keys)]
            assert len(flanks)==(1 if spec['kind']=='terminal' else 2)
            anchor=np.concatenate([residues[k].coord[np.isin(residues[k].atom_name,['N','CA','C','O'])] for k in flanks])
            geometry[cid,name]=(residues,anchor,flanks)
        provenance.append({'complex_id':cid,'reference_sha256':digest(source)})
    rows=[]
    with (out/'site_errors.csv').open() as f:
        for row in csv.DictReader(f):
            key=(row['complex_id'],row['variant'])
            if key not in geometry: continue
            residues,anchor,flanks=geometry[key]
            target=residues[row['chain'],int(row['resnum']),row['icode']]
            points=target.coord[np.isin(target.atom_name,SITE_ATOMS[row['group']])]
            assert len(points) and len(anchor)
            row['anchor_distance_A']=float(np.linalg.norm(points[:,None]-anchor[None,:],axis=-1).min())
            row['anchor_keys']=json.dumps(flanks); rows.append(row)
    write_csv(dest/'observations.csv',rows)
    atomic_json(dest/'provenance.json',{'references':provenance,'source_sha256':digest(out/'site_errors.csv'),
        'code_sha256':digest(Path(__file__)), 'distance':'Nearest target functional-group atom to visible flanking residue backbone N/CA/C/O; minimum over both flanks for internal deletions.',
        'limits':'Observed ordered deletions only; calibrated lengths are recorded in the analysis tables. No extrapolation, polymer likelihood, production masks, or validation on independent complexes. Complex bootstrap preserves repeated sites and variants. Pointwise confidence bounds are exploratory, not simultaneous guarantees. Confidence bounds require at least ten complexes.',
        'bootstrap_replicates':1000,'seed':20261003})
    subprocess.run([str(Path(os.environ['PKABENCH_RUNTIME'])/'envs/radial-plots/bin/python'),'-m','pkabench.anchor_analysis',str(dest)],check=True)


def render(dest):
    require_compute()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=Path(dest)
    with (dest/'observations.csv').open() as f: rows=list(csv.DictReader(f))
    rng=np.random.default_rng(20261003); summaries=[]; candidates=[]; shells=[]
    lengths=sorted({int(r['deleted_residues']) for r in rows if r['kind']=='terminal'})
    fig,axes=plt.subplots(2,len(lengths),figsize=(4*len(lengths),8),layout='constrained',squeeze=False)
    for col,length in enumerate(lengths):
        rr=[r for r in rows if r['kind']=='terminal' and int(r['deleted_residues'])==length and r['status']=='ok']
        x=np.array([float(r['anchor_distance_A']) for r in rr]); y=np.abs([float(r['delta_pka_error']) for r in rr])
        ids=sorted({r['complex_id'] for r in rr}); labels=np.array([ids.index(r['complex_id']) for r in rr])
        # Resample complete complexes, preserving within-complex observations.
        weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=1000)
        def bound(mask):
            order=np.argsort(y[mask]); yy=y[mask][order]; ll=labels[mask][order]
            ww=weights[:,ll]; cumulative=np.cumsum(ww,axis=1); total=cumulative[:,-1]
            valid=total>0; q=yy[np.argmax(cumulative[valid]>=.95*total[valid,None],axis=1)]
            return float(np.quantile(q,.95))
        ax=axes[0,col]; ax.scatter(x,np.maximum(y,1e-5),s=2,alpha=.15)
        for lo in range(0,80,5):
            keep=(x>=lo)&(x<lo+5); n=int(keep.sum()); nc=len(set(labels[keep]))
            if not n: continue
            upper=bound(keep) if nc>=10 and n>=20 else None
            shells.append({'length':length,'lo_A':lo,'hi_A':lo+5,'n':n,'complexes':nc,'p95_abs_error':float(np.quantile(y[keep],.95)),'p95_upper_pointwise_95':upper})
            if upper is not None: ax.plot(lo+2.5,upper,'k_',markersize=9)
        for end,color in [('N','tab:blue'),('C','tab:orange')]:
            m=np.array([r['variant'].startswith(end) for r in rr]); centres=[]; values=[]
            for lo in range(0,80,5):
                keep=m&(x>=lo)&(x<lo+5)
                if keep.sum(): centres.append(lo+2.5); values.append(np.quantile(y[keep],.95))
            ax.plot(centres,values,label=end+' terminal p95',color=color)
        ax.axhline(.1,color='black',ls=':'); ax.set(yscale='log',ylim=(1e-5,4),xlim=(0,80),title=f'{length} deleted; {len(ids)} complexes',xlabel='Visible anchor distance (Å)'); ax.legend(fontsize=8)
        series=[]
        for radius in range(0,61,2):
            keep=x>=radius; n=int(keep.sum()); nc=len(set(labels[keep]))
            if n<20: continue
            p=float(np.quantile(y[keep],.95)); upper=bound(keep) if nc>=10 else None
            record={'length':length,'radius_A':radius,'paired_retained':n,'complexes_retained':nc,'p95_abs_error':p,'p95_upper_pointwise_95':upper,'excluded_fraction':float(1-keep.mean())}
            summaries.append(record); series.append(record)
        ax=axes[1,col]
        ax.plot([s['radius_A'] for s in series],[s['p95_abs_error'] for s in series],label='Retained-site p95')
        ax.plot([s['radius_A'] for s in series],[s['p95_upper_pointwise_95'] for s in series],ls='--',label='Bootstrap upper 95%')
        ax.axhline(.1,color='black',ls=':'); ax.set(xlabel='Anchor exclusion radius (Å)',ylabel='Absolute ΔpKa error'); ax.legend(fontsize=8)
        passing=[s for i,s in enumerate(series) if all(t['p95_upper_pointwise_95'] is not None and t['p95_upper_pointwise_95']<.1 for t in series[i:])]
        local=[s for s in shells if s['length']==length and s['p95_upper_pointwise_95'] is not None]
        bad=[s for s in local if s['p95_upper_pointwise_95']>=.1]
        boundary=max((s['hi_A'] for s in bad),default=0)
        candidates.append({'length':length,'pooled_retained_radius_A':passing[0]['radius_A'] if passing else None,'last_failing_supported_shell_end_A':boundary,'supported_passing_shells_beyond':sum(s['lo_A']>=boundary for s in local),'note':'Pooled p95 can hide near-gap errors. Shell boundaries are exploratory; unsupported bins remain unknown. No production cutoff selected.'})
    axes[0,0].set_ylabel('Absolute ΔpKa error')
    fig.suptitle('Error versus observable terminal anchors — no new pKa calculations')
    for suffix in ('png','pdf'): fig.savefig(dest/f'anchor_error.{suffix}',dpi=170)
    plt.close(fig); write_csv(dest/'radius_summary.csv',summaries); write_csv(dest/'shell_summary.csv',shells)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for length in lengths:
        ss=[s for s in summaries if s['length']==length]
        ax.plot([s['radius_A'] for s in ss],[s['paired_retained'] for s in ss],label=f'{length} residues')
    ax.set(xlabel='Exclusion radius from visible anchor (Å)',ylabel='Retained paired site–deletion observations',title='Terminal anchor exclusion: retained supervision'); ax.legend()
    fig.savefig(dest/'anchor_retention.png',dpi=170); plt.close(fig)
    atomic_json(dest/'report.json',{'observations':len(rows),'candidates':candidates,'production_mask_changed':False})
    print(json.dumps({'output':str(dest),'candidates':candidates}))


if __name__=='__main__':
    render(sys.argv[1])
