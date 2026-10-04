"""Internal-deletion diagnostics from the existing observable-anchor table."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from .runtime import require_compute, atomic_json, digest


def prepare(out):
    require_compute(); out=Path(out); dest=out/'internal-anchor-analysis-v1'; dest.mkdir(exist_ok=False)
    source=out/'anchor-analysis-v2/observations.csv'
    # The existing table already checks contiguous deletions and two flanks.
    manifest=json.loads((out/'manifest.json').read_text()); membership={}
    for case in manifest['cases']:
        signatures={}
        for name,v in case['variants'].items():
            spec=v['perturbation']
            if not spec or spec['kind']=='terminal': continue
            signature=tuple(sorted(tuple(r['key']) for r in spec['residues']))
            category=('buried_charged' if spec['kind']=='buried_charged' else spec['kind'])+str(spec['length'])
            signatures.setdefault(signature,set()).add(category)
        for name,v in case['variants'].items():
            spec=v['perturbation']
            if spec and spec['kind']!='terminal':
                signature=tuple(sorted(tuple(r['key']) for r in spec['residues']))
                membership[case['complex_id'],name]=sorted(signatures[signature])
    with source.open() as f: rows=[r for r in csv.DictReader(f) if r['kind'] not in ('terminal','repeat')]
    for row in rows:
        assert len(json.loads(row['anchor_keys']))==2
        row['categories']='|'.join(membership[row['complex_id'],row['variant']])
    with (dest/'observations.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    atomic_json(dest/'provenance.json',{'source_sha256':digest(source),'manifest_sha256':digest(out/'manifest.json'),
        'code_sha256':digest(Path(__file__)),'bootstrap_replicates':1000,'seed':20261004,
        'distance':'Minimum distance from target functional atoms to backbone N/CA/C/O of either visible flank.',
        'limits':'Paired delta-pKa; observed resolved deletions of length 1 or 3 only. Categories overlap when a single deletion satisfies multiple designs; unique perturbations counted once within each category. Near/remote refer to deleted-region partner distance <=5 / >=15 A. Pointwise complex-bootstrap bounds are exploratory, not simultaneous or held-out validation. No long-tail inference or production mask change.'})
    subprocess.run([str(Path(os.environ['PKABENCH_RUNTIME'])/'envs/radial-plots/bin/python'),'-m','pkabench.internal_anchors',str(dest)],check=True)


def render(dest):
    require_compute()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=Path(dest)
    with (dest/'observations.csv').open() as f: rows=list(csv.DictReader(f))
    categories=['buried1','buried_charged1','buried3','internal_near3','internal_remote3']
    rng=np.random.default_rng(20261004); shells=[]; counts=[]; reports=[]
    fig,axes=plt.subplots(2,5,figsize=(20,8),layout='constrained')
    for col,category in enumerate(categories):
        allrows=[r for r in rows if category in r['categories'].split('|')]
        rr=[r for r in allrows if r['status']=='ok']
        if not rr:
            reports.append({'category':category,'paired_observations':0}); continue
        ids=sorted({r['complex_id'] for r in rr}); labels=np.array([ids.index(r['complex_id']) for r in rr])
        x=np.array([float(r['anchor_distance_A']) for r in rr]); y=np.abs([float(r['delta_pka_error']) for r in rr])
        weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=1000)
        def bound(mask):
            order=np.argsort(y[mask]); yy=y[mask][order]; ll=labels[mask][order]
            cw=np.cumsum(weights[:,ll],axis=1); total=cw[:,-1]; valid=total>0
            q=yy[np.argmax(cw[valid]>=.95*total[valid,None],axis=1)]
            return float(np.quantile(q,.95))
        ax=axes[0,col]; ax.scatter(x,np.maximum(y,1e-5),s=3,alpha=.16)
        local=[]
        for lo in range(0,80,5):
            mask=(x>=lo)&(x<lo+5); n=int(mask.sum()); nc=len(set(labels[mask]))
            if not n: continue
            upper=bound(mask) if nc>=10 and n>=20 else None
            row={'category':category,'lo_A':lo,'hi_A':lo+5,'n':n,'complexes':nc,'p95':float(np.quantile(y[mask],.95)),'upper95':upper}
            shells.append(row); local.append(row)
        ax.plot([s['lo_A']+2.5 for s in local],[s['p95'] for s in local],label='Local p95')
        supported=[s for s in local if s['upper95'] is not None]
        ax.plot([s['lo_A']+2.5 for s in supported],[s['upper95'] for s in supported],'k_',label='Bootstrap upper 95%')
        ax.axhline(.1,color='black',ls=':'); ax.set(yscale='log',ylim=(1e-5,4),xlim=(0,80),title=f'{category}\n{len(ids)} complexes; {len(rr):,} observations',xlabel='Nearest visible flank (Å)'); ax.legend(fontsize=7)
        alldist=np.array([float(r['anchor_distance_A']) for r in allrows]); series=[]
        for radius in range(0,61,2):
            mask=x>=radius
            r={'category':category,'radius_A':radius,'paired_retained':int(mask.sum()),'structural_retained':int((alldist>=radius).sum()),'complexes_retained':len(set(labels[mask]))}
            series.append(r); counts.append(r)
        axes[1,col].plot([s['radius_A'] for s in series],[s['paired_retained'] for s in series],label='Valid paired predictions')
        axes[1,col].plot([s['radius_A'] for s in series],[s['structural_retained'] for s in series],ls=':',label='Structurally eligible')
        axes[1,col].set(xlabel='Anchor exclusion radius (Å)',ylabel='Retained observations'); axes[1,col].legend(fontsize=7)
        bad=[s for s in supported if s['upper95']>=.1]; boundary=max((s['hi_A'] for s in bad),default=None)
        reports.append({'category':category,'complexes':len(ids),'paired_observations':len(rr),'structural_observations':len(allrows),
            'last_failing_supported_band_end_A':boundary,'supported_bands':len(supported),
            'passing_supported_bands_after':sum(s['lo_A']>=(boundary or 0) and s['upper95']<.1 for s in supported),
            'unsupported_populated_bands':[s['lo_A'] for s in local if s['upper95'] is None]})
    axes[0,0].set_ylabel('Absolute ΔpKa error')
    fig.suptitle('Internal deletions: error and retained sites versus visible flanks')
    for suffix in ('png','pdf'): fig.savefig(dest/f'internal_anchor_error.{suffix}',dpi=170)
    plt.close(fig)
    for name,data in [('shell_summary',shells),('retention',counts)]:
        with (dest/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    report={'categories':reports,'unique_structural_observations':len(rows),'production_mask_changed':False,'cutoffs_selected':False}
    atomic_json(dest/'report.json',report); print(json.dumps(report,indent=2))


if __name__=='__main__': render(sys.argv[1])
