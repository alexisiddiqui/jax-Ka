"""Plots of PROPKA component sensitivity, with model limits explicit."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from .runtime import require_compute,atomic_json


def launch(out):
    require_compute()
    subprocess.run([str(Path(os.environ['PKABENCH_RUNTIME'])/'envs/radial-plots/bin/python'),'-m','pkabench.component_plots',str(out)],check=True)


def render(out):
    require_compute()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out); dest=out/'plots'; dest.mkdir(exist_ok=True)
    with (out/'site_changes.csv').open() as f: rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained'); summary=[]
    for j,kind in enumerate(('ligand','metal')):
        for i,radius in enumerate((10,20)):
            rr=[r for r in rows if r['kind']==kind and float(r['distance_A'])>=radius]
            ax=axes[i,j]; x=np.array([float(r['exposed_fraction']) for r in rr]); y=np.abs([float(r['ab_pka_change']) for r in rr])
            ax.scatter(x,np.maximum(y,1e-7),s=7,alpha=.18)
            for lo,hi in ((0,.3),(.3,.7),(.7,1.01)):
                keep=(x>=lo)&(x<hi)
                if keep.any(): ax.plot((lo+min(hi,1))/2,max(float(np.quantile(y[keep],.95)),1e-7),'r_',markersize=22)
            ax.set(xlim=(-.02,1.02),yscale='log',ylim=(1e-7,10),xlabel='Component exposed fraction',ylabel='Absolute protein pKa change',title=f'{kind}: sites ≥{radius} Å; {len(rr)} observations')
            ax.axhline(.1,color='black',ls=':')
    fig.suptitle('PROPKA removal sensitivity versus component exposure\nRed marks: pooled p95; zeros plotted at 10⁻⁷. Model cutoffs limit interpretation.')
    fig.savefig(dest/'sasa_vs_change.png',dpi=180); fig.savefig(dest/'sasa_vs_change.pdf'); plt.close(fig)
    report=json.loads((out/'report.json').read_text()); selected=[s for s in report['summaries'] if s['exposure']=='all' and s['radius_A'] in (10,15,20,25)]
    # Group-specific support and component identities make sparsity visible.
    support={}
    for kind in ('ligand','metal'):
        rr=[r for r in rows if r['kind']==kind]
        support[kind]={'complexes':len({r['complex_id'] for r in rr}),'components':len({(r['complex_id'],r['component_index']) for r in rr}),'names':sorted({r['component_name'] for r in rr})}
    atomic_json(dest/'summary.json',{'support':support,'distance_summaries':selected,'limits':report['limits']}); print(json.dumps({'support':support,'distance_summaries':selected},indent=2))


if __name__=='__main__': render(sys.argv[1])
