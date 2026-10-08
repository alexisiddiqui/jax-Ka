"""Standalone buffer diagnostic figures; computational entry point is guarded."""
import csv
import json
import sys
from pathlib import Path
from .runtime import require_compute

def render(out):
    require_compute()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out); dest=out/'plots'; dest.mkdir(exist_ok=True)
    with (out/'site_changes.csv').open() as f: rows=list(csv.DictReader(f))
    report=json.loads((out/'report.json').read_text())
    individual=[r for r in rows if r['component_index']!='all']
    def save(fig,name):
        for ext in ('png','pdf','svg'): fig.savefig(dest/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    for variant,plotted,filename in [('individual deletions',individual,'error_vs_distance'),('all buffers removed',[r for r in rows if r['component_index']=='all'],'all_buffers_error_vs_distance')]:
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,metric,label in zip(axes,('ab_pka_change','delta_pka_change'),('Bound pKa','Bound-minus-free ΔpKa')):
            x=np.array([float(r['distance_A']) for r in plotted]); y=np.abs([float(r[metric]) for r in plotted])
            ax.scatter(x,y,s=5,alpha=.12)
            for q,color,name in ((.5,'orange','Median'),(.95,'crimson','95th percentile')):
                xx=[]; yy=[]
                for lo in range(0,80,5):
                    keep=(x>=lo)&(x<lo+5)
                    if keep.any(): xx.append(lo+2.5); yy.append(np.quantile(y[keep],q))
                ax.plot(xx,yy,color=color,label=name)
            ax.set(xlabel='Distance to removed buffer heavy atoms (Å)',ylabel=f'Absolute {label} change',xlim=(0,80),yscale='symlog'); ax.set_yscale('symlog',linthresh=1e-5)
            ax.axhline(.1,color='gray',ls=':'); ax.legend(); ax.set_ylim(bottom=0)
        fig.suptitle(f'Buffer removal: native PROPKA sensitivity — {variant}\nFinite model cutoffs; exact zeros retained; correlated sites')
        save(fig,filename)
    for field,label,name in [('exposed_fraction','Buffer exposed SASA fraction','error_vs_exposure'),('sasa_A2','Buffer bound SASA (Å²)','error_vs_sasa')]:
        fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
        for ax,radius in zip(axes.flat,(10,15,20,25)):
            rr=[r for r in individual if float(r['distance_A'])>=radius]
            ax.scatter([float(r[field]) for r in rr],np.abs([float(r['ab_pka_change']) for r in rr]),s=6,alpha=.2)
            ax.set_yscale('symlog',linthresh=1e-5); ax.set_ylim(bottom=0); ax.axhline(.1,color='gray',ls=':')
            ax.set(xlabel=label,ylabel='Absolute bound pKa change',title=f'≥{radius} Å: {len(rr)} observations, {len({r["complex_id"] for r in rr})} complexes')
        fig.suptitle('Buffer removal sensitivity versus SASA\nNative PROPKA, supported neutral buffers only; descriptive association')
        save(fig,name)
    rr=report['retention']; fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,fields in zip(axes,[('sites','interface_sites'),('pairs','interface_pairs')]):
        for field in fields: ax.plot([r['radius_A'] for r in rr],[r[field] for r in rr],marker='o',label=field.replace('_',' '))
        ax.set(xlabel='Radius excluded around all tested buffers (Å)',ylabel='Retained count'); ax.legend()
    fig.suptitle('Buffer-only retention among matched native eligible sites\nOther-component and natural-gap masks not applied')
    save(fig,'retained_sites_vs_radius')

if __name__=='__main__': render(sys.argv[1])
