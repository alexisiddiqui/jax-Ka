"""Standalone smoke diagnostics, never a production performance claim."""
import csv
import json
import sys
from pathlib import Path
from .runtime import require_compute


def render(campaign):
    require_compute()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    campaign=Path(campaign); out=campaign/'plots'; out.mkdir(exist_ok=True)
    summary=json.loads((campaign/'scores_set1.json').read_text())
    methods=['propka','jaxka','pkai','pkai_plus','null']
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,split in zip(axes,('train','val','test')):
        for i,method in enumerate(methods):
            rr=[r for r in summary if r['method']==method and r['split']==split and r['scope']=='pairwise' and r['subset']=='interface']
            if not rr or rr[0]['skill'] is None: continue
            r=rr[0]; ci=r['skill_ci95']; ax.plot(r['skill'],i,'o',color='navy')
            if ci: ax.plot(ci,[i,i],color='navy')
        ax.axvline(0,color='gray',ls=':'); ax.set(yticks=range(5),yticklabels=methods,xlabel='Group-macro ΔpKa skill versus zero shift',title=split)
    fig.suptitle('Engineering smoke: interface agreement with current PypKa\nPairwise support; group-bootstrap 95% intervals where supported; selected sample')
    for ext in ('png','pdf'): fig.savefig(out/f'interface_skill.{ext}',dpi=180)
    plt.close(fig)
    with (campaign/'error_cancellation.csv').open() as f: rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,4,figsize=(14,4),layout='constrained')
    for ax,method in zip(axes,methods[:-1]):
        rr=[r for r in rows if r['method']==method]; x=[float(r['free_error']) for r in rr]; y=[float(r['ab_error']) for r in rr]
        ax.scatter(x,y,s=6,alpha=.25); ax.axhline(0,color='gray',lw=.5); ax.axvline(0,color='gray',lw=.5)
        if x:
            lo=min(x+y); hi=max(x+y); ax.plot([lo,hi],[lo,hi],color='gray',ls=':')
        ax.set(xlabel='Free-state error vs PypKa',ylabel='Bound-state error vs PypKa',title=method)
    fig.suptitle('Absolute-error cancellation: masked interface sites\nAll smoke splits shown descriptively; repeated sites are correlated')
    for ext in ('png','pdf'): fig.savefig(out/f'error_cancellation.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__': render(sys.argv[1])
