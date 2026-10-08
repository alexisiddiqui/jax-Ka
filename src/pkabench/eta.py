"""Compute-node runtime diagnostic using observed residue counts and timings."""
import json
import os
from pathlib import Path
import subprocess
from collections import defaultdict
import numpy as np
from .runtime import require_compute, atomic_json
from .schema import read_table


def estimate(out):
    require_compute(); out=Path(out)
    manifest=json.loads((out/'manifest.json').read_text()); cases={c['complex_id']:c for c in manifest['cases']}
    live={}
    for line in subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%M|%N'],text=True).splitlines():
        job,elapsed,node=line.strip().split('|'); days=0
        if '-' in elapsed: d,elapsed=elapsed.split('-'); days=int(d)
        seconds=0
        for number in elapsed.split(':'): seconds=60*seconds+int(number)
        live[job]={'elapsed_min':(days*86400+seconds)/60,'node':node}
    done=[]; pending=[]
    for path,cid in manifest['tasks']:
        root=Path(path); receipt=root/'jobs/pypka'/f'{cid}.json'
        n=read_table(root/'structures.parquet')[0]['n_residues']
        row={'cid':cid,'pdb':cases[cid]['pdb_id'],'variant':root.name,'n_residues':n}
        if receipt.exists():
            side=json.loads(receipt.read_text())
            if side['status']=='complete': done.append({**row,'minutes':side['wall_seconds']/60,'node':side['node']})
        else:
            task=json.loads((root/'submission.json').read_text())[f'{cid}/pypka']; job=task['job'].split(';')[0]
            pending.append({**row,'job':job,**live.get(job,{})})
    groups=defaultdict(list)
    for row in done: groups[row['cid']].append(row)
    # Each reference contributes one median, rather than treating deletions as independent structures.
    reference=[(np.median([r['n_residues'] for r in rr]),np.median([r['minutes'] for r in rr])) for rr in groups.values()]
    x=np.log([r[0] for r in reference]); y=np.log([r[1] for r in reference])
    slope,intercept=np.polyfit(x,y,1); residual=y-(intercept+slope*x)
    for row in pending:
        prediction=float(np.exp(intercept+slope*np.log(row['n_residues'])))
        comparable=groups[row['cid']]; times=[r['minutes'] for r in comparable]
        row.update(residue_model_total_min=prediction,
            residue_model_reference_residual_band_min=[float(prediction*np.exp(q)) for q in np.quantile(residual,[.1,.9])],
            same_reference_completed=len(times),same_reference_median_min=float(np.median(times)) if times else None,
            same_reference_max_min=max(times,default=None),
            same_reference_runs=[{k:r[k] for k in ('variant','n_residues','minutes','node')} for r in comparable])
    result={'completed':len(done),'unfinished':pending,'power_law_exponent':float(slope),
        'log_fit_r_squared':float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)),
        'reference_medians':[{'pdb':cases[cid]['pdb_id'],'n_residues':float(np.median([r['n_residues'] for r in rr])),
            'minutes':float(np.median([r['minutes'] for r in rr]))} for cid,rr in groups.items()],
        'note':'Residue-only log-linear fit to completed reference medians; unfinished jobs are right-censored and may exceed this model. Residual band is descriptive, not a calibrated ETA interval.'}
    atomic_json(out/'runtime-estimate.json',result); print(json.dumps(result,indent=2))
