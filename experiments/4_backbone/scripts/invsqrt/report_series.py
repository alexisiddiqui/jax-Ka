import os,json,csv
from pathlib import Path
import numpy as np
from pkatrain.production_shift_weight import validation_bins
root=Path(os.environ['PKABENCH_RUNTIME']);code=Path(os.environ['CODE']);plan=json.loads((code/'plan.json').read_text());out=root/'audits/invsqrt-experiments-20261013';out.mkdir(parents=True,exist_ok=True)
results=[]
reference_sites=None
for r in plan['runs']+plan['existing_controls']:
 folder=root/'training/gqt-production-v2/runs'/r['run'];s=json.loads((folder/'selection.json').read_text());v=s['validation'];epoch=s['selected_epoch']
 rr=list(csv.DictReader((folder/f'predictions-pinder-epoch-{epoch:03d}.csv').open()))
 for row in rr:
  for k in ('teacher_ab','teacher_free','predicted_ab','predicted_free'):row[k]=float(row[k])
  row['interface']=row['interface']=='True'
 indexed={(r['complex_id'],r['site']):(r['teacher_ab'],r['teacher_free'],r['interface']) for r in rr}
 assert len(indexed)==len(rr), 'duplicate validation sites'
 if reference_sites is None: reference_sites=indexed
 assert indexed==reference_sites, 'runs do not share identical validation sites and targets'
 bins=validation_bins(rr)
 results.append({**r,'epoch':epoch,'selection':v['selection'],'state_mae':v['pinder']['state_mae'],'interface_mae':v['pinder']['interface_paired_mae'],'pkpdb_mae':v['pkpdb']['site_mae'],'benchmark_mae':v['benchmark']['overall']['mae'],'bins':bins})
(out/'results.json').write_text(json.dumps({'results':results,'test_data_included':False},indent=2)+'\n')
lines=['# Inverse-square-root reference and ablation results','','Unweighted validation; selected by the unchanged overall criterion. Architectural comparisons are exploratory single-seed results.','','| Run | Epoch | Selection | State | Interface Δ | pKPDB | Benchmark |','|---|---:|---:|---:|---:|---:|---:|']
for r in results:lines.append('| '+r['run']+' | '+str(r['epoch'])+' | '+' | '.join(f"{r[k]:.4f}" for k in ('selection','state_mae','interface_mae','pkpdb_mae','benchmark_mae'))+' |')
lines+=['','## Matched weighted minus unweighted reference differences','']
for fraction in (.5,1.):
 pairs=[]
 for seed in (17,29,43):
  a=next(r for r in results if r['fraction']==fraction and r['seed']==seed and r['weight']=='invsqrt' and r['variant']=='none')
  b=next(r for r in results if r['fraction']==fraction and r['seed']==seed and r['weight']=='none')
  pairs.append(a['selection']-b['selection'])
 lines.append(f'{fraction:.0%}: seed differences {pairs}; mean {np.mean(pairs):.4f}.')
(out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
