import os,json,csv
from pathlib import Path
import numpy as np
from pkatrain.production_shift_weight import validation_bins
root=Path(os.environ['PKABENCH_RUNTIME']);runs=root/'training/gqt-production-v2/runs';out=root/'audits/global-query-v1-50';out.mkdir(parents=True,exist_ok=True)
results=[];reference=None
for family,prefix in [('baseline','control-v2-50'),('global-query','global-query-v1-50')]:
 for seed in (17,29,43):
  run=f'{prefix}-s{seed}';folder=runs/run;s=json.loads((folder/'selection.json').read_text());v=s['validation'];epoch=s['selected_epoch']
  rr=list(csv.DictReader((folder/f'predictions-pinder-epoch-{epoch:03d}.csv').open()))
  for r in rr:
   for k in ('teacher_ab','teacher_free','predicted_ab','predicted_free'):r[k]=float(r[k])
   r['interface']=r['interface']=='True'
  keys={(r['complex_id'],r['site']):(r['teacher_ab'],r['teacher_free'],r['interface']) for r in rr};assert len(keys)==len(rr)
  if reference is None:reference=keys
  assert keys==reference,'validation support changed'
  results.append({'family':family,'seed':seed,'run':run,'epoch':epoch,'parameters':s['parameter_count'],'selection':v['selection'],'state_mae':v['pinder']['state_mae'],'interface_mae':v['pinder']['interface_paired_mae'],'pkpdb_mae':v['pkpdb']['site_mae'],'benchmark_mae':v['benchmark']['overall']['mae'],'bins':validation_bins(rr)})
assert len({r['parameters'] for r in results})==1,'parameter count changed'
metrics=['selection','state_mae','interface_mae','pkpdb_mae','benchmark_mae'];means={f:{k:float(np.mean([r[k] for r in results if r['family']==f])) for k in metrics} for f in ('baseline','global-query')}
differences={k:[next(r[k] for r in results if r['family']=='global-query' and r['seed']==s)-next(r[k] for r in results if r['family']=='baseline' and r['seed']==s) for s in (17,29,43)] for k in metrics}
(out/'results.json').write_text(json.dumps({'results':results,'means':means,'seed_paired_differences':differences,'test_data_included':False},indent=2)+'\n')
lines=['# Global site-query attention: 50%, three seeds','','Unweighted losses; identical parameter count and validation support. Original selection criterion retained.','','| Family | Seed | Selection | State | Interface Δ | pKPDB | Benchmark |','|---|---:|---:|---:|---:|---:|---:|']
for r in results:lines.append(f"| {r['family']} | {r['seed']} | "+' | '.join(f'{r[k]:.4f}' for k in metrics)+' |')
lines+=['','Mean global-query minus baseline: '+', '.join(f'{k} {means["global-query"][k]-means["baseline"][k]:+.4f}' for k in metrics)+'.']
(out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
