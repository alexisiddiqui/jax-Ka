import json,os
from pathlib import Path
import numpy as np
base=Path(os.environ['PKABENCH_RUNTIME'])/'audits';out=base/'frozen-transformer-v1';rows=[]
for seed in (17,29,43):
 reference=None
 for kind in ('original-mlp','transformer','mlp'):
  folder=base/'frozen-interface-v1'/f's{seed}'/'site' if kind=='original-mlp' else out/f's{seed}'/kind
  s=json.loads((folder/'selection.json').read_text());d=json.loads((folder/'per-complex.json').read_text())
  support={r['id']:(r['n'],r['positives']) for r in d}
  if reference is None:reference=support
  assert support==reference,'validation support differs'
  rows.append({'seed':seed,'head':kind,'epoch':s['epoch'],'site_macro_ap':s['site_macro_ap'],'pooled_ap':s['site']['average_precision'],'auroc':s['site']['auroc']})
means={k:{m:float(np.mean([r[m] for r in rows if r['head']==k])) for m in ('site_macro_ap','pooled_ap','auroc')} for k in ('original-mlp','transformer','mlp')}
(out/'comparison.json').write_text(json.dumps({'runs':rows,'means':means,'identical_validation_support':True,'test_data_included':False},indent=2)+'\n')
lines=['# Frozen free site transformer comparison','','Frozen 50% relational GQT, three seeds. Transformer 33,665 parameters; matched MLP 33,661. Both new heads use dropout 0.1. Original MLP uses no dropout.','','| Head | Seed | Selected epoch | Macro site AP | Pooled AP | AUROC |','|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append('| '+ ' | '.join(str(r[k]) for k in ('head','seed','epoch','site_macro_ap','pooled_ap','auroc'))+' |')
lines+=['','Means: '+json.dumps(means),'','Labels, free RSA >=0.1 surface gate, and validation support unchanged. Transformer attends within each chain, among cached eligible pKa sites only. No bound tokens, partner coordinates, or new geometric/positional input. GQT is frozen. Macro AP excludes no-positive complexes; pooled metrics include them. Selected validation results are not an independent test estimate.']
(out/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
