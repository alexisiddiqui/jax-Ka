import json,os
from pathlib import Path
import numpy as np
root=Path(os.environ['PKABENCH_RUNTIME'])/'audits/frozen-interface-v1'
rows=[]
for seed in (17,29,43):
 for head in ('site','pair'):
  p=root/f's{seed}'/head
  s=json.loads((p/'selection.json').read_text());protocol=json.loads((p/'protocol.json').read_text())
  rows.append({'seed':seed,'head':head,'epoch':s['epoch'],'site_macro_ap':s['site_macro_ap'],'site_pooled_ap':s['site']['average_precision'],'site_auroc':s['site']['auroc'],'pair_ap':s.get('pair',{}).get('average_precision'),'pair_macro_ap':s.get('pair_macro_ap'),'complexes':s['complexes'],'sites':s['site']['n'],'positives':s['site']['positives']})
means={h:{k:float(np.mean([r[k] for r in rows if r['head']==h])) for k in ('site_macro_ap','site_pooled_ap','site_auroc')} for h in ('site','pair')}
(root/'comparison.json').write_text(json.dumps({'runs':rows,'means':means,'test_data_included':False},indent=2)+'\n')
lines=['# Frozen relational GQT interface probes','','50% checkpoints, seeds 17/29/43. Free tokens only; GQT unchanged. Targets are contacts between existing pKa-labelled surface sites, not all protein residues.','','| Head | Seed | Epoch | Site macro AP | Site pooled AP | Site AUROC | Pair AP |','|---|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append('| '+ ' | '.join(str(r[k]) if r[k] is not None else '—' for k in ('head','seed','epoch','site_macro_ap','site_pooled_ap','site_auroc','pair_ap'))+' |')
lines+=['','Selection: best validation site macro AP. Macro AP excludes complexes with no positive site; pooled metrics retain them. No test set was used.','', 'Pair head uses maximum pair probability for site ranking. Balanced training and maximum aggregation mean scores should not be interpreted as calibrated probabilities. Pair and site heads differ in supervision and parameter count; this measures practical probe performance, not a parameter-matched attention ablation.','', 'Free means partner-removed bound conformation; generalization to independently measured unbound structures remains untested.']
(root/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
