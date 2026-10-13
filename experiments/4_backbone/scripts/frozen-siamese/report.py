import json,os
from pathlib import Path
import numpy as np
base=Path(os.environ['PKABENCH_RUNTIME'])/'audits';out=base/'frozen-siamese-v1';rows=[]
for seed in (17,29,43):
 for kind in ('small','mlp','transformer'):
  original=base/'frozen-interface-v1'/f's{seed}'/'site' if kind=='small' else base/'frozen-transformer-v1'/f's{seed}'/kind
  support=None
  for objective,folder in [('free-only',original),('siamese',out/f's{seed}'/kind)]:
   s=json.loads((folder/'selection.json').read_text());detail=json.loads((folder/'per-complex.json').read_text())
   keys={r['id']:(r['n'],r['positives']) for r in detail}
   if support is None:support=keys
   assert keys==support,'validation support differs'
   rows.append({'seed':seed,'head':kind,'objective':objective,'epoch':s['epoch'],'site_macro_ap':s['site_macro_ap'],'pooled_ap':s['site']['average_precision'],'auroc':s['site']['auroc'],'bound_macro_ap':s.get('bound_validation',{}).get('site_macro_ap')})
means={f'{kind}/{obj}':{m:float(np.mean([r[m] for r in rows if r['head']==kind and r['objective']==obj])) for m in ('site_macro_ap','pooled_ap','auroc')} for kind in ('small','mlp','transformer') for obj in ('free-only','siamese')}
(out/'comparison.json').write_text(json.dumps({'runs':rows,'means':means,'identical_validation_support':True,'test_data_included':False},indent=2)+'\n')
lines=['# Frozen Siamese interface heads','','Three frozen 50% relational GQT checkpoints. All primary results use free tokens only. Equal weights for free BCE, bound BCE and detached bound-to-free probability consistency.','','| Head | Objective | Seed | Epoch | Free macro AP | Free pooled AP | Free AUROC | Bound macro AP |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append('| '+' | '.join(str(r[k]) if r[k] is not None else '—' for k in ('head','objective','seed','epoch','site_macro_ap','pooled_ap','auroc','bound_macro_ap'))+' |')
lines+=['','Means: '+json.dumps(means),'','Selection uses free-only validation macro AP. GQT is frozen and its pKa predictions unchanged. This evaluates the combined bound-supervision plus consistency objective; separating those contributions needs a bound-supervision-only control. Labels cover contacts between eligible pKa-labelled surface sites, not all residues. Free means partner-removed bound conformation. Scores are not calibrated probabilities. No independent test data used.']
(out/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
