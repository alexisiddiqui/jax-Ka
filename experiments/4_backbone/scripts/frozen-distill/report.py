import os,json
from pathlib import Path
import numpy as np
base=Path(os.environ['PKABENCH_RUNTIME'])/'audits';out=base/'frozen-distill-v1';rows=[]
for seed in (17,29,43):
 support=None
 for name,folder in [('labels-only',base/'frozen-interface-v1'/f's{seed}'/'site'),('teacher',out/f's{seed}'/'teacher'),('lambda-0.1',out/f's{seed}'/'student-lambda-0.1'),('lambda-1',out/f's{seed}'/'student-lambda-1')]:
  s=json.loads((folder/'selection.json').read_text());detail=json.loads((folder/'per-complex.json').read_text());keys={r['id']:(r['n'],r['positives']) for r in detail}
  if support is None:support=keys
  assert keys==support
  rows.append({'seed':seed,'role':name,'input':'bound' if name=='teacher' else 'free','epoch':s['epoch'],'macro_ap':s['site_macro_ap'],'pooled_ap':s['site']['average_precision'],'auroc':s['site']['auroc']})
 verification=json.loads((out/f's{seed}'/'verification.json').read_text());assert verification['teacher_parameters_unchanged_during_student_training']
means={k:{m:float(np.mean([r[m] for r in rows if r['role']==k])) for m in ('macro_ap','pooled_ap','auroc')} for k in ('labels-only','teacher','lambda-0.1','lambda-1')}
(out/'comparison.json').write_text(json.dumps({'runs':rows,'means':means,'identical_validation_support':True,'teacher_frozen_verified':True,'test_data_included':False},indent=2)+'\n')
lines=['# Frozen teacher interface distillation','','Small MLPs on three frozen 50% relational GQT checkpoints. Bound teacher is independently trained, selected and frozen before free-only students train.','','| Role | Input | Seed | Epoch | Macro AP | Pooled AP | AUROC |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append('| '+' | '.join(str(r[k]) for k in ('role','input','seed','epoch','macro_ap','pooled_ap','auroc'))+' |')
lines+=['','Means: '+json.dumps(means),'','Teacher rows use bound input and are diagnostic, not directly comparable inference performance. Students use free tokens only. Teachers selected by bound validation AP; students by free validation AP. Teacher probabilities are training-only targets. Labels cover contacts between eligible pKa-labelled surface sites; free structures are partner-removed bound conformations. No GQT updates or independent test evaluation.']
(out/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
