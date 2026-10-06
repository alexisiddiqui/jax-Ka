"""Compare frozen hybrid models and their PypKa teacher on common experimental sites."""
import csv, json, os
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from pkabench.runtime import require_compute, atomic_json, digest
require_compute()
root=Path(os.environ['PKABENCH_RUNTIME'])/'experimental/hybrid-v1'
pilot=root.parent/'pilot-v2'
labels=json.loads((pilot/'audited_observations.json').read_text())
models=json.loads((root/'model_receipt.json').read_text())
assert not models['fit_performed'] and all(digest(p)==h for p,h in models['models_sha256'].items())
assert models['feature_sha256']==digest(root/'features.parquet')
assert json.loads((root/'feature_regression.json').read_text())['passed']
key=lambda r:(r['complex_id'],r['chain'],r['resnum'],r['icode'],r['group'])
pred={}; coverage=[]; hashes={}
for pdb in ('1BNI','1IGD','1PGB'):
    replacement_path=root/pdb/'replacements.json'
    replacements=json.loads(replacement_path.read_text())
    replaced={key(r['site']) for r in replacements}
    for method in ('teacher','catboost-17','catboost-29','catboost-43'):
        path=root/pdb/method/'result.json'; result=json.loads(path.read_text()); hashes[str(path)]=digest(path)
        assert result['energy_sha256']==digest(path.parent/'energies.json')
        assert result['source_sha256']==digest(root/pdb/'pypka/AB/mc-energies.json')
        assert result['replacement_sha256']==digest(replacement_path)
        assert result['interactions_unchanged']
        if method=='teacher': assert result['exact_replay'] and result['max_pka_error']==0 and result['max_curve_error']==0
        pred[pdb,method]={key(r):r for r in result['rows']}
        if method=='catboost-17': coverage.append(dict(pdb=pdb,total_native_sites=result['total_sites'],replaced_sites=result['replaced_sites'],retained_teacher_sites=result['retained_teacher_sites']))
    for label in labels:
        if label['scoring_pdb_id']!=pdb: continue
        k=(pdb,label['chain'],label['resnum'],'',label['group'])
        label['hybrid_intrinsic_replaced']=k in replaced
        for method in ('teacher','catboost-17','catboost-29','catboost-43'):
            label[method+'_pka']=pred[pdb,method].get(k,{}).get('pka')
methods=('null','propka','pkai','jaxka','teacher','catboost-17','catboost-29','catboost-43')
candidate=[r for r in labels if r['label_kind']=='point' and r['structural_eval_mask']]
common=[r for r in candidate if r['hybrid_intrinsic_replaced'] and all(r[m+'_pka'] is not None and np.isfinite(r[m+'_pka']) for m in methods)]
assert common
def score(rr,method):
    y=np.array([r['value'] for r in rr]); p=np.array([r[method+'_pka'] for r in rr]); null=np.array([r['null_pka'] for r in rr])
    mse=float(np.mean((p-y)**2)); nmse=float(np.mean((null-y)**2)); ds=y-null; ps=p-null; sign=np.abs(ds)>=.5
    return dict(n=len(rr),mae=float(np.abs(p-y).mean()),rmse=float(mse**.5),skill=1-mse/nmse,
        spearman=float(spearmanr(y,p).statistic),sign_n=int(sign.sum()),
        sign_accuracy=float(np.mean(np.sign(ds[sign])==np.sign(ps[sign]))) if sign.any() else None)
metrics=[dict(method=m,**score(common,m)) for m in methods]
families={f:[dict(method=m,**score([r for r in common if r['family']==f],m)) for m in methods] for f in sorted({r['family'] for r in common})}
available={m:sum(r[m+'_pka'] is not None for r in candidate) for m in methods}
counts=dict(candidate_records=len(labels),structurally_eligible_points=len(candidate),common_points=len(common),
    available_point_predictions=available,eligible_points_with_replaced_intrinsics=sum(r['hybrid_intrinsic_replaced'] for r in candidate))
atomic_json(root/'comparison.json',dict(counts=counts,metrics=metrics,by_family=families,coverage=coverage,
    common_ids=[r['observation_id'] for r in common],frozen_models=True,primary_verification_pending=True,
    experimental_training_performed=False,teacher_oracle_exact=True))
atomic_json(root/'observations.json',labels)
fields=['observation_id','protein','family','scoring_pdb_id','chain','resnum','group','raw_label','label_kind','structural_eval_mask','hybrid_intrinsic_replaced']+[m+'_pka' for m in methods]
with (root/'predictions.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(labels)
names={'null':'Constant null','propka':'PROPKA','pkai':'Frozen pKAI','jaxka':'Frozen JAX-Ka','teacher':'PypKa','catboost-17':'CatBoost–PypKa seed 17','catboost-29':'CatBoost–PypKa seed 29','catboost-43':'CatBoost–PypKa seed 43'}
lines=['# Frozen CatBoost hybrids on the experimental pilot','',
    '**Provisional comparison against secondary PKAD-R labels. Original measurement-table, construct and condition verification remains unresolved. No experimental training was performed.**','',
    f"The comparison uses {len(common)} identical point observations across all eight rows, from {len(families)} protein families. Of {len(candidate)} structurally eligible point records, {counts['eligible_points_with_replaced_intrinsics']} receive predicted hybrid intrinsics. Censored and approximate labels do not enter point-error metrics.",'',
    '| Method | Sites | MAE | RMSE | Skill vs null | Spearman | Shift sign accuracy |','|---|---:|---:|---:|---:|---:|---:|']
for r in metrics: lines.append(f"| {names[r['method']]} | {r['n']} | {r['mae']:.3f} | {r['rmse']:.3f} | {r['skill']:.3f} | {r['spearman']:.3f} | {100*r['sign_accuracy']:.0f}% |")
lines+=['','MAE/RMSE are in pKa units; skill is 1 − MSE/MSE_null. Spearman uses absolute pKas. Sign accuracy uses experimental minus fixed-null shifts of magnitude ≥0.5; these are not binding shifts. The null predicts zero shift, counted as incorrect on these nonzero shifts.','',
    '## Per-family MAE','', '| Method | Barnase | Protein G |','|---|---:|---:|']
for m in methods:
    v={f:next(r['mae'] for r in rr if r['method']==m) for f,rr in families.items()}
    lines.append(f"| {names[m]} | {v['barnase']:.3f} | {v['protein_g']:.3f} |")
lines+=['','Only two families are represented. Seed variation is not a confidence interval, and this pilot cannot establish a general ranking.','',
    '## What the hybrid contains','',
    'The saved geometry-to-intrinsic CatBoost models (seeds 17/29/43) were loaded without fitting. Their predictions replace all eligible native-tautomer intrinsic terms, including sites without experimental labels. PypKa supplies the original interaction matrix and Monte Carlo readout. Masked sites retain teacher intrinsics, so this is a teacher-assisted hybrid, not a standalone CatBoost predictor.','',
    '| Structure | Native sites | CatBoost-replaced sites | Teacher-retained sites |','|---|---:|---:|---:|']
for r in coverage: lines.append(f"| {r['pdb']} | {r['total_native_sites']} | {r['replaced_sites']} | {r['retained_teacher_sites']} |")
lines+=['','The artificial N-terminus of truncated barnase is explicitly excluded from replacement. The existing missing-region/component policy determines other replacements. The selected structures contain no stripped components.','',
    '## Verification','',
    'A saved training-campaign state verifies every inference feature against the original feature table at absolute tolerance 1e-12. Model hashes match the frozen fit receipts. Native energy conversion is checked against the original tautomer intrinsics. Interactions, reference states, padding and masked intrinsic terms are asserted unchanged.','',
    'PypKa intrinsic terms plus PypKa interactions reproduce the original PypKa curves and pKas exactly for all three structures. This oracle tests the replay/conversion; it does not imply zero error against experiment. Each hybrid runs only after that structure’s oracle passes.','',
    'Teacher parameters match the existing benchmark: PypKa 2.10.0, G54A7, εin 15, εsol 80, 0.1 M, 298.15 K, grid 81, nonperiodic boundaries, hydrogen optimization enabled, Ser/Thr titration disabled; pH −2…16 in 0.25 steps. Monte Carlo uses 200,000 steps, 1,000 equilibration steps and seed 1234567. These are standardized baseline settings, not a claim to reproduce every experimental condition.','',
    '## Coverage and scope','', '| Method | Available predictions among eligible point labels |','|---|---:|']
for m in methods: lines.append(f"| {names[m]} | {available[m]}/{len(candidate)} |")
lines+=['','The original PROPKA Glu60 suppression and pKAI terminal limitation remain recorded; no values were imputed. Original baseline outputs and masks were preserved. Family-overlap checks and source holds are inherited from [pilot v2](../pilot-v2/report.md). This is absolute-pKa evaluation; the PROPKA-plus-CatBoost binding-shift correction is not applicable to these labels.','',
    'Machine-readable outputs: comparison.json, observations.json and predictions.csv. Individual teacher/hybrid energies, requests, receipts and model/feature hashes are retained in this directory.']
(root/'report.md').write_text('\n'.join(lines)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(9,4))
ax.bar(range(len(metrics)),[r['mae'] for r in metrics],color=['gray','#709bc3','#709bc3','#709bc3','#c98a38','#57a67a','#57a67a','#57a67a'])
ax.set_xticks(range(len(metrics)),[names[m].replace('CatBoost–PypKa seed ','Hybrid\n') for m in methods],rotation=20,ha='right')
ax.set_ylabel('Absolute-pKa MAE'); ax.set_title(f'Provisional experimental pilot: {len(common)} common points, two families')
fig.tight_layout();fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)
atomic_json(root/'release_manifest.json',dict(results_sha256=hashes,model_receipt_sha256=digest(root/'model_receipt.json'),
    labels_sha256=digest(pilot/'audited_observations.json'),code_sha256=digest(Path(__file__)),
    outputs_sha256={p.name:digest(p) for p in root.iterdir() if p.is_file() and p.name!='release_manifest.json'},job=os.environ['SLURM_JOB_ID']))
print(json.dumps(dict(counts=counts,metrics=metrics,coverage=coverage),indent=2),flush=True)
