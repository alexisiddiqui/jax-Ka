"""Score frozen methods on the expanded PKAD-R audit without fitting models."""
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from pkabench.runtime import require_compute, atomic_json, digest

require_compute()
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from pkabench.schema import NULL_PKA

runtime = Path(os.environ['PKABENCH_RUNTIME'])
source = runtime/'experimental/pkadr-full-v1'
baseline = runtime/'experimental/pkadr-baselines-v1'
jax_recovery = runtime/'experimental/pkadr-jaxka-1024-v1'
use_jax_recovery = os.environ.get('PKADR_JAX_RECOVERY') == '1'
root = runtime/'experimental'/('pkadr-baselines-v2' if use_jax_recovery else 'pkadr-baselines-v1')
root.mkdir(parents=True,exist_ok=True)
methods = ('propka','pkai','pkai_plus','jaxka','pypka')
names = {'propka':'PROPKA','pkai':'pKAI','pkai_plus':'pKAI+',
         'jaxka':'JAX-Ka','pypka':'PypKa','null':'Fixed null'}
joined = json.loads((source/'joined-records.json').read_text())
tasks = json.loads((baseline/'tasks.json').read_text())
run_manifest=json.loads((baseline/'manifest.json').read_text())
assert run_manifest['source_release_sha256'] == digest(source/'release_manifest.json')
current_runner_sha=digest(Path(os.environ['PKABENCH_SOURCE'])/'experiments/experimental_pilot/full_baselines.py')
if use_jax_recovery:
    recovery_manifest=json.loads((jax_recovery/'manifest.json').read_text())
    assert recovery_manifest['steps']==1024 and recovery_manifest['tasks']==len(tasks)

def identity(r):
    return (r['complex_id'],r['chain'],int(r['resnum']),r['icode'],r['group'])

prediction_status = {m:Counter() for m in methods}
predictions = {}
v1_jax_predictions = {}
for task in tasks:
    task_root = baseline/'tasks'/task['task_id']
    receipt = json.loads((task_root/'receipt.json').read_text())
    assert receipt['task_id'] == task['task_id'] and tuple(receipt['methods']) == methods
    for method in methods:
        method_root=(jax_recovery/'tasks'/task['task_id']/method) if use_jax_recovery and method=='jaxka' else task_root/method
        path = method_root/'predictions.parquet'
        if use_jax_recovery and method=='jaxka':
            method_receipt=json.loads((method_root/'receipt.json').read_text())
            assert method_receipt['steps']==1024 and method_receipt['prediction_sha256']==digest(path)
        else:
            assert receipt['outputs'][method] == digest(path)
        rows = pq.read_table(path).to_pylist()
        assert len({identity(r) for r in rows}) == len(rows)
        for row in rows:
            predictions[method,identity(row)] = row
            prediction_status[method][row['status']] += 1
        if use_jax_recovery and method == 'jaxka':
            for row in pq.read_table(task_root/'jaxka'/'predictions.parquet').to_pylist():
                v1_jax_predictions[identity(row)] = row

rows=[]
for source_row in joined:
    if not source_row['structural_eval_mask']:
        continue
    row={k:source_row.get(k) for k in ('record_id','task_id','pdb','author_chain','chain','resnum','icode',
         'group','label_kind','value','raw_label','family_id','local_independent_candidate','natural_gap_tier')}
    row['raw_label']=row['raw_label'] or source_row['raw'].get('Expt_pKa')
    row['protein_name']=source_row['raw'].get('Protein_Name')
    row['classification']=source_row['raw'].get('pKa_Classification')
    row['experimental_method']=source_row['raw'].get('Expt_Method')
    row['mutation']=source_row['raw'].get('Mut_Pos')
    row['reference']=source_row['raw'].get('Reference')
    row['null_pka']=NULL_PKA[source_row['group']]
    key=(source_row['task_id'],source_row['chain'],int(source_row['resnum']),source_row['icode'],source_row['group'])
    for method in methods:
        pred=predictions.get((method,key),{})
        row[method+'_status']=pred.get('status','not_reported')
        row[method+'_pka']=pred.get('pka') if pred.get('status')=='ok' else None
    rows.append(row)

point=[r for r in rows if r['label_kind']=='point']
local=[r for r in point if r['local_independent_candidate']]
common=[r for r in local if all(r[m+'_pka'] is not None for m in methods)]
broad_methods=('propka','pkai','pkai_plus','pypka')
broad_common=[r for r in local if all(r[m+'_pka'] is not None for m in broad_methods)]
assert common and all(r['value'] is not None for r in common)

jax_recovery_summary=None
if use_jax_recovery:
    keys=[(r['task_id'],r['chain'],int(r['resnum']),r['icode'],r['group']) for r in local]
    old_ok=[v1_jax_predictions.get(k,{}).get('status')=='ok' for k in keys]
    new_ok=[predictions.get(('jaxka',k),{}).get('status')=='ok' for k in keys]
    jax_recovery_summary=dict(
        eligible_points=len(keys),v1_reported=sum(old_ok),v2_reported=sum(new_ok),
        newly_reported=sum(new and not old for old,new in zip(old_ok,new_ok)),
        no_longer_reported=sum(old and not new for old,new in zip(old_ok,new_ok)),
        v1_coverage=sum(old_ok)/len(keys),v2_coverage=sum(new_ok)/len(keys))

def corr(a,b):
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0: return None
    x=spearmanr(a,b).statistic
    return float(x) if np.isfinite(x) else None

def site_metrics(data,method):
    truth=np.array([r['value'] for r in data],float)
    null=np.array([r['null_pka'] for r in data],float)
    pred=null if method=='null' else np.array([r[method+'_pka'] for r in data],float)
    err=np.abs(pred-truth); ds=truth-null; ps=pred-null
    shifted=np.abs(ds)>=.5
    return dict(n=len(data),families=len({r['family_id'] for r in data}),
        mae=float(err.mean()),rmse=float(np.sqrt(np.mean((pred-truth)**2))),
        skill=float(1-np.mean((pred-truth)**2)/np.mean((null-truth)**2)),
        spearman_shift=corr(ds,ps),shifted_n=int(shifted.sum()),
        shifted_mae=float(err[shifted].mean()) if shifted.any() else None,
        shifted_sign_accuracy=float(np.mean(np.sign(ds[shifted])==np.sign(ps[shifted]))) if shifted.any() else None)

families=sorted({r['family_id'] for r in common})
by_family={f:{m:site_metrics([r for r in common if r['family_id']==f],m)
              for m in (*methods,'null')} for f in families}

def macro(method, subset=lambda r:True):
    vals=[]
    for family in families:
        rr=[r for r in common if r['family_id']==family and subset(r)]
        if rr: vals.append(site_metrics(rr,method)['mae'])
    return float(np.mean(vals)) if vals else None, len(vals)

rng=np.random.default_rng(20261005)
bootstrap={}
for method in methods:
    diffs=np.array([by_family[f][method]['mae']-by_family[f]['null']['mae'] for f in families])
    boot=np.mean(rng.choice(diffs,(20000,len(diffs)),replace=True),axis=1)
    bootstrap[method]=dict(point=float(diffs.mean()),ci95=[float(x) for x in np.quantile(boot,[.025,.975])],
                           probability_better_than_null=float(np.mean(boot<0)))

metrics=[]
for method in (*methods,'null'):
    item=dict(method=method,display_name=names[method],**site_metrics(common,method))
    item['family_macro_mae'],item['family_macro_n']=macro(method)
    item['family_macro_shifted_mae'],item['family_macro_shifted_n']=macro(method,lambda r:abs(r['value']-r['null_pka'])>=.5)
    metrics.append(item)

def comparison(data, compared_methods):
    ff=sorted({r['family_id'] for r in data})
    output=[]
    for method in (*compared_methods,'null'):
        family_mae=[site_metrics([r for r in data if r['family_id']==f],method)['mae'] for f in ff]
        item=dict(method=method,**site_metrics(data,method))
        item['family_macro_mae']=float(np.mean(family_mae))
        output.append(item)
    return output

broad_metrics=comparison(broad_common,broad_methods)

coverage=[]
for method in methods:
    ok=[r for r in local if r[method+'_pka'] is not None]
    coverage.append(dict(method=method,eligible_points=len(local),reported=len(ok),coverage=len(ok)/len(local),
                         families=len({r['family_id'] for r in ok}),statuses=dict(prediction_status[method])))

structural_zeros=[]
signal=[r for r in common if abs(r['value']-r['null_pka'])>=.1]
for method in methods:
    zero=sum(abs(r[method+'_pka']-r['null_pka'])<.01 for r in signal)
    structural_zeros.append(dict(method=method,n=len(signal),zeros=zero,fraction=zero/len(signal) if signal else None))

jax_failed=defaultdict(list)
for (method,key),pred in predictions.items():
    if method=='jaxka' and pred['status'] in ('failed','not_titrating'):
        jax_failed[pred['status']].append(dict(task_id=key[0],chain=key[1],resnum=key[2],icode=key[3],group=key[4]))

censored=[]
import re
for row in [r for r in rows if r['label_kind']=='censored']:
    nums=re.findall(r'-?\d+(?:\.\d+)?',row['raw_label'] or '')
    if not nums: continue
    bound=float(nums[0])
    for method in methods:
        pred=row[method+'_pka']
        if pred is None: continue
        violation=max(0.,bound-pred) if '>' in row['raw_label'] else max(0.,pred-bound)
        censored.append(dict(record_id=row['record_id'],family_id=row['family_id'],method=method,
                             raw_label=row['raw_label'],prediction=pred,one_sided_violation=violation))

atomic_json(root/'metrics.json',dict(
    scope='Provisional secondary-label benchmark; frozen methods; no model fit',
    main_population='Structurally retained numeric points without local frozen train/validation family overlap; common-method intersection',
    metrics=metrics,broad_four_method_metrics=broad_metrics,family_bootstrap_vs_null=bootstrap,coverage=coverage,structural_zeros=structural_zeros,
    jaxka_numerical_recovery=jax_recovery_summary,
    jaxka_unreported_sites=dict(jax_failed),
    by_family=by_family,censored_diagnostics=censored,common_record_ids=[r['record_id'] for r in common],
    caveats=['Original experimental tables, constructs, states and conditions are not yet verified.',
             'pKAI+ regularization used experimental-set performance and is not an independent frozen baseline.',
             'Local sequence checks do not establish absence from upstream model-development data.',
             'PKAD-R contains mutants and repeated/heterogeneous experimental contexts; these remain separate records.']))
with (root/'scored_records.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

count_by_family=Counter(r['family_id'] for r in common)
method_by_id={m['method']:m for m in metrics}
lines=[f"# Expanded PKAD-R frozen-baseline audit — {'v2' if use_jax_recovery else 'v1'}",'',
 '**This is a provisional secondary-label benchmark. It does not yet certify experimental conditions, constructs, or independence from upstream model development, and no model was trained here.**','',
 '## Evaluation population','', '| Stage | Records | Families |','|---|---:|---:|',
 f"| Structurally retained records | {len(rows)} | {len({r['family_id'] for r in rows if r['family_id']})} |",
 f"| Numeric point labels | {len(point)} | {len({r['family_id'] for r in point})} |",
 f"| Locally train/validation-disjoint points | {len(local)} | {len({r['family_id'] for r in local})} |",
 f"| Common points reported by every method | {len(common)} | {len(families)} |",'',
 '## Main comparison: each sequence family has equal weight','',
 '| Method | Families | Family-macro MAE | Shifted-family MAE | Family-bootstrap difference vs null (95% CI) | P(better than null) |',
 '|---|---:|---:|---:|---:|---:|']
for method in (*methods,'null'):
    m=method_by_id[method]; shifted='—' if m['family_macro_shifted_mae'] is None else f"{m['family_macro_shifted_mae']:.3f}"
    if method=='null': diff='reference'; probability='—'
    else:
        b=bootstrap[method]; diff=f"{b['point']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]"; probability=f"{100*b['probability_better_than_null']:.1f}%"
    lines.append(f"| {names[method]} | {m['family_macro_n']} | {m['family_macro_mae']:.3f} | {shifted} | {diff} | {probability} |")
lines += ['', '“Shifted” means |experimental pKa − residue-type null| ≥ 0.5. Bootstrap units are whole sequence families (20,000 resamples); sites within a family are never resampled independently. A confidence interval spanning zero does not resolve the method against the null.','',
 '## Supporting pooled-site view','',
 '| Method | Sites | MAE | RMSE | Skill vs null | Shifted sites | Shifted MAE | Shift sign accuracy | Shift Spearman |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for method in (*methods,'null'):
    m=method_by_id[method]
    sp='—' if m['spearman_shift'] is None else f"{m['spearman_shift']:.3f}"
    sg='—' if m['shifted_sign_accuracy'] is None else f"{100*m['shifted_sign_accuracy']:.1f}%"
    lines.append(f"| {names[method]} | {m['n']} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['skill']:.3f} | {m['shifted_n']} | {m['shifted_mae']:.3f} | {sg} | {sp} |")
lines += ['', 'The pooled table is descriptive because large families contribute more rows. Shift Spearman is computed after subtracting the residue-type null; absolute-pKa Spearman is deliberately omitted because it mostly ranks residue types.','',
 '## Coverage sensitivity without JAX-Ka','',
 f"JAX-Ka reports {100*next(c['coverage'] for c in coverage if c['method']=='jaxka'):.1f}% of eligible points in this run. This table removes JAX-Ka from the required intersection and recomputes the four remaining frozen methods on the broader, still exactly matched set. It is a coverage sensitivity analysis, not a replacement ranking.",'',
 '| Method | Common sites | Families | Family-macro MAE | Pooled MAE |','|---|---:|---:|---:|---:|']
for m in broad_metrics:
    lines.append(f"| {names[m['method']]} | {m['n']} | {m['families']} | {m['family_macro_mae']:.3f} | {m['mae']:.3f} |")
lines += ['', '## Method coverage','', '| Method | Reported / eligible points | Coverage | Families represented |','|---|---:|---:|---:|']
for c in coverage: lines.append(f"| {names[c['method']]} | {c['reported']} / {c['eligible_points']} | {100*c['coverage']:.1f}% | {c['families']} |")
lines += ['', 'Missing predictions are not imputed. The main comparison uses the exact common-method intersection. pKAI+ is shown for diagnostics, but its regularization weight was chosen using experimental performance and it cannot support an independent ranking claim.','',
 '## Interpretation limits','',
 'The structural and local-overlap filters turn the archive into a much broader test bed than the two-family pilot, but they do not make every database row verified experimental truth. Mutants, conditions, duplicate measurements and experimental states remain explicit in scored_records.csv. Point estimates must be checked against primary measurement tables before the set is frozen for training or headline evaluation. The 23 structurally retained censored observations are evaluated separately as one-sided violations and do not enter point metrics; eight approximate or compound observations are preserved but not scored as points.','',
 'metrics.json contains every family result, the family-bootstrap distributions, coverage counts, structural-zero diagnostics and censored-label diagnostics.']
lines += ['', '## Execution amendment','',
 'Seven tasks contained repeated experimental measurements of the same physical site. The initial runner rejected their duplicate prediction keys before running any method. The amended runner deduplicated only the method input, predicted each physical site once, and retained every measurement as a separate scoring record. Previously completed tasks contained unique sites and required no recomputation. The initial and amended runner hashes are recorded in scoring_release.json.']
if use_jax_recovery:
    lines += ['', '## JAX-Ka numerical recovery','',
      'The v1 experimental adapter used the 64-step constructor default. This v2 report replaces only JAX-Ka with a new immutable 1,024-step run, matching the accepted frozen production configuration and retaining the same 2 × 10⁻⁵ residual and midpoint-validity rules. All other method predictions are byte-identical v1 inputs. The 64-step release remains preserved.','',
      '| Configuration | Reported / eligible points | Coverage |','|---|---:|---:|',
      f"| v1: 64 steps | {jax_recovery_summary['v1_reported']} / {jax_recovery_summary['eligible_points']} | {100*jax_recovery_summary['v1_coverage']:.1f}% |",
      f"| v2: 1,024 steps | {jax_recovery_summary['v2_reported']} / {jax_recovery_summary['eligible_points']} | {100*jax_recovery_summary['v2_coverage']:.1f}% |",'',
      f"The corrected configuration recovered {jax_recovery_summary['newly_reported']} eligible predictions and lost {jax_recovery_summary['no_longer_reported']}. Remaining unreported sites retain their explicit solver or chemistry status; none are imputed."]
(root/'report.md').write_text('\n'.join(lines)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
order=list(methods)+['null']
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
axes[0].bar([names[m] for m in order],[method_by_id[m]['family_macro_mae'] for m in order],color='#3976a8')
axes[0].set_ylabel('Family-macro MAE (pKa)'); axes[0].tick_params(axis='x',rotation=35); axes[0].set_title('Equal weight per sequence family')
bins=[0,.25,.5,1,2,np.inf]; labels=['<0.25','0.25–0.5','0.5–1','1–2','≥2']
counts=[sum(bins[i]<=abs(r['value']-r['null_pka'])<bins[i+1] for r in common) for i in range(len(labels))]
axes[1].bar(labels,counts,color='#dd862a'); axes[1].set_ylabel('Common experimental sites'); axes[1].set_xlabel('|experimental pKa − residue-type null|'); axes[1].set_title('Signal-strength composition')
fig.tight_layout(); fig.savefig(root/'family_metrics_and_signal.png',dpi=180); plt.close(fig)
atomic_json(root/'scoring_release.json',dict(scoring_code_sha256=digest(Path(__file__)),source_release_sha256=digest(source/'release_manifest.json'),
    initial_runner_code_sha256=run_manifest['code_sha256'],amended_runner_code_sha256=current_runner_sha,
    amendment='Deduplicate identical physical sites in adapter input; preserve every experimental record during scoring.',
    jax_recovery_manifest_sha256=digest(jax_recovery/'manifest.json') if use_jax_recovery else None,
    outputs={p.name:digest(p) for p in root.iterdir() if p.is_file() and p.name!='scoring_release.json'},
    model_fit=False,experimental_training=False,job=os.environ['SLURM_JOB_ID']))
print(json.dumps(dict(common_sites=len(common),families=len(families),metrics=metrics),indent=2),flush=True)
