"""Report frozen baseline diagnostics without promoting unresolved labels to truth."""
import csv
import json
import os
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from pkabench.runtime import require_compute, atomic_json, digest
from pkabench.schema import NULL_PKA

require_compute()
runtime=Path(os.environ['PKABENCH_RUNTIME'])
root=runtime/'experimental/pilot-v2'
source=runtime/'experimental/pilot-v1'
labels=json.loads((source/'lead_residue_candidates.json').read_text())
preparation_receipt=json.loads((root/'preparation_receipt.json').read_text())
assert preparation_receipt['source_labels_sha256']==digest(source/'lead_residue_candidates.json')
structures=json.loads((root/'structure_preflight.json').read_text())
prepared={r['label_pdb_id']:r for r in structures if r['status']=='prepared'}
methods=('propka','pkai','jaxka')
identity=lambda r:(r['complex_id'],r['chain'],r['resnum'],r['icode'],r['group'])
gaps={identity(r):r for r in pq.read_table(root/'natural-gap-tiers-final/site_tiers.parquet').to_pylist()}
sites={}
components={}
predictions={}
prediction_status={}
for structure in prepared.values():
    pdb=structure['pdb_id']
    path=root/'structures'/pdb
    sites.update({identity(r):r for r in pq.read_table(path/'sites.parquet').to_pylist()})
    components.update({identity(r):r for r in json.loads((path/'component_masks.json').read_text())})
    for method in methods:
        result=root/'predictions'/pdb/method
        receipt=json.loads((result/'receipt.json').read_text())
        assert receipt['predictions_sha256']==digest(result/'predictions.parquet')
        assert receipt['input_sha256']==structure['prepared_sha256']
        rr=pq.read_table(result/'predictions.parquet').to_pylist()
        assert len({identity(r) for r in rr})==len(rr)
        predictions[pdb,method]={identity(r):r for r in rr}
        prediction_status[pdb+'_'+method]=dict(Counter(r['status'] for r in rr))

source_audit={
 '10.1021/bi9630927':dict(primary_url='https://pubmed.ncbi.nlm.nih.gov/9132009/',
    status='abstract_verified; full measurement table unavailable',
    findings=['Primary abstract confirms NMR measurements for B1/B2 and basic-site estimates.',
      'PKAD-R calls the 1IGD entry domain III; exact B2/domain-III construct equivalence requires checking.',
      'pH versus corrected pD conventions and per-site conditions require the full methods.']),
 '10.1021/bi00029a018':dict(primary_url='https://pubmed.ncbi.nlm.nih.gov/7626612/',
    status='abstract_verified; full measurement table unavailable',
    findings=['Both native and denatured measurements are described; preserve native-state assignment as unverified.',
      'Asp93 was inferred using stability/mutation measurements, not the generic NMR method attached to every database row.',
      '50 and 600 mM conditions must not be merged or treated as a single ionic strength.']),
 '10.1021/bi00241a021':dict(primary_url='https://pubmed.ncbi.nlm.nih.gov/2065058/',
    status='His18 value 7.75 corroborated in primary abstract; conditions/error unresolved',
    findings=['The 1991 paper describes fluorescence-based ionization assigned to His18 and compares it with earlier NMR.',
      'Do not silently retain the database NMR method as verified for this reference.']),
 '3173493':dict(primary_url='https://pubmed.ncbi.nlm.nih.gov/3173493/',
    status='abstract_verified; His102 value not verified from abstract',
    findings=['Abstract focuses on His18 and its native/denatured shift; the 6.3 value alone cannot verify the His102 row.',
      'This is an unresolved assignment check, not proof that the database is wrong.'])}
atomic_json(root/'source_audit.json',source_audit)
atomic_json(root/'pkai_overlap_assessment.json',dict(
    primary_source='https://pmc.ncbi.nlm.nih.gov/articles/PMC9369009/',
    source_sha256=digest(root/'sources/pkai-paper.xml'),
    published_policy='Exclude proteins containing a chain >90% identical to experimental/theoretical test chains from synthetic training.',
    experimental_test='736 sites in 97 PKAD proteins; exact membership of these candidates not independently established here.',
    exact_training_inventory_available=False,
    current_status='Upstream pKAI training membership unresolved; no independent-generalisation certification.',
    pkai_plus_note='Same synthetic labels as pKAI, but the regularization weight was selected using experimental-set performance; not run in this pilot.',
    distinction='Our local 30%/80% audit cannot certify upstream pKAI, PROPKA or inherited JAX parameter development data.'))

rows=[]
for label in labels:
    structure=prepared[label['pdb_id']]
    group={'C-term':'CTERM','N-term':'NTERM'}.get(label['residue_name'],label['residue_name'])
    key=(structure['pdb_id'],structure['chain'],int(label['residue_id']),'',group)
    site=sites.get(key)
    tier=gaps.get(key,{}).get('tier','unmapped')
    match=site is not None and (group in ('NTERM','CTERM') or site['restype']==group)
    component_ok=components.get(key,{}).get('component_eval_mask',False)
    mask=bool(match and tier in ('clean','uncertain') and component_ok)
    refs=label['primary_reference']
    audit_id=next((k for k in source_audit if k in refs),None)
    assert audit_id is not None, refs
    row=dict(observation_id=label['observation_id'], protein=label['protein'],
        family='barnase' if 'barnase' in label['protein'].lower() else 'protein_g',
        original_pdb_id=label['pdb_id'], scoring_pdb_id=structure['pdb_id'],
        chain=key[1],resnum=key[2],group=group,value=label['value'],raw_label=label['raw_label'],
        label_kind=label['label_kind'], natural_gap_tier=tier, structural_eval_mask=mask,
        residue_mapping_verified=bool(match), source_audit_id=audit_id,
        source_status=source_audit[audit_id]['status'], raw_conditions=label['source_row'],
        primary_value_corroborated=label['observation_id']=='pkadr-26',
        experimental_condition_match_verified=False, experimental_construct_match_verified=False,
        training_eligible=False, independent_evaluation_eligible=False,
        null_pka=NULL_PKA[group])
    for method in methods:
        prediction=predictions[structure['pdb_id'],method].get(key,{})
        row[method+'_status']=prediction.get('status','not_reported')
        row[method+'_pka']=prediction.get('pka') if prediction.get('status')=='ok' else None
    rows.append(row)

common=[r for r in rows if r['label_kind']=='point' and r['structural_eval_mask'] and all(r[m+'_pka'] is not None for m in methods)]
assert common and len({r['observation_id'] for r in rows})==42
assert not any(r['training_eligible'] or r['independent_evaluation_eligible'] for r in rows)
assert all(r['value'] is None for r in rows if r['label_kind']!='point')

def metrics(data,method):
    truth=np.array([r['value'] for r in data]); pred=np.array([r[method+'_pka'] for r in data]); null=np.array([r['null_pka'] for r in data])
    mse=float(np.mean((pred-truth)**2)); null_mse=float(np.mean((null-truth)**2))
    ds=truth-null; ps=pred-null; sign=np.abs(ds)>=.5
    corr=lambda a,b:float(spearmanr(a,b).statistic) if len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0 else None
    return dict(n=len(data),mae=float(np.abs(pred-truth).mean()),rmse=float(np.sqrt(mse)),
        skill=1-mse/null_mse if null_mse else None,spearman_absolute=corr(truth,pred),
        spearman_shift=corr(ds,ps),sign_shift_n=int(sign.sum()),
        sign_shift_accuracy=float(np.mean(np.sign(ds[sign])==np.sign(ps[sign]))) if sign.any() else None)

table=[dict(method=m,**metrics(common,m)) for m in (*methods,'null')]
families={f:[dict(method=m,**metrics([r for r in common if r['family']==f],m)) for m in (*methods,'null')]
          for f in sorted({r['family'] for r in common})}
censored=[]
for r in rows:
    if r['label_kind']!='censored' or not r['structural_eval_mask']: continue
    import re
    bound=float(re.findall(r'\d+(?:\.\d+)?',r['raw_label'])[0])
    for m in methods:
        pred=r[m+'_pka']
        if pred is not None:
            violation=max(0.,bound-pred) if '>' in r['raw_label'] else max(0.,pred-bound)
            censored.append(dict(observation_id=r['observation_id'],method=m,raw_label=r['raw_label'],prediction=pred,one_sided_violation=violation))
atomic_json(root/'audited_observations.json',rows)
atomic_json(root/'baseline_metrics.json',dict(scope='PROVISIONAL secondary-label diagnostics; not a verified independent experimental benchmark',
    common_site_metrics=table,by_family=families,common_ids=[r['observation_id'] for r in common],
    prediction_status=prediction_status,censored_diagnostics=censored,
    shift_definition='absolute pKa minus the group-specific fixed null; NOT a binding shift',
    uncertainty='No bootstrap CI: only two protein families. No claim of model ranking generalisation.'))
with (root/'baseline_sites.csv').open('w') as f:
    fields=[k for k in rows[0] if k!='raw_conditions']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
counts=dict(candidates=len(rows),point_labels=sum(r['label_kind']=='point' for r in rows),
    structural_mask_retained=sum(r['structural_eval_mask'] for r in rows),
    point_mask_retained=sum(r['structural_eval_mask'] and r['label_kind']=='point' for r in rows),
    common_scored_points=len(common),gap_excluded=sum(r['natural_gap_tier']=='near_gap' for r in rows),
    training_ready=0,independent_evaluation_ready=0,
    families={f:dict(candidates=sum(r['family']==f for r in rows),retained=sum(r['family']==f and r['structural_eval_mask'] for r in rows),
                    common_points=sum(r['family']==f for r in common)) for f in ('barnase','protein_g')})
atomic_json(root/'counts.json',counts)
overlap=json.loads((root/'synthetic_overlap.json').read_text())
assert all(c['train']==0 and c['val']==0 for c in overlap['counts'].values())
assert overlap['assignment_sha256']==digest(runtime/'universe/structural-freeze-v1/assignments.parquet')
missing=json.loads((root/'missing_output_explanations.json').read_text())
assert any(r['original_site']==['A',60,''] for r in missing['propka_suppressed'])
assert all(not (r['group']=='NTERM' and r['scoring_pdb_id']=='1BNI') for r in rows)

lines=['# Experimental pilot v2: structural audit and frozen baselines','',
    '**These are provisional diagnostics against PKAD-R labels. Primary tables and exact experimental conditions/constructs remain unresolved; no labels have been admitted to training or independent headline evaluation.**','',
    '| Stage | Count |','|---|---:|',
    f"| Candidate measurements | {len(rows)} |",f"| Numeric point labels | {counts['point_labels']} |",
    f"| Structurally retained records, including censored/approximate | {counts['structural_mask_retained']} |",
    f"| Structurally retained point labels | {counts['point_mask_retained']} |",
    f"| Common point labels scored by all three methods | {len(common)} |",
    f"| Near-gap exclusions | {counts['gap_excluded']} |",'| Training-ready / verified independent evaluation | 0 / 0 |','',
    '## Frozen baseline comparison','',
    'All rows use exactly the same masked point observations. MAE/RMSE are in pKa units; skill is 1 − MSE/MSE_null. This is absolute-pKa scoring, not bound-minus-free scoring. Censored/approximate records do not enter these metrics.','',
    '| Method | Sites | MAE | RMSE | Skill | Spearman, absolute pKa | Shift sign accuracy |','|---|---:|---:|---:|---:|---:|---:|']
for m in table:
    lines.append(f"| {m['method']} | {m['n']} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['skill']:.3f} | {m['spearman_absolute']:.3f} | {100*m['sign_shift_accuracy']:.0f}% |")
lines += ['', 'Sign accuracy uses the 10 common sites with |experimental pKa − null pKa| ≥ 0.5. Null predicts zero shift, counted as incorrect for these nonzero shifts.', '',
    '| Family | Common sites | PROPKA MAE | pKAI MAE | JAX-Ka MAE | Null MAE |', '|---|---:|---:|---:|---:|---:|']
for family,rr in families.items():
    mm={r['method']:r for r in rr}
    lines.append(f"| {family} | {rr[0]['n']} | {mm['propka']['mae']:.3f} | {mm['pkai']['mae']:.3f} | {mm['jaxka']['mae']:.3f} | {mm['null']['mae']:.3f} |")
lines += ['', 'Absolute and null-relative Spearman, null-relative sign accuracy at |experimental shift| ≥ 0.5, per-family metrics and individual predictions are included in baseline_metrics.json / baseline_sites.csv. These shifts are not binding shifts. Two protein families do not support a reliable bootstrap confidence interval or general model ranking.','',
    'Of the 30 structurally retained point labels, two are absent from the common-method subset: PROPKA suppresses barnase Glu60 through its covalent-coupling heuristic (verified from the live model object; not a claim of a real covalent bond), and released pKAI does not implement terminal predictions for the barnase C-terminus. Missing values are not imputed.','',
    '## Structure selection and masks','',
    '| Candidate | Outcome |','|---|---|',
    '| 1A2P barnase | Excluded: declared coordinated zinc; not silently stripped |',
    '| 1BNI barnase alternative | Deposited WT structure, exact canonical sequence match to 1A2P; chain A isolated; two missing N-terminal residues |',
    '| 1PGB / 1IGD | Both prepared; no missing-segment or component exclusions |','',
    'Preparation reuses PDB2PQR completion with observed heavy atoms preserved, the first deposited residue-coherent positive-occupancy alternate, and the existing gap/component code. No structure was predicted. The isolated-chain adapter does not invent a second partner or interface. Omitted barnase crystal copies B/C are recorded. [1BNI deposition](https://www.rcsb.org/structure/1BNI) identifies monomeric biological assemblies.','',
    'The two-residue tail uses the existing rounded-up 15 Å anchor rule; clean and uncertain tiers are retained. The same natural-gap tier is used for training/evaluation in the current policy, with separate component radii. There are no removable components in the selected prepared structures. These masks were calibrated for synthetic paired labels: using them for absolute experimental pKas remains an operational extrapolation, not an experimental error guarantee.','',
    '## Primary-source checks','',
    '| Source | Findings and unresolved work |','|---|---|',
    '| [Protein G, Khare 1997](https://pubmed.ncbi.nlm.nih.gov/9132009/) | Abstract confirms B1/B2 NMR. Exact table values, conditions, pD convention and the 1IGD/domain-III naming need full-text verification. |',
    '| [Barnase carboxyls, Oliveberg 1995](https://pubmed.ncbi.nlm.nih.gov/7626612/) | Distinguish native/denatured and 50/600 mM conditions. Asp93 uses mutation/stability inference; generic NMR metadata is insufficient. |',
    '| [Barnase His18, Loewenthal 1991](https://pubmed.ncbi.nlm.nih.gov/2065058/) | Value 7.75 is corroborated by the abstract; fluorescence/NMR method attribution, error and conditions unresolved. |',
    '| [Barnase histidines, Sali 1988](https://pubmed.ncbi.nlm.nih.gov/3173493/) | Abstract discusses His18 native/denatured values; it does not establish the database His102 assignment. |','',
    'Full original measurement tables were not available from the public pages retrieved. No values, uncertainties or conditions were silently corrected. The raw source records and primary abstract receipts remain available. C-terminal Arg110 is mapped as a terminal group, separately from its ARG side chain.','',
    '## Family overlap','',
    '| Structural query | Frozen train matches | Validation matches | Test matches |','|---|---:|---:|---:|']
for q,c in overlap['counts'].items(): lines.append(f"| {q} | {c['train']} | {c['val']} | {c['test']} |")
lines += ['',
    'Matches use 30% sequence identity and 80% bidirectional coverage against all frozen candidate chains. No local training/validation matches were found. This is not proof of absence of remote homologues or shorter embedded domains. Existing synthetic assignments were not changed.','',
    '[The pKAI paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9369009/) describes a >90% chain-identity exclusion around its experimental test set. We have not established these candidates’ exact membership in that published test inventory or inspected the full upstream training list. Local disjointness therefore does not certify upstream pKAI independence. pKAI+ used the same synthetic training targets, but its regularization was chosen using experimental performance; pKAI+ was not run here. Historical PROPKA/JAX parameter-development overlap also remains unverified.','',
    '## Remaining admission work','',
    'Obtain and check the four primary measurement papers; resolve construct/condition/state assignments, then freeze the eligible experimental evaluation manifest. Keep these two families reserved as evaluation candidates. Additional unrelated families from the archived PKAD-R inventory are needed before experimental fine-tuning.','',
    '## Artifacts and validation','',
    'audited_observations.json and baseline_sites.csv contain every candidate and its mask/prediction status. Source audit, model-overlap assessment, numerical metrics, source/coordinate hashes, prediction receipts and sequence-search output are stored alongside this report. All nine method/structure runs completed; per-site failures and unsupported outputs are counted rather than converted to values. All scientific work ran in Slurm allocations excluding comp1400, at two CPUs and 2 GB/core per task.']
(root/'report.md').write_text('\n'.join(lines)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
for ax,m in zip(axes,methods):
    for family,color in [('barnase','#e68613'),('protein_g','#2676b8')]:
        rr=[r for r in common if r['family']==family]
        ax.scatter([r['value'] for r in rr],[r[m+'_pka'] for r in rr],s=28,label=family,color=color,alpha=.8)
    ax.plot([0,13],[0,13],color='gray',linestyle='--',linewidth=1)
    ax.set(title=m,xlabel='PKAD-R experimental pKa',xlim=(0,13),ylim=(0,13))
axes[0].set_ylabel('Predicted pKa'); axes[-1].legend(fontsize=8)
fig.suptitle(f'Provisional secondary-label diagnostics: {len(common)} common masked points')
fig.tight_layout(); fig.savefig(root/'baseline_comparison.png',dpi=180); plt.close(fig)
atomic_json(root/'release_manifest.json',dict(version='experimental-pilot-v2',job=os.environ['SLURM_JOB_ID'],
    scoring_implementation_sha256=digest(Path(__file__)),
    outputs={p.name:digest(p) for p in root.iterdir() if p.is_file() and p.name!='release_manifest.json'},
    sources_unchanged=True,training_started=False))
print(json.dumps(dict(counts=counts,metrics=table),indent=2),flush=True)
