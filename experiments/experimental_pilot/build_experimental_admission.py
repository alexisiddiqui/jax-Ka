"""Build the versioned PKAD-R admission ledger and family-blocked CV folds.

This consumes immutable structural/baseline releases. It does not fit a model
or promote unverified secondary labels to experimental truth.
"""
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from pkabench.runtime import require_compute, atomic_json, config_hash, digest

require_compute()
import pyarrow as pa
import pyarrow.parquet as pq
from pkabench.schema import NULL_PKA

runtime = Path(os.environ['PKABENCH_RUNTIME'])
source = runtime/'experimental/pkadr-full-v1'
baselines = runtime/'experimental/pkadr-baselines-v2'
recovery = runtime/'experimental/pkadr-dsba-reduced-v1'
trx_recovery = runtime/'experimental/pkadr-human-thioredoxin-recovery-v4'
repo = Path(os.environ['PKABENCH_SOURCE'])
decisions_path = repo/'experiments/1_benchmark/curation/primary_label_checks_v1.json'
out = runtime/'experimental/experimental-admission-v7'
out.mkdir(parents=True, exist_ok=False)

joined = json.loads((source/'joined-records.json').read_text())
checks = json.loads(decisions_path.read_text())
scored = {r['record_id']:r for r in csv.DictReader((baselines/'scored_records.csv').open())}
check_by_record = {}
for check in checks['checks']:
    for record_id in check['records']:
        key=str(record_id)
        if key in check_by_record:
            raise ValueError(f'duplicate primary decision for record {key}')
        check_by_record[key]=check

dsba_manifest=json.loads((recovery/'manifest.json').read_text())
dsba_gap=json.loads((recovery/'gap_annotation.json').read_text())
assert dsba_manifest['original_record_id']=='142'
assert dsba_manifest['sequence_matches_1DSB'] and dsba_manifest['redox_state']=='reduced'
assert dsba_gap['structural_train_mask'] and dsba_gap['structural_eval_mask']
trx_manifest=json.loads((trx_recovery/'manifest.json').read_text())
trx_by_record={str(record): item for item in trx_manifest['structures'] for record in item['records']}
for record,item in trx_by_record.items():
    if digest(trx_recovery/item['pdb']/'prepared.cif')!=item['prepared_sha256']:
        raise ValueError(f'thioredoxin recovery structure changed for record {record}')

def primary_state(record_id):
    check=check_by_record.get(record_id)
    if check is None:
        return 'pending_primary', 'Primary measurement, construct and state have not been checked.', None
    decision=check['decision']
    if check.get('recovery_decision','').startswith('admit_candidate_as_new_'):
        if check.get('recovery_result'):
            return 'recovered_exact_candidate', check['recovery_result'], check
        return 'recovery_pending', check['reason'], check
    if decision=='admit_candidate':
        return 'exact_candidate', check.get('remaining_gate','Final primary-source gate remains open.'), check
    if 'construct_mismatch' in decision:
        return 'surrogate', check['reason'], check
    return 'held', check['reason'], check

ledger=[]
for r in joined:
    raw=r['raw']; record_id=r['record_id']
    primary_status, rationale, check=primary_state(record_id)
    local_point=bool(r['local_independent_candidate'] and r['label_kind']=='point')
    if r['label_kind']=='censored':
        status='censored_pending' if r['structural_eval_mask'] else 'structural_hold'
    elif not r['structural_eval_mask']:
        status='structural_hold'
    elif not local_point:
        status='overlap_or_nonpoint_hold'
    else:
        status=primary_status
    structure_pdb=r['pdb']; structure_task_id=r['task_id']; structure_sha=r.get('prepared_sha256')
    natural_gap_tier=r.get('natural_gap_tier'); train_mask=bool(r['structural_train_mask']); eval_mask=bool(r['structural_eval_mask'])
    replacement=False
    if record_id=='142':
        structure_pdb='1A2L'; structure_task_id=dsba_manifest['task_id']; structure_sha=dsba_manifest['prepared_sha256']
        natural_gap_tier=dsba_gap['natural_gap_tier']; train_mask=dsba_gap['structural_train_mask']; eval_mask=dsba_gap['structural_eval_mask']
        replacement=True
    elif record_id in trx_by_record:
        recovered=trx_by_record[record_id]
        structure_pdb=recovered['pdb']; structure_task_id=f"human-trx-{recovered['pdb'].lower()}"
        structure_sha=recovered['prepared_sha256']; natural_gap_tier='clean'
        train_mask=True; eval_mask=True; replacement=True
    null=NULL_PKA.get(r.get('group'))
    shift=abs(float(r['value'])-null) if r['label_kind']=='point' and r['value'] is not None and null is not None else None
    priority=0 if check else 1 if shift is not None and shift>=2 else 2 if shift is not None and shift>=.5 else 3
    pred=scored.get(record_id,{})
    ledger.append(dict(
        record_id=record_id,family_id=r.get('family_id'),original_pdb=r['pdb'],author_chain=r['author_chain'],
        structure_pdb=structure_pdb,structure_task_id=structure_task_id,structure_replacement=replacement,
        prepared_sha256=structure_sha,chain=r.get('chain'),resnum=r.get('resnum'),icode=r.get('icode'),group=r.get('group'),
        protein_name=raw.get('Protein_Name'),species=raw.get('Species'),mutation=raw.get('Mut_Pos'),
        label_kind=r['label_kind'],experimental_pka=r['value'],raw_label=raw.get('Expt_pKa'),
        experimental_uncertainty=raw.get('Expt_Uncertainty'),experimental_temperature=raw.get('Expt_Temp'),
        experimental_pH=raw.get('Expt_pH'),experimental_salt=raw.get('Expt_Salt_Concentration'),
        experimental_method=raw.get('Expt_Method'),reference=raw.get('Reference'),archive_warning=raw.get('Warning'),
        archive_notes=raw.get('Notes'),null_pka=null,null_relative_shift=shift,
        task_status=r['task_status'],task_reason=r.get('task_reason'),mapped=bool(r['mapped']),
        natural_gap_tier=natural_gap_tier,structural_train_mask=train_mask,structural_eval_mask=eval_mask,
        frozen_train_matches=r['frozen_train_matches'],frozen_val_matches=r['frozen_val_matches'],
        frozen_test_matches=r['frozen_test_matches'],local_independent_candidate=local_point,
        primary_check_status=primary_status,admission_status=status,admission_rationale=rationale,
        proposed_structure_pdb=None if check is None else check.get('recovery_structure'),
        remaining_gate=None if check is None else check.get('remaining_gate'),manual_review_priority=priority,
        primary_metadata_corrections=None if check is None else json.dumps(check.get('metadata_corrections',{}),sort_keys=True),
        label_interpretation=None if check is None else check.get('label_interpretation'),
        interpretation_panel=bool(local_point and (check is not None or (shift is not None and shift>=2))),
        propka_status=pred.get('propka_status'),pkai_status=pred.get('pkai_status'),
        pkai_plus_status=pred.get('pkai_plus_status'),jaxka_status=pred.get('jaxka_status'),pypka_status=pred.get('pypka_status'),
        eligible_for_model_fit=False,eligible_for_headline_evaluation=False))

assert len(ledger)==1024 and len({r['record_id'] for r in ledger})==1024

# Five deterministic family folds over the locally independent point-label
# candidate universe. These are candidate folds: label admission remains a
# separate column and cannot be inferred from a fold assignment.
candidates=[r for r in ledger if r['local_independent_candidate']]
families=defaultdict(list)
for row in candidates:
    if not row['family_id']: raise ValueError(f"candidate {row['record_id']} lacks family")
    families[row['family_id']].append(row)
residue_types=sorted({r['group'] for r in candidates})
stats={}
for family,rows in families.items():
    stats[family]=dict(total=len(rows),shifted=sum((r['null_relative_shift'] or 0)>=.5 for r in rows),
        large=sum((r['null_relative_shift'] or 0)>=2 for r in rows),
        residue=Counter(r['group'] for r in rows))
totals=dict(total=len(candidates),shifted=sum(s['shifted'] for s in stats.values()),large=sum(s['large'] for s in stats.values()),
            residue=Counter(r['group'] for r in candidates))
loads=[dict(total=0,shifted=0,large=0,residue=Counter(),families=[]) for _ in range(5)]

def fold_cost(load,stat):
    # Normalize each dimension by its expected per-fold load. Large families
    # and shifted examples receive explicit weight; deterministic hashes break ties.
    values=((load['total']+stat['total'],totals['total']/5,2.),
            (load['shifted']+stat['shifted'],max(totals['shifted']/5,1),2.),
            (load['large']+stat['large'],max(totals['large']/5,1),1.))
    score=sum(weight*(value/target)**2 for value,target,weight in values)
    for group in residue_types:
        target=max(totals['residue'][group]/5,1)
        score+=.5*((load['residue'][group]+stat['residue'][group])/target)**2
    return score

order=sorted(families,key=lambda f:(-stats[f]['total'],-stats[f]['shifted'],-stats[f]['large'],f))
family_fold={}
for family in order:
    scores=[(fold_cost(loads[i],stats[family]),config_hash([family,str(i)]),i) for i in range(5)]
    fold=min(scores)[2]; family_fold[family]=fold; load=loads[fold]; stat=stats[family]
    for key in ('total','shifted','large'): load[key]+=stat[key]
    load['residue'].update(stat['residue']); load['families'].append(family)

def assignment_loads(assignment):
    current=[dict(total=0,shifted=0,large=0,residue=Counter(),families=[]) for _ in range(5)]
    for family,fold in assignment.items():
        load=current[fold]; stat=stats[family]
        for key in ('total','shifted','large'): load[key]+=stat[key]
        load['residue'].update(stat['residue']); load['families'].append(family)
    return current

def assignment_cost(assignment):
    current=assignment_loads(assignment); score=0.
    targets=dict(total=totals['total']/5,shifted=max(totals['shifted']/5,1),large=max(totals['large']/5,1))
    for load in current:
        score += 4*((load['total']-targets['total'])/targets['total'])**2
        score += 2*((load['shifted']-targets['shifted'])/targets['shifted'])**2
        score += ((load['large']-targets['large'])/targets['large'])**2
        score += .1*((len(load['families'])-len(families)/5)/max(len(families)/5,1))**2
        for group in residue_types:
            target=max(totals['residue'][group]/5,1)
            score += .25*((load['residue'][group]-target)/target)**2
    return score

# Greedy placement is followed by deterministic single-family moves and swaps.
# This matters when a few large families make the first placement order sticky.
for _ in range(100):
    current=assignment_cost(family_fold); candidates_moves=[]
    counts=Counter(family_fold.values())
    for family in sorted(family_fold):
        old=family_fold[family]
        if counts[old]>1:
            for new in range(5):
                if new==old: continue
                trial=dict(family_fold); trial[family]=new
                candidates_moves.append((assignment_cost(trial),('move',family,old,new),trial))
    names=sorted(family_fold)
    for ai,a in enumerate(names):
        for b in names[ai+1:]:
            if family_fold[a]==family_fold[b]: continue
            trial=dict(family_fold); trial[a],trial[b]=trial[b],trial[a]
            candidates_moves.append((assignment_cost(trial),('swap',a,b),trial))
    best=min(candidates_moves,key=lambda x:(x[0],x[1]))
    if best[0] >= current-1e-12: break
    family_fold=best[2]
loads=assignment_loads(family_fold)

for row in ledger:
    row['cv_fold']=family_fold.get(row['family_id'])
    row['fold_scope']='candidate_family_cv' if row['family_id'] in family_fold else None

for family,rows in families.items():
    assert {r['cv_fold'] for r in rows}=={family_fold[family]}
assert set(family_fold.values())==set(range(5))

folds=[]
for i,load in enumerate(loads):
    folds.append(dict(fold=i,families=len(load['families']),records=load['total'],shifted_records=load['shifted'],
        large_shift_records=load['large'],residue_types=dict(sorted(load['residue'].items())),
        family_ids=sorted(load['families'])))

pq.write_table(pa.Table.from_pylist(ledger),out/'admission_ledger.parquet')
with (out/'admission_ledger.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(ledger[0])); writer.writeheader(); writer.writerows(ledger)
with (out/'candidate_ledger.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(ledger[0])); writer.writeheader(); writer.writerows(candidates)
review_fields=('manual_review_priority','record_id','family_id','protein_name','species','original_pdb','author_chain',
               'group','experimental_pka','null_relative_shift','mutation','experimental_method','reference',
               'archive_warning','archive_notes','admission_status')
review_queue=sorted((r for r in candidates if r['primary_check_status']=='pending_primary'),
                    key=lambda r:(r['manual_review_priority'],-(r['null_relative_shift'] or 0),r['family_id'],r['record_id']))
with (out/'primary_review_queue.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=review_fields); writer.writeheader()
    writer.writerows({k:r[k] for k in review_fields} for r in review_queue)
atomic_json(out/'folds.json',folds)
atomic_json(out/'primary_checks_snapshot.json',checks)

status_counts=Counter(r['admission_status'] for r in ledger)
candidate_status=Counter(r['admission_status'] for r in candidates)
summary=dict(records=len(ledger),local_independent_point_candidates=len(candidates),candidate_families=len(families),
    status_counts=dict(status_counts),candidate_status_counts=dict(candidate_status),primary_checked_records=len(check_by_record),
    interpretation_panel_records=sum(r['interpretation_panel'] for r in ledger),fold_assignment_cost=assignment_cost(family_fold),folds=folds,
    admitted_for_model_fit=0,admitted_for_headline_evaluation=0,
    note='Fold assignment does not imply label admission. Candidate and recovered-candidate decisions retain their named remaining gates.')
atomic_json(out/'summary.json',summary)

lines=['# Experimental admission ledger and candidate folds — v7','',
 '**No label is admitted to model fitting or headline evaluation by this release.** The ledger separates structural candidacy, primary-source status and fold assignment so downstream code cannot equate one with another.','',
 '## Candidate population','', '| Quantity | Count |','|---|---:|',
 f"| Archive records | {len(ledger)} |",f"| Locally independent structural point candidates | {len(candidates)} |",
 f"| Candidate sequence families | {len(families)} |",f"| Records covered by current primary checks | {len(check_by_record)} |",
 f"| Interpretation-panel records | {summary['interpretation_panel_records']} |",'',
 '## Admission state for the candidate population','', '| State | Records |','|---|---:|']
for status,n in candidate_status.most_common(): lines.append(f'| {status} | {n} |')
lines += ['', 'An `exact_candidate` still has the explicit `remaining_gate` recorded by its primary check. `recovered_exact_candidate` applies to DsbA Cys30 on reduced 1A2L and the three prepared human-thioredoxin replacements; every original mismatched mapping remains preserved in the ledger. `pending_primary` records have not been promoted from the secondary archive.','',
 '## Frozen candidate folds','', '| Fold | Families | Records | Shifted >=0.5 | Large shift >=2 |','|---:|---:|---:|---:|---:|']
for f in folds: lines.append(f"| {f['fold']} | {f['families']} | {f['records']} | {f['shifted_records']} | {f['large_shift_records']} |")
lines += ['', 'Every sequence family occurs in exactly one fold. The deterministic greedy assignment balances family size, null-relative signal and residue composition. These folds cover all candidate labels without making an admission decision; future primary-source decisions change eligibility columns, not fold membership.','',
 '## Required next gate','',
 'Review `manual_review_priority` in order: already checked records, shifts of at least 2 pKa, shifts of at least 0.5 pKa, then near-null records. Exact construct, mutation, state and primary measurement provenance must be resolved before setting either eligibility flag. Censored observations remain outside the point-label folds until an interval-aware objective is released.','',
 '`candidate_ledger.csv` is the fold population and `primary_review_queue.csv` orders its unchecked labels by information content. Other machine-readable outputs are `admission_ledger.parquet`, `admission_ledger.csv`, `folds.json`, `summary.json` and the immutable primary-check snapshot.']
(out/'report.md').write_text('\n'.join(lines)+'\n')
atomic_json(out/'release.json',dict(version='experimental-admission-v7',created='2026-10-06',
    source_joined_sha256=digest(source/'joined-records.json'),baseline_scored_sha256=digest(baselines/'scored_records.csv'),
    dsba_release_sha256=digest(recovery/'release.json'),trx_recovery_sha256=digest(trx_recovery/'manifest.json'),primary_checks_sha256=digest(decisions_path),
    code_sha256=digest(Path(__file__)),outputs={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='release.json'},
    model_fit=False,eligibility_changed=False,job=os.environ['SLURM_JOB_ID']))
print(json.dumps(summary,indent=2),flush=True)
