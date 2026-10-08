"""Validate the immutable v8 experimental-admission release on a compute node."""
import csv
import json
import os
from pathlib import Path

from pkabench.runtime import digest, require_compute

require_compute()

runtime = Path(os.environ['PKABENCH_RUNTIME'])/'experimental'
repo = Path(os.environ['PKABENCH_SOURCE'])
v7 = runtime/'experimental-admission-v7'
v8 = runtime/'experimental-admission-v8'
expected = {'142','311','317','584','585','589','590','916','917','1021','1022','1023'}

rows = list(csv.DictReader((v8/'admission_ledger.csv').open()))
assert len(rows) == 1024
by_id = {r['record_id']: r for r in rows}
assert len(by_id) == len(rows)
eligible = {r['record_id'] for r in rows if r['eligible_for_model_fit'] == 'True'}
assert eligible == expected, (sorted(eligible), sorted(expected))
assert not [r for r in rows if r['eligible_for_headline_evaluation'] == 'True']
for record_id in eligible:
    row = by_id[record_id]
    assert row['fit_gate_status'] == 'passed'
    assert row['admission_status'] in ('exact_candidate','recovered_exact_candidate')
    assert row['local_independent_candidate'] == 'True'
    assert row['structural_train_mask'] == 'True'
    assert row['structural_eval_mask'] == 'True'

expected_holds = {
    '102':'condition_context_hold', '110':'condition_domain_hold',
    '169':'condition_domain_hold', '318':'approximate_label_hold',
    '595':'evidence_pending', '918':'construct_mixture_hold',
    '944':'evidence_pending', '978':'evidence_pending',
    '1020':'label_correction_hold'}
for record_id,status in expected_holds.items():
    assert by_id[record_id]['fit_gate_status'] == status
    assert by_id[record_id]['eligible_for_model_fit'] == 'False'

expected_curated = {
    '584':('6.33','0.07'), '589':('5.71','0.05'),
    '585':('7.4','0.1'), '590':('7.5','0.2'), '1020':('6.7','0.1')}
for record_id,values in expected_curated.items():
    assert (by_id[record_id]['curated_experimental_pka'],
            by_id[record_id]['curated_experimental_uncertainty']) == values

old = {r['record_id']:(r['family_id'],r['cv_fold'])
       for r in csv.DictReader((v7/'candidate_ledger.csv').open())}
new = {r['record_id']:(r['family_id'],r['cv_fold'])
       for r in csv.DictReader((v8/'candidate_ledger.csv').open())}
assert new == old

summary = json.loads((v8/'summary.json').read_text())
assert summary['admitted_for_model_fit'] == 12
assert summary['admitted_for_headline_evaluation'] == 0
assert set(summary['fit_eligible_record_ids']) == expected

release = json.loads((v8/'release.json').read_text())
assert release['version'] == 'experimental-admission-v8'
assert release['eligibility_changed'] is True and release['model_fit'] is False
assert release['fit_gates_sha256'] == digest(repo/'experiments/1_benchmark/curation/experimental_fit_gates_v1.json')
for name,sha256 in release['outputs'].items():
    assert digest(v8/name) == sha256, name

print(json.dumps({
    'status':'PASS', 'records':len(rows),
    'fit_eligible_record_ids':sorted(eligible,key=int),
    'headline_eligible':0, 'folds_unchanged_from_v7':True,
    'release_outputs_verified':len(release['outputs'])}, indent=2))
