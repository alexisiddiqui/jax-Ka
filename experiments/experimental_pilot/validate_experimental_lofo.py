"""Validate LOFO provenance, held-out isolation, selections and hashes."""
import csv
import json
import os
from pathlib import Path

from pkabench.runtime import digest, require_compute

require_compute()
root=Path(os.environ['PKABENCH_RUNTIME'])/'experimental'
source=root/'experimental-fit-v1';out=root/'experimental-lofo-v1'
manifest=json.loads((out/'manifest.json').read_text())
release=json.loads((out/'release.json').read_text())
rows=list(csv.DictReader((out/'heldout_predictions.csv').open()))

assert release['version']=='experimental-lofo-v1' and release['independent_evaluation']
assert release['source_release_sha256']==digest(source/'release.json')
assert release['manifest_sha256']==digest(out/'manifest.json')
for name,sha256 in release['outputs'].items():assert digest(out/name)==sha256,name
assert len(rows)==12 and len({r['record_id'] for r in rows})==12
assert len(manifest['families'])==5

heldout_seen=set()
for index,family in enumerate(manifest['families']):
    folder=out/f'fold-{index}'
    result=json.loads((folder/'result.json').read_text())
    assert result['heldout_family']==family and result['selection_used_heldout'] is False
    assert result['training_families']==4
    assert result['manifest_sha256']==digest(out/'manifest.json')
    assert digest(folder/'history.jsonl')==result['history_sha256']
    assert digest(folder/'checkpoint_audit.csv')==result['audit_sha256']
    assert digest(folder/'heldout_predictions.csv')==result['predictions_sha256']
    fold_rows=list(csv.DictReader((folder/'heldout_predictions.csv').open()))
    ids={r['record_id'] for r in fold_rows}
    assert ids==set(result['heldout_record_ids']) and not ids&heldout_seen
    assert {r['family_id'] for r in fold_rows}=={family}
    heldout_seen |= ids
    audit=list(csv.DictReader((folder/'checkpoint_audit.csv').open()))
    assert len(audit)==81
    selected=[r for r in audit if int(r['step'])==result['selected_step']][0]
    assert selected['coverage_preserving']=='True'
    best=min(float(r['training_objective']) for r in audit if r['coverage_preserving']=='True')
    assert abs(float(selected['training_objective'])-best)<1e-7
    assert digest(folder/'result.json')==release['fold_results_sha256'][f'fold-{index}']

assert heldout_seen=={r['record_id'] for r in rows}
frozen_valid=sum(r['frozen_valid']=='True' for r in rows)
lofo_valid=sum(r['lofo_valid']=='True' for r in rows)
common=sum(r['frozen_valid']=='True' and r['lofo_valid']=='True' for r in rows)
assert (frozen_valid,lofo_valid,common)==(10,6,6)
print(json.dumps({'status':'PASS','folds':5,'labels':12,'frozen_valid':frozen_valid,
    'lofo_valid':lofo_valid,'common_support':common,'outputs_verified':len(release['outputs'])},indent=2))
