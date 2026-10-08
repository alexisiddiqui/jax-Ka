"""Validate experimental-fit-v2 provenance, selection and output hashes."""
import csv
import json
import os
from pathlib import Path

from pkabench.runtime import digest, require_compute

require_compute()
root=Path(os.environ['PKABENCH_RUNTIME'])/'experimental'
v1=root/'experimental-fit-v1';v2=root/'experimental-fit-v2'
release=json.loads((v2/'release.json').read_text())
selection=json.loads((v2/'selection.json').read_text())
rows=list(csv.DictReader((v2/'predictions.csv').open()))
audit=list(csv.DictReader((v2/'checkpoint_audit.csv').open()))

assert release['version']=='experimental-fit-v2' and release['coverage_gate_passed']
assert release['parent_release_sha256']==digest(v1/'release.json')
assert release['manifest_sha256']==digest(v2/'manifest.json')
for name,sha256 in release['outputs'].items():assert digest(v2/name)==sha256,name
assert len(rows)==12 and len({r['record_id'] for r in rows})==12
assert len(audit)==81 and selection['checkpoints_audited']==81
assert selection['selected_step']==3
assert selection['selected_valid_labels']==10
assert selection['unconstrained_final_valid_labels']==6
required=set(selection['required_record_ids'])
assert len(required)==10
assert all(r['selected_valid']=='True' for r in rows if r['record_id'] in required)
assert all(r['frozen_valid']=='True' for r in rows if r['record_id'] in required)
selected=[r for r in audit if int(r['step'])==selection['selected_step']][0]
assert selected['coverage_preserving']=='True'
admissible=[r for r in audit if r['coverage_preserving']=='True']
assert float(selected['objective'])==min(float(r['objective']) for r in admissible)
print(json.dumps({'status':'PASS','labels':len(rows),'checkpoints':len(audit),
    'selected_step':selection['selected_step'],'valid_labels':selection['selected_valid_labels'],
    'outputs_verified':len(release['outputs'])},indent=2))
