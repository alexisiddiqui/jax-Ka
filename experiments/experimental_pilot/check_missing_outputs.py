"""Explain absent method outputs without changing predictions or admission gates."""
import json
import os
from pathlib import Path
from pkabench.runtime import require_compute, atomic_json, digest
require_compute()
root=Path(os.environ['PKABENCH_RUNTIME'])/'experimental/pilot-v2'
work=root/'method_output_checks'
work.mkdir(exist_ok=True)
os.chdir(work)
from propka.run import single
model=single(str(root/'structures/1BNI/input.pdb'),write_pka=False)
excluded=[dict(chain=g.atom.chain_id,exported_resnum=int(g.atom.res_num),group=g.residue_type,
    coupled_label=g.coupled_titrating_group.label,reason='propka_covalent_coupling_suppression')
    for g in model.conformations['AVR'].groups if g.coupled_titrating_group and model.version.parameters.remove_penalised_group]
mapping={(r['chain'],r['resnum']):r['original'] for r in json.loads((root/'structures/1BNI/pdb_mapping.json').read_text())}
for r in excluded: r['original_site']=mapping[r['chain'],r['exported_resnum']]
atomic_json(root/'missing_output_explanations.json',dict(propka_suppressed=excluded,
    pkai_terminal_support='Released pKAI Protein.termini is explicitly unimplemented; terminal groups are not predicted.',
    pkai_source_sha256=digest(Path(os.environ['PKABENCH_RUNTIME'])/'sources/pKAI/pKAI/protein.py'),
    predictions_changed=False))
print(json.dumps(excluded,indent=2),flush=True)
