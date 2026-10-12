import json
from pathlib import Path
B=Path(__file__).resolve().parent
d=json.loads((B/'reference_membership.json').read_text())
ids=set(d['benchmark_val'])|set(d['benchmark_test'])
seq=lambda name:{s for ss in d[name].values() for s in ss}
r=dict(reserved_outside_val_test=sorted(set(d['benchmark_reserved'])-ids),
       heldout_sequences_outside_val_test=sorted(seq('heldout_all')-(seq('benchmark_val')|seq('benchmark_test'))),
       heldout_ids_outside_val_test=sorted(set(d['heldout_all'])-ids))
(B/'reference-scope-check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
