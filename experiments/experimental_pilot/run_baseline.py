"""Run one frozen predictor on one prepared experimental candidate structure."""
import json
import os
from pathlib import Path
import sys
from pkabench.runtime import require_compute, atomic_json, digest

require_compute()
runtime = Path(os.environ['PKABENCH_RUNTIME'])
campaign = runtime / 'experimental/pilot-v2'
pdb, method = sys.argv[1:3]
prepared = {r['pdb_id']:r for r in json.loads((campaign/'structure_preflight.json').read_text())}
assert prepared[pdb]['status'] == 'prepared'
root = campaign/'structures'/pdb
assert digest(root/'AB.cif') == prepared[pdb]['prepared_sha256']
out = campaign/'predictions'/pdb/method
if method == 'jaxka':
    from pkabench.production_worker_v2 import run
    run(campaign, pdb, 'AB', out)
else:
    from pkabench.adapters.base import Adapter
    from pkabench.schema import read_table, write_table
    out.mkdir(parents=True, exist_ok=False)
    adapter = Adapter(method, read_table(root/'sites.parquet'), runtime, timeout=900)
    rows = adapter.run({'AB':root/'AB.cif'}, out)
    write_table(out/'predictions.parquet', 'predictions', rows)
    atomic_json(out/'receipt.json', dict(method=method, pdb_id=pdb, input_sha256=digest(root/'AB.cif'),
        job=os.environ['SLURM_JOB_ID'], errors=adapter.errors, timings=adapter.timings,
        implementation_sha256=digest(Path(__file__)), predictions_sha256=digest(out/'predictions.parquet')))
print(f'{pdb} {method} complete', flush=True)
