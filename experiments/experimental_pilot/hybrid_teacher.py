import json, os, sys
from pathlib import Path
from pkabench.runtime import require_compute, atomic_json, digest
require_compute()
from pkabench.adapters.base import Adapter
from pkabench.schema import read_table, write_table
runtime=Path(os.environ['PKABENCH_RUNTIME']); pdb=sys.argv[1]
source=runtime/'experimental/pilot-v2/structures'/pdb
out=runtime/'experimental/hybrid-v1'/pdb/'pypka'
out.mkdir(parents=True,exist_ok=False)
adapter=Adapter('pypka',read_table(source/'sites.parquet'),runtime,timeout=1800)
rows=adapter.run({'AB':source/'AB.cif'},out)
write_table(out/'predictions.parquet','predictions',rows)
atomic_json(out/'receipt.json',dict(errors=adapter.errors,timings=adapter.timings,
    input_sha256=digest(source/'AB.cif'),predictions_sha256=digest(out/'predictions.parquet'),job=os.environ['SLURM_JOB_ID']))
assert not adapter.errors,adapter.errors
print(pdb,'PypKa complete',flush=True)
