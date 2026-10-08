"""Run the accepted 1,024-step JAX-Ka configuration without altering v1."""
import json
import os
import sys
from pathlib import Path

from pkabench.runtime import require_compute, atomic_json, digest

require_compute()
runtime=Path(os.environ['PKABENCH_RUNTIME'])
source=runtime/'experimental/pkadr-full-v1'
baseline=runtime/'experimental/pkadr-baselines-v1'
out=runtime/'experimental/pkadr-jaxka-1024-v1'

def init():
    tasks=json.loads((baseline/'tasks.json').read_text())
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'tasks.json',tasks)
    atomic_json(out/'manifest.json',dict(
        source=str(source),baseline=str(baseline),tasks=len(tasks),method='jaxka',steps=1024,
        residual_tolerance=2e-5,model_fit=False,experimental_training=False,
        policy='Accepted frozen production numerical configuration; v1 64-step outputs preserved.',
        code_sha256=digest(Path(__file__)),worker_sha256=digest(Path(os.environ['PKABENCH_SOURCE'])/'src/pkabench/adapters/worker.py'),
        source_release_sha256=digest(source/'release_manifest.json')))
    print(json.dumps(dict(tasks=len(tasks),steps=1024),indent=2),flush=True)

def work(index):
    from pkabench.adapters.base import Adapter
    from pkabench.schema import write_table
    tasks=json.loads((out/'tasks.json').read_text()); task=tasks[index]
    joined={r['record_id']:r for r in json.loads((source/'joined-records.json').read_text())}
    rows=[joined[i] for i in task['records']]
    sites=[dict(complex_id=task['task_id'],chain=r['chain'],resnum=r['resnum'],icode=r['icode'],group=r['group']) for r in rows]
    sites=list({tuple(s[k] for k in ('complex_id','chain','resnum','icode','group')):s for s in sites}.values())
    structure=source/'structures'/task['task_id']/'prepared.cif'
    assert digest(structure)==task['prepared_sha256']
    root=out/'tasks'/task['task_id']; dest=root/'jaxka'
    root.mkdir(parents=True,exist_ok=False); dest.mkdir()
    adapter=Adapter('jaxka',sites,runtime,timeout=1800)
    adapter.config['steps']=1024
    pred=adapter.run({'AB':structure},dest)
    write_table(dest/'predictions.parquet','predictions',pred)
    receipt=dict(method='jaxka',task_id=task['task_id'],pdb=task['pdb'],steps=1024,
        input_sha256=task['prepared_sha256'],prediction_sha256=digest(dest/'predictions.parquet'),
        errors=adapter.errors,timings=adapter.timings,extra=adapter.extras,job=os.environ['SLURM_JOB_ID'])
    atomic_json(dest/'receipt.json',receipt)
    atomic_json(root/'receipt.json',dict(task_id=task['task_id'],methods=['jaxka'],
        outputs={'jaxka':receipt['prediction_sha256']},job=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(index=index,task_id=task['task_id'],statuses={s:sum(r['status']==s for r in pred) for s in set(r['status'] for r in pred)},
                               max_residual=adapter.extras.get('AB',{}).get('max_residual'))),flush=True)

if sys.argv[1]=='init': init()
else: work(int(os.environ['SLURM_ARRAY_TASK_ID']))
