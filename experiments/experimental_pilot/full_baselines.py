"""Frozen absolute-pKa baselines over the structurally retained PKAD-R inventory."""
import json
import os
import sys
from pathlib import Path

from pkabench.runtime import require_compute, atomic_json, digest

require_compute()
runtime=Path(os.environ['PKABENCH_RUNTIME'])
source=runtime/'experimental/pkadr-full-v1'
out=runtime/'experimental/pkadr-baselines-v1'

def init():
    joined=json.loads((source/'joined-records.json').read_text())
    tasks=[]
    for task_id in sorted({r['task_id'] for r in joined if r['structural_eval_mask']}):
        rows=[r for r in joined if r['task_id']==task_id and r['structural_eval_mask']]
        assert rows and len({r['prepared_sha256'] for r in rows})==1
        tasks.append(dict(task_id=task_id,pdb=rows[0]['pdb'],author_chain=rows[0]['author_chain'],
                          records=[r['record_id'] for r in rows],prepared_sha256=rows[0]['prepared_sha256']))
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'tasks.json',tasks)
    atomic_json(out/'manifest.json',dict(source=str(source),source_release_sha256=digest(source/'release_manifest.json'),
        tasks=len(tasks),methods=['propka','pkai','pkai_plus','jaxka','pypka'],code_sha256=digest(Path(__file__)),
        settings='Existing frozen adapters; standardized structures/settings, not experimental-condition matching',
        model_fit=False,experimental_training=False))
    print(json.dumps(dict(tasks=len(tasks),retained_records=sum(len(t['records']) for t in tasks)),indent=2),flush=True)

def work(index):
    from pkabench.adapters.base import Adapter
    from pkabench.schema import write_table
    tasks=json.loads((out/'tasks.json').read_text()); task=tasks[index]
    joined={r['record_id']:r for r in json.loads((source/'joined-records.json').read_text())}
    rows=[joined[i] for i in task['records']]
    sites=[]
    for r in rows:
        sites.append(dict(complex_id=task['task_id'],chain=r['chain'],resnum=r['resnum'],icode=r['icode'],group=r['group']))
    # PKAD-R can contain distinct measurements/conditions for the same physical
    # site. Predict that site once; the scorer joins it back to every record.
    unique={tuple(r[k] for k in ('complex_id','chain','resnum','icode','group')):r for r in sites}
    sites=list(unique.values())
    structure=source/'structures'/task['task_id']/'prepared.cif'
    assert digest(structure)==task['prepared_sha256']
    root=out/'tasks'/task['task_id']; root.mkdir(parents=True,exist_ok=True)
    completed=[]
    for method in ('propka','pkai','pkai_plus','jaxka','pypka'):
        dest=root/method
        if (dest/'receipt.json').exists():
            receipt=json.loads((dest/'receipt.json').read_text())
            assert receipt['input_sha256']==task['prepared_sha256'] and digest(dest/'predictions.parquet')==receipt['prediction_sha256']
            completed.append(method); continue
        dest.mkdir(parents=True,exist_ok=False)
        adapter=Adapter(method,sites,runtime,timeout=2700)
        pred=adapter.run({'AB':structure},dest)
        write_table(dest/'predictions.parquet','predictions',pred)
        atomic_json(dest/'receipt.json',dict(method=method,task_id=task['task_id'],pdb=task['pdb'],
            input_sha256=task['prepared_sha256'],prediction_sha256=digest(dest/'predictions.parquet'),
            errors=adapter.errors,timings=adapter.timings,job=os.environ['SLURM_JOB_ID']))
        completed.append(method)
    atomic_json(root/'receipt.json',dict(task_id=task['task_id'],methods=completed,
        outputs={m:digest(root/m/'predictions.parquet') for m in completed},job=os.environ['SLURM_JOB_ID']))
    print(task['task_id'],task['pdb'],completed,flush=True)

if sys.argv[1]=='init': init()
else: work(int(os.environ['SLURM_ARRAY_TASK_ID']))
