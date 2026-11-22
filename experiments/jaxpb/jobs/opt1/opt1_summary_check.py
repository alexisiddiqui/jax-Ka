"""Check old/new timing-summary readers against a real unit-test label run, on a CPU node."""
import ast
import glob
import json
import os
from pathlib import Path

import numpy as np
from speed import state_timings

assert os.environ.get('SLURM_JOB_ID')
T = Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline')
root = T / 'opt1/unit-api/full'
s = json.load(open(sorted(root.glob('run_*.json'))[-1]))
rows = list(state_timings(s))
old = dict(s, per_state=rows)
assert len(rows) == s['state_timing']['count'] == 12
assert list(state_timings(old)) == rows
# The M6 module is a CLI; extract only the analysis function, avoiding execution of another campaign.
tree = ast.parse(Path('tests/test_m6.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'throughput')
ns = dict(np=np, state_timings=state_timings)
exec(compile(ast.Module(body=[node], type_ignores=[]), '<throughput>', 'exec'), ns)
assert ns['throughput'](old) == ns['throughput'](s)
resumed = dict(s, shards_written=[s['shards_written'][-1]])
assert len(list(state_timings(resumed))) == 2
json.dump(dict(passed=True, states=12, old_new_throughput_equal=True, resume_written_shards_only=True,
               job=os.environ['SLURM_JOB_ID']), open(T/'opt1/summary-reader-tests.json','w'), indent=2)
print('summary readers passed', flush=True)
