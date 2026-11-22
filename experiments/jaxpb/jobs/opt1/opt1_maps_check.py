"""Localize CPU map parity in frozen revisions; no PDE/MC is run."""
import importlib.util
import json
import os
import sys
from pathlib import Path

T = Path('/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline')
sys.path.insert(0, str(T / 'opt1-C7-20261121/src'))
import jax
import numpy as np
from jaxpb import api as API, pb as PB, geometry as GEO, records as RC

assert os.environ.get('SLURM_JOB_ID') and jax.default_backend() == 'cpu'
sid = '4066edfcdb3ea8e3_AB'
r = next(r for k in range(3) for r in RC.load_states(str(T / f'm0/shards/m0_goldens_0{k}.npz'), only=[sid], with_ref=False))
sites = list(range(((r.n_sites - 1) // 16) * 16, r.n_sites))
lay, _ = PB._site_layout(r, sites, 16, lambda t, k: .99)
res = {}; outputs = {}
with GEO.x32():
    geo = GEO.geometry_state(r.pos, r.rad, r.size_class)
    sp, *_ = PB.site_probes(lay['spos'], lay['srad'], RC.MP_MODEL)
    args = tuple(jax.device_put(x) for x in (geo.pos, geo.rad, geo.probes, lay['ac'].astype(np.float32),
                 lay['spos'].astype(np.float32), lay['srad'].astype(np.float32), sp))
    for tag in ('C5', 'C6', 'C7'):
        name = 'jaxpb.maps_' + tag
        spec = importlib.util.spec_from_file_location(name, T / f'opt1-{tag}-20261121/src/jaxpb/maps.py')
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
        outputs[tag] = jax.tree_util.tree_map(np.asarray, m.maps_batch(*args, sz=m.Sizes()))
    for tag in ('C6', 'C7'):
        rows = {}
        for grid, a, b in zip(('coarse','focused','model'), outputs['C5'][:3], outputs[tag][:3]):
            rows[grid] = {}
            for key in a:
                neq = a[key] != b[key]
                row = dict(different=int(neq.sum()))
                if a[key].dtype.kind == 'f':
                    row['max_abs'] = float(np.max(np.abs(a[key]-b[key])))
                    ix = np.argwhere(neq)[:5]
                    row['examples'] = [dict(index=x.tolist(), old=float(a[key][tuple(x)]), new=float(b[key][tuple(x)])) for x in ix]
                rows[grid][key] = row
        res[tag] = rows
result = dict(state=sid, sites=sites, comparisons_vs_C5=res, job=os.environ['SLURM_JOB_ID'])
json.dump(result, open(T / 'opt1/cpu-tail-localization.json', 'w'), indent=2)
print(json.dumps(result, indent=1), flush=True)
