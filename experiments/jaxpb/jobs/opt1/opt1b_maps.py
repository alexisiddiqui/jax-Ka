"""Bytewise map gate against frozen dense baseline; no PDE or MC execution."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import jax
import numpy as np
import bench_jaxpb as BJ
import prof_geomaps as PG
from jaxpb import maps as MP, pb as PB, records as RC, geometry as GEO

assert os.environ.get('SLURM_JOB_ID')
T = Path(BJ.TP)
cpu = jax.default_backend() == 'cpu'
name = 'jaxpb.opt1b_dense_baseline'
spec = importlib.util.spec_from_file_location(name, T / 'opt1-baseline-20261121/src/jaxpb/maps.py')
baseline = importlib.util.module_from_spec(spec); sys.modules[name] = baseline; spec.loader.exec_module(baseline)
records = PG.load(BJ.INPUTS, BJ.STATES)
chosen = {}
for rec in records: chosen.setdefault(rec.size_class, rec)
# The frozen 104-state corpus covers classes 0..3; exercise class 4 with the same largest real state, padded upward.
for cls in range(len(RC.NA_C)):
    if cls not in chosen: chosen[cls] = chosen[max(chosen)]
rows = []
for cls, rec in sorted(chosen.items()):
    width = 4 if cpu else 16
    sites = list(range(min(width, rec.n_sites)))
    lay, _ = PB._site_layout(rec, sites, width, lambda t, k: .99)
    active = np.arange(width) < len(sites)
    with GEO.x32():
        geo = GEO.geometry_state(rec.pos, rec.rad, cls)
        sprb, *_ = PB.site_probes(lay['spos'], lay['srad'], RC.MP_MODEL)
        args = tuple(jax.device_put(x) for x in (geo.pos, geo.rad, geo.probes, lay['ac'].astype(np.float32),
                     lay['spos'].astype(np.float32), lay['srad'].astype(np.float32), sprb))
        old = jax.tree_util.tree_map(np.asarray, baseline.maps_batch(*args, sz=baseline.Sizes()))
        new = jax.tree_util.tree_map(np.asarray, MP.maps_batch(*args, sz=MP.Sizes(), active=jax.device_put(active)))
    checks = {}; differences = {}
    for grid, left, right in zip(('coarse', 'focused', 'model'), old[:3], new[:3]):
        for key in left:
            aa, bb = np.ascontiguousarray(left[key][active]), np.ascontiguousarray(right[key][active])
            label = grid + '/' + key
            checks[label] = aa.dtype == bb.dtype and aa.tobytes() == bb.tobytes()
            if not checks[label]:
                differences[label] = dict(different=int((aa != bb).sum()), max_abs=float(np.max(np.abs(aa.astype(float)-bb))))
    checks['counters'] = old[3][active].tobytes() == new[3][active].tobytes()
    if not checks['counters']:
        differences['counters'] = dict(old=old[3][active].tolist(), new=new[3][active].tolist())
    rows.append(dict(state=rec.state_id, size_class=cls, original_size_class=rec.size_class, width=width, sites=len(sites), checks=checks, differences=differences))
    result = dict(job=os.environ['SLURM_JOB_ID'], frozen=str(Path.cwd()), baseline=str(T/'opt1-baseline-20261121'),
                  cpu=cpu, rows=rows, passed=len(rows)==len(RC.NA_C) and all(all(r['checks'].values()) for r in rows))
    BJ.jdump(result, str(T / 'opt1b' / ('maps-cpu.json' if cpu else 'maps-gpu.json')))
    print(rows[-1], flush=True)
    del old, new, args, sprb, geo
    jax.clear_caches()
assert result['passed'], 'map bytes differ from frozen baseline'
