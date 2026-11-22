"""Static slab/tile/group sweep, executed only on a Coulson compute allocation."""
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import jax
import numpy as np
from jaxpb import maps as MP, pb as PB, geometry as GEO, records as RC
import bench_jaxpb as BJ
import prof_geomaps as PG

assert os.environ.get('SLURM_JOB_ID')
T = Path(BJ.TP)
cpu = jax.default_backend() == 'cpu'
source = Path('src/jaxpb/maps.py').read_text()
rec = PG.load(BJ.INPUTS, ['4066edfcdb3ea8e3_AB' if cpu else 'd6004cf2ee4008d2_AB'])[0]
lay, _ = PB._site_layout(rec, list(range(min(16, rec.n_sites))), 16, lambda t, k: .99)
rows = []
with GEO.x32():
    geo = GEO.geometry_state(rec.pos, rec.rad, rec.size_class)
    sprb, *_ = PB.site_probes(lay['spos'], lay['srad'], RC.MP_MODEL)
    args = tuple(jax.device_put(x) for x in (geo.pos, geo.rad, geo.probes, lay['ac'].astype(np.float32),
                 lay['spos'].astype(np.float32), lay['srad'].astype(np.float32), sprb))
    active = jax.device_put(np.arange(16) < min(16, rec.n_sites))
    tail_sites = list(range((rec.n_sites // 16) * 16, rec.n_sites))
    tail_lay, _ = PB._site_layout(rec, tail_sites, 16, lambda t, k: .99)
    tail_sprb, *_ = PB.site_probes(tail_lay['spos'], tail_lay['srad'], RC.MP_MODEL)
    tail_args = args[:3] + tuple(jax.device_put(x) for x in (tail_lay['ac'].astype(np.float32),
                 tail_lay['spos'].astype(np.float32), tail_lay['srad'].astype(np.float32), tail_sprb))
    tail_active = jax.device_put(np.arange(16) < len(tail_sites))
    expected_tail = None
    options = [(slab, mib, 16) for slab in (128, 256) for mib in (8, 16, 32)] if cpu else [
               (slab, 16, group) for slab in (32, 128, 256) for group in (16, 8)]
    expected = None
    for index, (slab, mib, group) in enumerate(options):
        text = source.replace('CSR_SLAB = 128', f'CSR_SLAB = {slab}')
        text = text.replace('QUERY_BYTES = 16 << 20', f'QUERY_BYTES = {mib} << 20')
        text = text.replace('REACH_BYTES = 16 << 20', f'REACH_BYTES = {mib} << 20')
        text = text.replace('else B\n    active', f'else min({group}, B)\n    active')
        name = f'jaxpb.opt1b_tune_{index}'
        mod = types.ModuleType(name); sys.modules[name] = mod
        exec(compile(text, name, 'exec'), mod.__dict__)
        t0 = time.perf_counter()
        out = mod.maps_batch(*args, sz=mod.Sizes(), active=active)
        jax.block_until_ready(out)
        cold = time.perf_counter() - t0
        leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(out)]
        hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in leaves]
        if expected is None: expected = hashes
        times = []
        for repeat in range(2):
            t0 = time.perf_counter()
            jax.block_until_ready(mod.maps_batch(*args, sz=mod.Sizes(), active=active))
            times.append(time.perf_counter() - t0)
        tail_times = []
        tail_exact = None
        if not cpu:
            tail_out = mod.maps_batch(*tail_args, sz=mod.Sizes(), active=tail_active)
            jax.block_until_ready(tail_out)
            tail_hashes = [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in jax.tree_util.tree_leaves(tail_out)]
            if expected_tail is None: expected_tail = tail_hashes
            tail_exact = tail_hashes == expected_tail
            for repeat in range(2):
                t0 = time.perf_counter()
                jax.block_until_ready(mod.maps_batch(*tail_args, sz=mod.Sizes(), active=tail_active))
                tail_times.append(time.perf_counter()-t0)
            del tail_out
        row = dict(tail_seconds=tail_times, tail_exact=tail_exact, tail_sites=len(tail_sites), slab=slab, mib=mib, group=group, cold=cold, seconds=times,
                   bitwise_vs_first=hashes == expected, hashes=hashes,
                   source_sha256=hashlib.sha256(text.encode()).hexdigest(), gpu=BJ.env_info()['backend'],
                   host_hwm_kb=BJ.status('VmHWM'))
        rows.append(row)
        result = dict(job=os.environ['SLURM_JOB_ID'], frozen=str(Path.cwd()), state=rec.state_id,
                      sites=int(np.asarray(active).sum()), cpu=cpu, rows=rows)
        BJ.jdump(result, str(T / 'opt1b' / ('tune-cpu.json' if cpu else 'tune-gpu-large.json')))
        print({k:v for k,v in row.items() if k!='hashes'}, flush=True)
        del out, leaves, mod
        jax.clear_caches()
