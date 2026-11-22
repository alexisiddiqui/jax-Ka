"""Phase-1 exact regression and bounded stage profiling. Run only inside Slurm."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import bench_jaxpb as BJ
import prof_geomaps as PG
import prof2
import speed as SP

CPU_IDS = ['4066edfcdb3ea8e3_AB', '4066edfcdb3ea8e3_A', '4066edfcdb3ea8e3_B',
           '4de4ae106157408f_A', '70c9c3eaa126c736_AB']
FIELDS = ('esolvM', 'esolvS', 'esolvM0', 'sitpotM', 'sweeps', 'maxchange', 'rho')


def digest(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Coulson compute nodes only'
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--backend', required=True)
    p.add_argument('--selection', choices=('all', 'bench', 'xla12', 'cpu'), required=True)
    p.add_argument('--passes', type=int, default=1)
    p.add_argument('--reference')
    p.add_argument('--fresh-reference')
    a = p.parse_args()
    import jax
    from jaxpb import api as API, records as RC
    from jaxpb.compile_budget import CompileCounter
    out = Path(BJ.TP) / 'opt1b'
    out.mkdir(exist_ok=True)
    plan = json.load(open(BJ.TP + '/a40/m6/m6/runA/plan.json'))
    items = [it for sh in plan['shards'] for it in sh]
    ids = {'all': list(dict.fromkeys(it['state_id'] for it in items)), 'bench': BJ.STATES,
           'xla12': BJ.STATES[:12], 'cpu': CPU_IDS}[a.selection]
    recs = SP.triple_order(PG.load(BJ.INPUTS, ids), True)
    assert set(ids) == {r.state_id for r in recs}, 'missing inputs'
    ref = json.load(open(a.reference)) if a.reference else None
    fresh = json.load(open(a.fresh_reference)) if a.fresh_reference else None
    old_energy = prof2.ref_energies([BJ.TP + '/bench/gpu/energies_gpu.npz'] if a.backend == 'pallas' else
                                  sorted(str(x) for x in Path(BJ.TP + '/bench/jaxcpu').glob('energies_cpu*.npz'))
                                  if jax.default_backend() == 'cpu' else [])
    cfg = API.Config(backend=a.backend)
    ct = BJ.CompileTime()
    inst = prof2.Inst(ct, jax.default_backend() == 'cpu')
    inst.install(memana_on=False)
    result = dict(tag=a.tag, env=BJ.env_info(), code_sha=API.code_sha(), selection=a.selection,
                  job=os.environ['SLURM_JOB_ID'], frozen=str(Path.cwd()), states=[r.state_id for r in recs], passes=[], reference=a.reference, fresh_reference=a.fresh_reference)
    captured = {}
    solve = API.solve_pb

    def capture(*args, **kwargs):
        pb, tm = solve(*args, **kwargs)
        captured['pb'] = {k: digest(getattr(pb, k)) for k in FIELDS}
        return pb, tm
    API.solve_pb = capture
    with CompileCounter() as cc:
        for rep in range(a.passes):
            rows = {}; gc = {}; cstart = cc.n_compiled
            for r in recs:
                inst.begin(r); BJ.reset_hwm(); c0 = cc.n_compiled; t0 = time.perf_counter()
                tl = []; e = API.energies([r], cfg, timing=tl, geo_cache=gc)[0]
                wall = time.perf_counter() - t0
                detail = inst.end()
                hashes = dict(captured['pb'])
                hashes.update({k: digest(getattr(e, k)) for k in ('g', 'occ', 'W', 'start', 'nst', 'pairs')})
                hashes['pkint'] = digest(e.diag['pkint'])
                targets = []
                if ref: targets.append(('baseline', ref['passes'][0]['rows'][r.state_id]['hashes']))
                if fresh: targets.append(('fresh', fresh['passes'][0]['rows'][r.state_id]['hashes']))
                if rep: targets.append(('repeat', result['passes'][0]['rows'][r.state_id]['hashes']))
                checks = {name: {k: v == target[k] for k, v in hashes.items()} for name, target in targets}
                stages = {k: max(0., v['t'] - v['comp_s']) for k, v in detail['stages'].items()}
                row = dict(hashes=hashes, checks=checks, legacy_energy=prof2.bitwise(e, old_energy.get(r.state_id)),
                           timing=tl[0], stages=stages, wall=wall, compiles=cc.n_compiled-c0,
                           compiles_total=cc.n_compiled, memory=detail['mem'], gpu=API.gpu_mem(),
                           host_hwm_kb=BJ.status('VmHWM'), n_sites=r.n_sites, size_class=r.size_class)
                rows[r.state_id] = row
                result['current'] = dict(rep=rep, rows=rows)
                BJ.jdump(result, str(out / (a.tag + '.json')))
                print(a.tag, rep, r.state_id, round(wall, 2), 'comp', row['compiles'],
                      'exact', all(all(v.values()) for v in checks.values()), 'legacy', row['legacy_energy'], flush=True)
            result['passes'].append(dict(rows=rows, new_compiles=cc.n_compiled-cstart, compiles_total=cc.n_compiled))
            result.pop('current', None)
            BJ.jdump(result, str(out / (a.tag + '.json')))
        result['compiles'] = cc.summary()
    inst.uninstall()
    result['passed'] = all(all(all(v.values()) for v in row['checks'].values()) and row['legacy_energy'] is not False
                           for pp in result['passes'] for row in pp['rows'].values())
    BJ.jdump(result, str(out / (a.tag + '.json')))
    print('RESULT', a.tag, result['passed'], result['compiles']['n_compiled'], flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
