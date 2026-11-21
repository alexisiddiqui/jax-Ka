"""M4 tests (energy track): titrate.py.

  python tests/test_jaxpb_titrate.py local                 quick CPU checks (Mac, any JAX): toys vs brute force, ref vs Pallas interpret,
                                                pKhalf on synthetic curves, padding/bucket invariance.
  python tests/test_jaxpb_titrate.py m4 <out_dir>          closing tests on Isambard (GPU job): the 426 mf-surrogate-v2 states at 50k and
                                                200k sweeps vs v2 g_results (bitwise counts) and the v2 report rows, ref vs
                                                Pallas bitwise, pKhalf vs PyPKA's own pKa, compile counts.
  python tests/test_jaxpb_titrate.py cachecheck <out_dir>  second process with the persistent compile cache: compiles on a re-run.
  python tests/test_jaxpb_titrate.py cmf <out_dir>         Coulson: cluster MF port vs v2 arm C (c.pkl) on the 426 states.

pytest: local() only (the other modes are cluster closing tests; cmf/cpu read JAXPB_MFS_V2 = the mf-surrogate-v2
directory, on Coulson $R/tierB/mf-surrogate-v2).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax

from jaxpb import titrate as T


class CompileCounter:
    """Counts XLA backend compiles via jax.monitoring duration events."""
    def __init__(self):
        self.n = 0; self.events = {}
        jax.monitoring.register_event_duration_secs_listener(self._cb)

    def _cb(self, event, *a, **k):
        self.events[event] = self.events.get(event, 0) + 1
        if 'backend_compile' in event:
            self.n += 1


def load_pack(path):
    z = np.load(path)
    names = [s.split('|') for s in z['states']]
    states, pka = [], []
    for cid, st in names:
        p = f'{cid}|{st}|'
        states.append(T.TitrationState(g=z[p + 'g'], occ=z[p + 'occ'], W=z[p + 'W'], start=z[p + 'start'],
                                       nst=z[p + 'nst'], pairs=z[p + 'pairs']))
        pka.append(z[p + 'pka'])
    return z, names, states, pka


# ------------------------------------------------------------------ local
def local():
    res = {}
    toys = [T.toy([3, 2, 4, 2, 3], 1, 6.0), T.toy([5, 3, 2, 4], 3, 4.0), T.toy([3, 3, 3, 3, 3, 2], 5, 0.8)]
    res['toy_pairs'] = [len(t.pairs) for t in toys]
    # ref vs pallas interpret on one Mp=64 batch, 2 blocks
    keys = T.chain_keys(11, np.arange(len(toys) * T.NPH)).reshape(len(toys), T.NPH)
    B = T.pack_bucket(toys, [0, 1, 2], 256, 2, keys, S=4)
    args = (T.PH_TERM, B['g'], B['occ'], B['W'], B['start'], B['nst'], B['pa'], B['pb'], B['meta'], B['keys'])
    t0 = time.time(); cr = np.asarray(T.mc_ref(*args)); res['ref_s'] = time.time() - t0
    t0 = time.time(); cp = np.asarray(T.mc_pallas(*args, interpret=True)); res['pallas_interp_s'] = time.time() - t0
    res['ref_vs_pallas_interp_maxdiff'] = float(np.abs(cr - cp).max())
    res['padding_state_counts_sum'] = float(cr[3].sum())
    # bucket invariance: same states in Mp=512 bucket with different S
    B2 = T.pack_bucket(toys, [0, 1, 2], 512, 2, keys, S=3)
    c2 = np.asarray(T.mc_ref(T.PH_TERM, B2['g'], B2['occ'], B2['W'], B2['start'], B2['nst'], B2['pa'], B2['pb'],
                             B2['meta'], B2['keys']))
    res['bucket256_vs_512_maxdiff'] = float(np.abs(cr[:3, :, :256] - c2[:, :, :256]).max())
    # toy vs brute force with the driver (ref backend, 30k sweeps)
    out = T.titrate(toys, mc_sweeps=30000, seed=3, backend='ref', S=4)
    for k, t in enumerate(toys):
        ex = T.brute(t)
        res[f'toy{k}_mc_vs_brute_max'] = float(np.abs(out[k]['curves'] - ex).max())
        rc = T.cluster_mf(t)
        res[f'toy{k}_cmf_vs_brute_max'] = float(np.abs(rc['curve'] - ex).max())
    # pKhalf numpy vs jax (f32 default here) on synthetic curves incl. no-crossing and 1.5 cases
    rng = np.random.default_rng(0)
    cur = np.clip(1 / (1 + 10 ** (np.asarray(T.PH)[:, None] - rng.uniform(-4, 18, 50)[None, :])), 0, 1)
    cur[:, 0] = 0.7; cur[:, 1] = np.linspace(2, 0, 73)
    a = T.curves_to_pka(cur)
    b = np.asarray(T.curves_to_pka_jax(cur.astype(np.float32), np.asarray(T.PH, np.float32)))
    res['pkhalf_np_vs_jax_f32_max'] = float(np.nanmax(np.abs(a - b)))
    res['pkhalf_nan_equal'] = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    res['passed'] = bool(res['ref_vs_pallas_interp_maxdiff'] == 0 and res['bucket256_vs_512_maxdiff'] == 0
                         and max(res[f'toy{k}_mc_vs_brute_max'] for k in range(3)) < 0.05
                         and max(res[f'toy{k}_cmf_vs_brute_max'] for k in (0, 1)) < 1e-6
                         and res['pkhalf_nan_equal'] and res['padding_state_counts_sum'] == 0)
    print(json.dumps(res, indent=1))
    return res


def diagnose(states, names, out, sel, eq, arm, sweeps, seed, g2, fn):
    """Mismatching states: rerun each alone with v2's bucket dims (gmc.make_buckets: Mp = pow2(M,64), Np = pow2(max n,16),
    Pp = pow2(max P) over the v2 bucket) to separate 'bucket change (promotion)' from 'arithmetic differences'."""
    import math

    def p2(x, lo=2):
        return max(lo, 1 << int(math.ceil(math.log2(max(1, x)))))
    v2b = {}
    for s in states:
        b = v2b.setdefault(p2(len(s.g), 64), [0, 0]); b[0] = max(b[0], len(s.nst)); b[1] = max(b[1], len(s.pairs))
    mism = []
    for o, k, e in zip(out, sel, eq):
        if e:
            continue
        s = states[k]; Mp2 = p2(len(s.g), 64); Np2, Pp2 = p2(v2b[Mp2][0], 16), p2(v2b[Mp2][1])
        keys = T.chain_keys(seed, np.int64(k) * T.NPH + np.arange(T.NPH)).reshape(1, T.NPH)
        B = T.pack_bucket(states, [k], Mp2, sweeps // T.BLOCK, keys, S=1, NpPp=(Np2, Pp2))
        cnt = np.asarray(fn(T.PH_TERM, B['g'], B['occ'], B['W'], B['start'], B['nst'], B['pa'], B['pb'], B['meta'], B['keys']))
        prot, _ = T.counts_to_prot(s, cnt[0])
        ref = g2[f'{arm}|{names[k][0]}|{names[k][1]}']
        mism.append(dict(state=int(k), name='|'.join(names[k]), Mp=o['Mp'], promoted=bool(o['promoted']),
                         v2_dims=[Mp2, Np2, Pp2], v2_dims_rerun_equal_v2=bool(np.array_equal(prot, ref)),
                         v2_dims_rerun_equal_ours=bool(np.array_equal(prot, o['prot'])),
                         max_count_diff=int(np.abs(o['prot'] - ref).max())))
    return mism


# ------------------------------------------------------------------ Isambard closing tests
def m4(out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cc = CompileCounter()
    R = Path(os.environ['R']) / 'audits' / 'mf-surrogate-v2'
    z, names, states, pka_t = load_pack(R / 'states-v2.npz')
    g2 = np.load(R / 'g_results.npz')
    res = dict(jax=jax.__version__, devices=str(jax.devices()), n_states=len(states), pallas=T.pallas_available())
    # bucket table
    fits = [T.fit_bucket(len(s.g), len(s.nst), len(s.pairs)) for s in states]
    res['buckets'] = {str(Mp): sum(f[0] == Mp for f in fits) for Mp in T.MP_BUCKETS}
    res['promoted'] = int(sum(f[1] for f in fits)); res['no_bucket'] = int(sum(f[0] is None for f in fits))
    touched = sorted({f[0] for f in fits if f[0] is not None})
    gi = np.arange(len(states))

    def score(arm, out, sweeps, seed):
        same = 0; nchk = 0; errs = []; eq = []
        for k, (cid, st) in enumerate(names):
            ref = g2[f'{arm}|{cid}|{st}']
            eq.append(np.array_equal(ref, out[k]['prot']))
            nchk += 1; same += int(eq[-1])
            pk, pt = out[k]['pka'], pka_t[k]
            ok = ~np.isnan(pk) & ~np.isnan(pt)
            errs.append(np.abs(pk[ok] - pt[ok]))
        e = np.concatenate(errs)
        mism = diagnose(states, names, out, list(range(len(states))), eq, arm, sweeps, seed, g2, T.mc_pallas)
        return dict(states_bitwise_equal_v2=same, states=nchk, mismatches=mism, pka_mae_vs_teacher=float(e.mean()), pka_n=len(e),
                    pka_p95=float(np.quantile(e, .95)), pka_max=float(e.max()),
                    methods=sorted({o['method'] for o in out}), flags=sorted({str(o['flag']) for o in out}))

    for arm, sweeps, seed in (('G50k', 50000, 5505), ('G200k', 200000, 20005)):
        c0 = cc.n; t0 = time.time()
        out = T.titrate(states, mc_sweeps=sweeps, seed=seed, chain_ids=gi, backend='pallas')
        wall = time.time() - t0
        c1 = cc.n
        r = score(arm, out, sweeps, seed)
        r.update(compiles=c1 - c0, diag_compiles=cc.n - c1, wall_s=wall, gpu_s_per_state=sum(t['seconds'] for t in T.titrate.last_timing) / len(states),
                 batches=len(T.titrate.last_timing))
        res[arm] = r
        print(arm, json.dumps(r), flush=True)
    # toys vs brute force at 100k sweeps (v2 gmc test criterion: max curve error < 0.02)
    toys = [T.toy([3, 2, 4, 2, 3], 1, 6.0), T.toy([5, 3, 2, 4], 3, 4.0), T.toy([3, 3, 3, 3, 3, 2], 5, 0.8)]
    ot = T.titrate(toys, mc_sweeps=100000, seed=11, backend='pallas')
    res['toys_vs_brute_max'] = [float(np.abs(o['curves'] - T.brute(t)).max()) for o, t in zip(ot, toys)]
    # ref vs pallas bitwise on all states, 2k counted sweeps (3 blocks)
    c0 = cc.n
    t0 = time.time(); op = T.titrate(states, mc_sweeps=2000, seed=77, chain_ids=gi, backend='pallas'); tp = time.time() - t0
    c1 = cc.n
    t0 = time.time(); orf = T.titrate(states, mc_sweeps=2000, seed=77, chain_ids=gi, backend='ref'); tr = time.time() - t0
    c2 = cc.n
    eq = [np.array_equal(a['prot'], b['prot']) for a, b in zip(op, orf)]
    res['ref_vs_pallas_2k'] = dict(states_bitwise_equal=int(sum(eq)), states=len(eq), pallas_compiles=c1 - c0,
                                   ref_compiles=c2 - c1, pallas_s=tp, ref_s=tr,
                                   max_curve_diff=float(max(np.abs(a['curves'] - b['curves']).max() for a, b in zip(op, orf))))
    print('ref_vs_pallas', json.dumps(res['ref_vs_pallas_2k']), flush=True)
    # padded states are free: an all-padding batch at 200k sweeps per bucket
    pad = {}
    for Mp in touched:
        B = T.pack_bucket(states, [], Mp, 200, np.zeros((0, T.NPH), np.int32))
        t0 = time.time()
        jax.block_until_ready(T.mc_pallas(T.PH_TERM, B['g'], B['occ'], B['W'], B['start'], B['nst'], B['pa'], B['pb'],
                                          B['meta'], B['keys']))
        pad[str(Mp)] = time.time() - t0
    res['all_padding_batch_seconds_200k'] = pad
    # pKhalf vs PyPKA on the teacher curves (PyPKA's own avgs/mcsteps and getpK)
    jax.config.update('jax_enable_x64', True)
    same_np = same_jax = tot = 0; maxd = 0.0
    for k, (cid, st) in enumerate(names):
        cur = z[f'{cid}|{st}|teacher']
        a = T.curves_to_pka(cur); b = np.asarray(T.curves_to_pka_jax(cur, np.asarray(T.PH, np.float64)))
        t = pka_t[k]
        tot += len(t)
        same_np += int(np.sum((np.isnan(a) & np.isnan(t)) | (a == t)))
        same_jax += int(np.sum((np.isnan(b) & np.isnan(t)) | (b == t)))
        m = ~np.isnan(a) & ~np.isnan(t)
        if m.any():
            maxd = max(maxd, float(np.abs(a[m] - t[m]).max()))
    res['pkhalf_vs_pypka'] = dict(sites=tot, identical_numpy=same_np, identical_jax_f64=same_jax, max_abs_diff=maxd)
    # the pipeline runs with process-wide x64 (design 6.9): MC counts must not change
    c0 = cc.n
    ox = T.titrate(states, mc_sweeps=2000, seed=77, chain_ids=gi, backend='pallas')
    res['x64_pallas_2k'] = dict(states_bitwise_equal_x64_off=int(sum(np.array_equal(a['prot'], b['prot']) for a, b in zip(op, ox))),
                                compiles=cc.n - c0)
    res['compiles_total'] = cc.n
    res['buckets_touched'] = touched
    res['compile_events'] = {k: v for k, v in cc.events.items() if 'compile' in k}
    v2rep = dict(G50k=dict(pka_mae=0.0063), G200k=dict(pka_mae=0.0040))
    res['v2_report'] = v2rep
    res['passed'] = bool(res['G50k']['states_bitwise_equal_v2'] == len(states)
                         and res['G200k']['states_bitwise_equal_v2'] == len(states)
                         and res['ref_vs_pallas_2k']['states_bitwise_equal'] == len(states)
                         and res['G50k']['compiles'] == len(touched) and res['G200k']['compiles'] == 0
                         and res['pkhalf_vs_pypka']['identical_numpy'] == tot and max(res['toys_vs_brute_max']) < 0.02)
    (out_dir / 'test_m4.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


def cachecheck(out_dir):
    """Run after m4 in the same job with the same jax_compilation_cache_dir: a fresh process should compile nothing."""
    cc = CompileCounter()
    R = Path(os.environ['R']) / 'audits' / 'mf-surrogate-v2'
    z, names, states, pka_t = load_pack(R / 'states-v2.npz')
    out = T.titrate(states, mc_sweeps=2000, seed=77, chain_ids=np.arange(len(states)), backend='pallas')
    res = dict(compiles=cc.n, events={k: v for k, v in cc.events.items() if 'cach' in k or 'compile' in k},
               cache_dir=jax.config.jax_compilation_cache_dir, states=len(out))
    p = Path(out_dir) / 'test_m4_cachecheck.json'
    p.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1), flush=True)


def cmf_check(out_dir):
    """Coulson: cluster_mf port on the pack (W f32) vs v2 arm C (c.pkl, W f64 from mc-energies.json)."""
    import pickle
    from multiprocessing import Pool
    V2 = Path(os.environ['JAXPB_MFS_V2'])
    z, names, states, pka_t = load_pack(V2 / 'states-v2.npz')
    c = pickle.loads((V2 / 'work' / 'c.pkl').read_bytes())
    order = sorted(range(len(states)), key=lambda k: -len(states[k].g))
    with Pool(len(os.sched_getaffinity(0))) as pool:
        rs = dict(zip(order, pool.map(T.cluster_mf, [states[k] for k in order], chunksize=1)))
    d = []; dp = []; same_clusters = 0
    for k, (cid, st) in enumerate(names):
        ref = c[(cid, st, 'C')]
        d.append(float(np.abs(rs[k]['curve'] - ref['curve']).max()))
        a, b = T.curves_to_pka(rs[k]['curve']), T.curves_to_pka(ref['curve'])
        m = ~np.isnan(a) & ~np.isnan(b)
        dp.append(float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0)
        same_clusters += int(rs[k]['n_multi'] == ref['n_multi'] and rs[k]['max_conf'] == ref['max_conf'])
    res = dict(states=len(names), curve_maxdiff_max=max(d), curve_maxdiff_median=float(np.median(d)),
               pka_maxdiff_max=max(dp), same_cluster_structure=same_clusters)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / 'test_m4_cmf.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def cpu_probe(out_dir, arm, buckets):
    """Coulson CPU: XLA 'ref' backend vs v2's GPU Pallas counts (g_results) for the states of the given Mp buckets, plus
    pKhalf vs PyPKA's pKa on all 426 teacher curves. Not the closing test (that is m4 on the GH200)."""
    V2 = Path(os.environ['JAXPB_MFS_V2'])
    sweeps, seed = dict(G5k=(5000, 1105), G20k=(20000, 2205), G50k=(50000, 5505), G200k=(200000, 20005))[arm]
    cc = CompileCounter()
    z, names, states, pka_t = load_pack(V2 / 'states-v2.npz')
    g2 = np.load(V2 / 'work' / 'g_results.npz')
    want = {int(b) for b in buckets.split(',')}
    sel = [k for k, s in enumerate(states) if T.fit_bucket(len(s.g), len(s.nst), len(s.pairs))[0] in want]
    t0 = time.time()
    out = T.titrate([states[k] for k in sel], mc_sweeps=sweeps, seed=seed, chain_ids=np.array(sel), backend='ref')
    wall = time.time() - t0
    eq = [np.array_equal(o['prot'], g2[f'{arm}|{names[k][0]}|{names[k][1]}']) for o, k in zip(out, sel)]
    mism = diagnose(states, names, out, sel, eq, arm, sweeps, seed, g2, T.mc_ref)
    same = tot = 0
    for k, (cid, st) in enumerate(names):
        a, t = T.curves_to_pka(z[f'{cid}|{st}|teacher']), pka_t[k]
        tot += len(t); same += int(np.sum((np.isnan(a) & np.isnan(t)) | (a == t)))
    res = dict(arm=arm, buckets=sorted(want), states=len(sel), states_bitwise_equal_v2=int(sum(eq)), mismatches=mism,
               promoted_in_selection=int(sum(o['promoted'] for o in out)),
               max_curve_diff=max(dif), wall_s=wall, compiles=cc.n, jax=jax.__version__,
               pkhalf_vs_pypka=dict(sites=tot, identical_numpy=same))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f'test_m4_cpu_{arm}_{"-".join(map(str, sorted(want)))}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)



def test_local():
    assert local()['passed']


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'local':
        sys.exit(0 if local()['passed'] else 1)
    assert os.environ.get('SLURM_JOB_ID'), 'compute node only'
    {'m4': m4, 'cachecheck': cachecheck, 'cmf': cmf_check, 'cpu': cpu_probe}[mode](*sys.argv[2:])
