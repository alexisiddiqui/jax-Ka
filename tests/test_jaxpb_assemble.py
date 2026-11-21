"""M3 tests (energy track): assemble.py vs PyPKA's own saved MC energies.

  python tests/test_jaxpb_assemble.py local                      tiny synthetic state: JAX assembly vs a literal numpy port of the
                                                      PyPKA loops (calcBackEnergy / calcpKint / calcInteractionWith).
  python tests/test_jaxpb_assemble.py m0 <out_dir> <shard.npz> [...]   the ~100 M0 golden states (records shards), same criteria.
  python tests/test_jaxpb_assemble.py m3 <out_dir> <ref_root> <pack.npz>
        <ref_root>/<cid>_<state>/{rec.npz, mc-energies.json} for the 5 pilot states (DelPhi's recorded outputs from
        jaxpb-pilot-v1 pbrec): g, W (and occ, nst, pairs) vs the saved mc-energies.json (criterion |diff| <= 1e-5 kT),
        e_back / pKint vs PyPKA's recorded tau_eback / tau_pkint; f32 (default) and f64 accumulation; compile counts
        (= size classes touched, 0 on a second pass). <pack.npz>: mf-surrogate-v2 states-v2.npz -> select_pairs vs the
        426 saved pair lists.

pytest: local() always; the pKmod table vs the PyPKA sts files when JAXPB_PYPKA_STS points at
<pypka>/G54A7/sts; the M0 golden shards (m0 mode) when JAXPB_GOLDENS is a glob / os.pathsep list of records shards.
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
import pytest

from jaxpb import assemble as A

PILOT = ['d0d5c619bd23b13d_B', 'd0d5c619bd23b13d_A', '24d02660df3ac695_A', 'd0d5c619bd23b13d_AB', 'd6004cf2ee4008d2_AB']
# pKmod = first line of pypka/G54A7/sts/<RES>tau<k>.st (pypka env on Coulson); k = tautomer number + 1
PKMOD_G54A7 = {'ASP': [3.4644, 3.4644, 2.2293, 2.2293], 'CTR': [2.5744, 2.5744, 1.3393, 1.3393],
               'CYS': [8.1929, 8.1929, 8.1929], 'GLU': [3.8744, 3.8744, 2.6393, 2.6393], 'HIS': [7.2629, 6.8949],
               'LYS': [10.9371, 10.9371, 10.9371], 'NTR': [8.4671, 8.4671, 8.4671], 'PJX': [0.98897, 0.98897],
               'SER': [100.0, 100.0, 100.0], 'THR': [100.0, 100.0, 100.0], 'TYR': [9.2890, 9.2890]}
# Coulson: $R/envs/pypka/lib/python3.10/site-packages/pypka/G54A7/sts
STS = Path(os.environ.get('JAXPB_PYPKA_STS', '/nonexistent/pypka/G54A7/sts'))


class CompileCounter:
    def __init__(self):
        self.n = 0
        jax.monitoring.register_event_duration_secs_listener(self._cb)

    def _cb(self, event, *a, **k):
        if 'backend_compile' in event:
            self.n += 1


def check_pkmod_table():
    if not STS.exists():
        return None
    bad = []
    for res, vals in PKMOD_G54A7.items():
        for k, v in enumerate(vals):
            first = float((STS / f'{res}tau{k + 1}.st').read_text().splitlines()[0])
            if first != v:
                bad.append((res, k, first, v))
    return bad


# ------------------------------------------------------------------ pilot adapter (jaxpb-pilot-v1 rec.npz layout)
def pilot_paths(ref_root, s):
    """Coulson layout <ref>/<state>/{rec.npz, mc-energies.json}; Isambard layout <ref>/<state>.rec.npz plus
    <ref>/pilot5-mce.npz or $JAXPB_MCE (key = state, value = the mc-energies.json text). Returns (rec path, json text)."""
    d = ref_root / s
    if (d / 'rec.npz').exists():
        return d / 'rec.npz', (d / 'mc-energies.json').read_text()
    mce = Path(os.environ.get('JAXPB_MCE', ref_root / 'pilot5-mce.npz'))
    return ref_root / f'{s}.rec.npz', str(np.load(mce)[s])


def from_rec(ref_root, s):
    """Pilot state -> (AssemblyInput, saved dict, diagnostics)."""
    rp, mce = pilot_paths(ref_root, s)
    z = np.load(rp); e = json.loads(mce)
    N = int(z['natoms'])
    tmap = {(str(c), int(s), int(nm[-1])): t for t, (c, s, nm) in enumerate(zip(z['tau_chain'], z['tau_site'], z['tau_name']))}
    nst = np.array(e['npossible_states'], np.int32); n = len(nst); M = int(nst.sum())
    look = e['interactions_look']
    start = np.array([look[i][0] for i in range(n)])
    assert all(look[i][:nst[i]] == list(range(start[i], start[i] + nst[i])) for i in range(n))
    sa_off, sa_all, mq_off = z['site_atoms_off'], z['site_atoms'], z['mq_off']
    taus = np.zeros(M, int); pkmod = np.zeros(M); anionic = np.zeros(n, bool)
    site_atoms = np.full((n, A.A_SITE), -1, np.int32); q_taut = np.zeros((M, A.A_SITE), np.float32)
    ref_charge_vs_bg = 0.0; stray = 0
    for i, tok in enumerate(e['all_sites']):
        chain, res, num = tok.rsplit('_', 2)
        ts = [tmap[(chain, int(num), j)] for j in range(nst[i])]
        sa = sa_all[sa_off[ts[0]]:sa_off[ts[0] + 1]]
        for t in ts:
            assert np.array_equal(sa_all[sa_off[t]:sa_off[t + 1]], sa)
        assert len(sa) <= A.A_SITE, len(sa)
        site_atoms[i, :len(sa)] = sa
        anionic[i] = z['tau_type'][ts[0]] == 'a'
        assert z['tau_isref'][ts[-1]] and not z['tau_isref'][ts[:-1]].any()
        for j, t in enumerate(ts):
            m = start[i] + j; taus[m] = t
            dense = np.zeros(N, np.float32)
            idx = z['mq_idx'][mq_off[t]:mq_off[t + 1]]
            dense[idx] = z['mq_val'][mq_off[t]:mq_off[t + 1]]
            stray += int(np.setdiff1d(idx, sa).size)
            q_taut[m, :len(sa)] = dense[sa]
            if j < nst[i] - 1:
                pkmod[m] = PKMOD_G54A7[res][j]
            else:
                ref_charge_vs_bg = max(ref_charge_vs_bg, float(np.abs(dense[sa] - z['base_chg'][sa]).max()))
    inp = A.AssemblyInput(q_bg=z['base_chg'].astype(np.float32), phi=z['sitpotM'][taus].astype(np.float32),
                          esolvM=z['esolvM'][taus], esolvS=z['esolvS'][taus], site_atoms=site_atoms, q_taut=q_taut,
                          anionic=anionic, pkmod=pkmod, nst=nst)
    g = np.concatenate([np.array(e['possible_states_g'][i][:nst[i]], float) for i in range(n)])
    occ = np.concatenate([np.array(e['possible_states_occ'][i][:nst[i]], int) for i in range(n)])
    W = np.array(e['interactions'], float)
    site_of = np.repeat(np.arange(n), nst)
    same = site_of[:, None] == site_of[None, :]
    saved = dict(g=g, occ=occ, W=W, same=same, start=start, nst=nst, eback=z['tau_eback'][taus],
                 pkint=z['tau_pkint'][taus], isref=z['tau_isref'][taus])
    diag = dict(N=N, M=M, n=n, max_site_atoms=int((site_atoms >= 0).sum(1).max()), stray_charges=stray,
                ref_charges_vs_bg_max=ref_charge_vs_bg)
    return inp, saved, diag


def saved_from_mce(mce, tau_isref, eback=None, pkint=None):
    """Saved PyPKA MC energies (mc-energies.json text) -> arrays in MC order."""
    e = json.loads(mce)
    nst = np.array(e['npossible_states'], np.int32); n = len(nst)
    look = e['interactions_look']
    start = np.array([look[i][0] for i in range(n)])
    assert all(look[i][:nst[i]] == list(range(start[i], start[i] + nst[i])) for i in range(n))
    g = np.concatenate([np.array(e['possible_states_g'][i][:nst[i]], float) for i in range(n)])
    occ = np.concatenate([np.array(e['possible_states_occ'][i][:nst[i]], int) for i in range(n)])
    W = np.array(e['interactions'], float)
    site_of = np.repeat(np.arange(n), nst)
    M = len(g)
    return dict(g=g, occ=occ, W=W, same=site_of[:, None] == site_of[None, :], start=start, nst=nst,
                eback=np.zeros(M) if eback is None else eback, pkint=np.zeros(M) if pkint is None else pkint,
                isref=np.asarray(tau_isref), all_sites=e['all_sites'])


def records_check(recs, cc):
    """records.StateRecord list (ref holds esolvM/esolvS/sitpotM/mc_energies) -> assemble.energies -> compare."""
    from jaxpb import records as RC
    pbs = [RC.PBOut(esolvM=r.ref['esolvM'], esolvS=r.ref['esolvS'], esolvM0=r.ref.get('esolvM0'), sitpotM=r.ref['sitpotM'],
                    sweeps=None, maxchange=None) for r in recs]
    c0 = cc.n; t0 = time.time()
    en = A.energies(recs, pbs)
    out = dict(compiles_first_pass=cc.n - c0, seconds_first_pass=time.time() - t0)
    c0 = cc.n; en2 = A.energies(recs, pbs)
    out['compiles_second_pass'] = cc.n - c0
    out['second_pass_identical'] = all(np.array_equal(a.W, b.W) and np.array_equal(a.g, b.g) for a, b in zip(en, en2))
    per = {}
    for r, e in zip(recs, en):
        sv = saved_from_mce(r.ref['mc_energies'], r.tau_isref, r.ref.get('eback'), r.ref.get('pkint'))
        toks = [f'{c}_{n}' for c, n in zip(r.site_chain, r.site_resnum)]
        order_ok = toks == ['_'.join([t.rsplit('_', 2)[0], t.rsplit('_', 2)[2]]) for t in sv['all_sites']]
        d = dict(g=e.g, occ=e.occ, W=e.W, start=e.start, nst=e.nst, pairs=e.pairs, e_back=e.diag['e_back'],
                 pkint=e.diag['pkint'], size_class=e.diag['size_class'], promoted=e.diag['promoted'])
        per[r.state_id] = dict(compare(d, sv), site_order_matches_all_sites=order_ok,
                               occ_matches_record=e.diag['occ_matches_record'], M=int(len(e.g)), N=r.n_atoms,
                               record_size_class=r.size_class)
    out['states'] = per
    out['classes_touched'] = sorted({v['size_class'] for v in per.values()})
    out['worst_g_or_W'] = max(max(v['g_maxabs'], v['W_maxabs']) for v in per.values())
    out['all_exact_fields'] = all(v['occ_equal'] and v['start_nst_equal'] and v['pairs_equal']
                                  and v['site_order_matches_all_sites'] for v in per.values())
    return out


def pilot_records(ref_root):
    """records.from_pilot_rec for the 5 pilot states + pKmod from the G54A7 table (pilot records carry no pKmod)."""
    from jaxpb import records as RC
    recs = []
    for s in PILOT:
        rp, mce = pilot_paths(ref_root, s)
        r = RC.from_pilot_rec(str(rp), state_id=s, mc_energies=mce)
        names = [t.rsplit('_', 2)[1] for t in json.loads(mce)['all_sites']]
        pk = np.full(r.n_taus, np.nan)
        for i, res in enumerate(names):
            taus = r.site_taus(i)
            for j, t in enumerate(taus[:-1]):
                assert int(str(r.tau_name[t])[-1]) == j
                pk[t] = PKMOD_G54A7[res][j]
        r.tau_pkmod = pk
        r.params['cutoff_back'] = -1.0
        recs.append(r)
    return recs


def compare(r, saved):
    from jaxpb.titrate import select_pairs_np
    same = saved['same']; taut = ~saved['isref']
    dW = np.abs(r['W'].astype(np.float64) - saved['W'])[~same]
    pairs_saved = select_pairs_np(saved['W'].astype(np.float32), saved['start'], saved['nst'])
    return dict(g_maxabs=float(np.abs(r['g'].astype(np.float64) - saved['g']).max()),
                W_maxabs=float(dW.max()), W_p99abs=float(np.quantile(dW, .99)),
                W_maxabs_saved=float(np.abs(saved['W'][~same]).max()),
                occ_equal=bool(np.array_equal(r['occ'], saved['occ'])),
                start_nst_equal=bool(np.array_equal(r['start'], saved['start']) and np.array_equal(r['nst'], saved['nst'])),
                pairs_equal=bool(np.array_equal(r['pairs'], pairs_saved)), npairs=int(len(pairs_saved)),
                eback_maxabs=float(np.abs(r['e_back'] - saved['eback']).max()),
                pkint_maxabs=float(np.abs(r['pkint'][taut] - saved['pkint'][taut]).max()),
                size_class=int(r['size_class']), promoted=bool(r['promoted']))


def m3(out_dir, ref_root, pack):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); ref_root = Path(ref_root)
    cc = CompileCounter()
    res = dict(jax=jax.__version__, backend=jax.default_backend(), pkmod_table_mismatch=check_pkmod_table())
    data = {s: from_rec(ref_root, s) for s in PILOT}
    res['inputs'] = {s: d[2] for s, d in data.items()}
    jax.config.update('jax_enable_x64', True)      # pipeline setting (design §6.9): process-wide x64, explicit dtypes
    for tag, acc in (('f64', 'f64'), ('f32', 'f32')):
        c0 = cc.n; t0 = time.time()
        outs = A.assemble([data[s][0] for s in PILOT], acc=acc)
        res[f'compiles_first_pass_{tag}'] = cc.n - c0; res[f'seconds_first_pass_{tag}'] = time.time() - t0
        c0 = cc.n; t0 = time.time()
        outs2 = A.assemble([data[s][0] for s in PILOT], acc=acc)
        res[f'compiles_second_pass_{tag}'] = cc.n - c0; res[f'seconds_second_pass_{tag}'] = time.time() - t0
        res[f'second_pass_identical_{tag}'] = all(np.array_equal(a['W'], b['W']) and np.array_equal(a['g'], b['g'])
                                                  for a, b in zip(outs, outs2))
        res[tag] = {s: compare(o, data[s][1]) for s, o in zip(PILOT, outs)}
        res[f'classes_touched_{tag}'] = sorted({o['size_class'] for o in outs})
    # the same 5 states through the records.py interface (StateRecord + PBOut -> Energies)
    res['records_interface'] = records_check(pilot_records(ref_root), cc)
    # 426 saved pair lists (mf-surrogate-v2 pack) through the jitted select_pairs
    z = np.load(pack); names = list(z['states'])
    ok = 0; c0 = cc.n; classes = set()
    for nm in names:
        p = nm + '|'
        W, st, ns, pairs = z[p + 'W'], z[p + 'start'], z[p + 'nst'], z[p + 'pairs']
        M, n = len(W), len(ns)
        c = next(c for c in range(A.N_CLASSES) if M <= A.class_dims(c)[1] and n <= A.class_dims(c)[2]
                 and len(pairs) <= A.class_dims(c)[3])
        classes.add(c)
        Na, Mc, Nc, Pc, _ = A.class_dims(c)
        Wp = np.zeros((Mc, Mc), np.float32); Wp[:M, :M] = W
        so = np.full(Mc, Nc, np.int32); so[:M] = np.repeat(np.arange(n), ns)
        pr, npr = A.select_pairs(Wp, so, A.couple_threshold(), Nc=Nc, Pc=Pc)
        ok += int(int(npr) == len(pairs) and np.array_equal(np.asarray(pr)[:int(npr)], pairs))
    res['pack_pairs'] = dict(states=len(names), equal=ok, compiles=cc.n - c0, classes=sorted(classes))
    worst = max(max(v['g_maxabs'], v['W_maxabs']) for v in res['f64'].values())
    res['worst_g_or_W'] = worst
    res['worst_g_or_W_f32_diagnostic'] = max(max(v['g_maxabs'], v['W_maxabs']) for v in res['f32'].values())
    res['passed'] = bool(worst <= 1e-5 and all(v['occ_equal'] and v['start_nst_equal'] and v['pairs_equal']
                                               for v in res['f64'].values())
                         and res['compiles_second_pass_f64'] == 0
                         and res['compiles_first_pass_f64'] == len(res['classes_touched_f64'])
                         and res['pack_pairs']['equal'] == len(names)
                         and res['records_interface']['worst_g_or_W'] <= 1e-5 and res['records_interface']['all_exact_fields']
                         and res['records_interface']['compiles_second_pass'] == 0)
    (out_dir / 'test_m3.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


# ------------------------------------------------------------------ local: literal PyPKA-loop port on a toy
def literal(inp):
    """PyPKA loops, f64, straight from tautomer.py / titsite.py / main.py (no vectorisation)."""
    nst = inp.nst; n = len(nst); start = np.concatenate([[0], np.cumsum(nst)[:-1]]); M = int(nst.sum())
    q = inp.q_bg.astype(float); phi = inp.phi.astype(float)
    eback = np.zeros(M)
    for i in range(n):
        sa = set(int(a) for a in inp.site_atoms[i] if a >= 0)
        for j in range(nst[i]):
            m = start[i] + j
            eback[m] = sum(q[a] * phi[m, a] for a in range(len(q)) if a not in sa)
    g = np.zeros(M); occ = np.zeros(M, int)
    for i in range(n):
        ref = start[i] + nst[i] - 1
        for j in range(nst[i] - 1):
            m = start[i] + j
            dM = (inp.esolvM[ref] - inp.esolvM[m]) / A.LOG10; dS = (inp.esolvS[ref] - inp.esolvS[m]) / A.LOG10
            dB = (eback[ref] - eback[m]) / A.LOG10
            if inp.anionic[i]:
                pk = inp.pkmod[m] + (dM - dS + dB); cd = -1
            else:
                pk = inp.pkmod[m] - (dM - dS + dB); cd = 1
            K, T = 5.98435e-6, 298.15
            g[m] = pk * A.LOG10 * K * T * cd / (K * T)
            occ[m] = 1 if inp.anionic[i] else 0
        occ[ref] = 0 if inp.anionic[i] else 1
    W = np.zeros((M, M))
    for i in range(n):
        for k in range(i + 1, n):
            for s1, s2 in ((i, k), (k, i)):
                sg = (-1 if inp.anionic[s1] else 1) * (-1 if inp.anionic[s2] else 1)
                r2 = start[s2] + nst[s2] - 1
                for a in range(nst[s1]):
                    for b in range(nst[s2]):
                        t1, t2 = start[s1] + a, start[s2] + b
                        x = 0.0
                        for c, at in enumerate(inp.site_atoms[s1]):
                            if at >= 0:
                                x += (q[at] - inp.q_taut[t1, c]) * (phi[r2, at] - phi[t2, at])
                        gg = sg * abs(x)
                        if s1 == i:      # first direction: interactions[a][b] was -999999 -> set both
                            W[t1, t2] = gg; W[t2, t1] = gg
                        else:            # second direction: += gg, /= 2 (main.py:524-528)
                            W[t1, t2] = (W[t1, t2] + gg) / 2; W[t2, t1] = W[t1, t2]
    return g, occ, W, eback


def toy_input(seed=0):
    rng = np.random.default_rng(seed)
    nst = np.array([3, 4, 2, 5], np.int32); n = len(nst); M = int(nst.sum()); N = 60
    site_atoms = np.full((n, A.A_SITE), -1, np.int32)
    for i in range(n):
        site_atoms[i, :6] = np.arange(i * 10, i * 10 + 6)
    q = rng.normal(0, 0.4, N).astype(np.float32)
    site_of = np.repeat(np.arange(n), nst)
    qt = np.zeros((M, A.A_SITE), np.float32)
    qt[:, :6] = (q[site_atoms[site_of, :6]] + rng.normal(0, 0.3, (M, 6))).astype(np.float32)
    start = np.concatenate([[0], np.cumsum(nst)[:-1]])
    qt[start + nst - 1, :6] = q[site_atoms[:, :6]]          # reference tautomer = background charges
    return A.AssemblyInput(q_bg=q, phi=rng.normal(0, 3, (M, N)).astype(np.float32),
                           esolvM=rng.normal(-20, 5, M).astype(np.float32), esolvS=rng.normal(-25, 5, M).astype(np.float32),
                           site_atoms=site_atoms, q_taut=qt, anionic=np.array([True, False, True, False]),
                           pkmod=rng.uniform(3, 11, M).astype(np.float32), nst=nst)


def local():
    jax.config.update('jax_enable_x64', True)
    inp = toy_input()
    r = A.assemble([inp])[0]
    g, occ, W, eb = literal(inp)
    res = dict(g_maxabs=float(np.abs(r['g'] - g).max()), W_maxabs=float(np.abs(r['W'] - W).max()),
               eback_maxabs=float(np.abs(r['e_back'] - eb).max()), occ_equal=bool(np.array_equal(r['occ'], occ)),
               W_symmetric=bool(np.array_equal(r['W'], r['W'].T)), size_class=r['size_class'], npairs=int(r['npairs']))
    from jaxpb.titrate import select_pairs_np
    res['pairs_equal'] = bool(np.array_equal(r['pairs'], select_pairs_np(r['W'], r['start'], r['nst'])))
    res['passed'] = bool(res['g_maxabs'] < 1e-5 and res['W_maxabs'] < 1e-5 and res['occ_equal'] and res['pairs_equal'])
    print(json.dumps(res, indent=1))
    return res


def m0_shards(out_dir, *shards):
    """M0 golden shards (records.save_states npz with ref esolvM/esolvS/sitpotM/mc_energies): g, W vs saved."""
    from jaxpb import records as RC
    jax.config.update('jax_enable_x64', True)
    cc = CompileCounter()
    recs = [r for sh in shards for r in RC.load_states(sh)]
    res = dict(jax=jax.__version__, backend=jax.default_backend(), shards=list(shards), n_states=len(recs))
    res.update(records_check(recs, cc))
    res['passed'] = bool(res['worst_g_or_W'] <= 1e-5 and res['all_exact_fields'] and res['compiles_second_pass'] == 0
                         and res['compiles_first_pass'] <= len(res['classes_touched']))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / 'test_m3_m0shards.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'states'}, indent=1), flush=True)


def pack_mce(ref_root, out):
    """Coulson: the 5 pilot mc-energies.json texts -> one npz (for the Isambard layout)."""
    ref_root = Path(ref_root)
    np.savez_compressed(out, **{s: np.array((ref_root / s / 'mc-energies.json').read_text()) for s in PILOT})
    import hashlib
    print(hashlib.sha256(Path(out).read_bytes()).hexdigest(), out)



def test_local():
    assert local()['passed']


def test_pkmod_table():
    if not STS.exists():
        pytest.skip('JAXPB_PYPKA_STS unset or missing')
    assert check_pkmod_table() == []


def test_m0_golden_shards(tmp_path):
    spec = os.environ.get('JAXPB_GOLDENS', '')
    shards = sorted(p for part in spec.split(os.pathsep) if part for p in glob.glob(part))
    if not shards:
        pytest.skip('JAXPB_GOLDENS unset or matches no records shard')
    m0_shards(tmp_path, *shards)
    assert json.loads((tmp_path / 'test_m3_m0shards.json').read_text())['passed']


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'local':
        sys.exit(0 if local()['passed'] else 1)
    assert os.environ.get('SLURM_JOB_ID'), 'compute node only'
    if mode == 'm3':
        m3(*sys.argv[2:5])
    elif mode == 'm0':
        m0_shards(*sys.argv[2:])
    elif mode == 'packmce':
        pack_mce(*sys.argv[2:4])
