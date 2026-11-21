"""M5 part A closing test: run-to-run determinism of the PB pipeline (GPU).
phases
  stages  one site batch of one state: every stage (geometry, maps, setup1, SOR pool, setup2, post) run twice on identical
          inputs and compared bitwise; spread_scatter (v2 atomics) vs spread (deterministic) repeated R times.
  passes  NP passes in ONE process over the 5 pilot states (logged rho, parity vs v2 on all their tautomers) and n M0 states
          (relfac estimator); bitwise comparison of every pass with pass 1; pass-1 arrays saved to <out>/det_ref.npz.
  fresh   a NEW process: one pass, bitwise comparison with det_ref.npz.
  rfprobe one site batch through post, then the reaction-field pieces (sum(sv), sum(q), E_raw, E_Gauss) of the coarse,
          focused and model grids with jnp.sum (M1) and with the fixed-order tree sum; the first process saves them,
          every later process (fresh compile) compares bitwise -> names the op whose order changes between compiles.
Compared arrays: esolvM, esolvS, esolvM0, sitpotM, sweeps, maxchange, rho.
usage: python tests/test_determinism.py <phase> <out> --goldens pilot.npz --shards s0 s1 s2 [--n-m0 10] [--npasses 3]
       [--cache tar] [--backend xla|pallas] [--tag name]"""
import sys, os, json, time, argparse, hashlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("phase", choices=["stages", "passes", "fresh", "rfprobe"]); ap.add_argument("out")
ap.add_argument("--goldens", required=True); ap.add_argument("--shards", nargs="*", default=[])
ap.add_argument("--n-m0", type=int, default=10); ap.add_argument("--npasses", type=int, default=3)
ap.add_argument("--cache", default=None); ap.add_argument("--backend", default="xla"); ap.add_argument("--tag", default="det")
ap.add_argument("--state", default="d0d5c619bd23b13d_A"); ap.add_argument("--R", type=int, default=5)
ap.add_argument("--pilot-only", action="store_true")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
from jaxpb import records as RC
from jaxpb.compile_budget import CompileCounter, cache_open, cache_close, budget
cinfo = cache_open(A.cache) if A.cache else None
import jax
import jax.numpy as jnp
from jaxpb import pb as PB, maps as MP, sor as SOR, geometry as GEO

KEYS = ("esolvM", "esolvS", "esolvM0", "sitpotM", "sweeps", "maxchange", "rho")
res = {"jax": jax.__version__, "device": str(jax.devices()[0]), "phase": A.phase, "backend": A.backend,
       "xla_flags": os.environ.get("XLA_FLAGS", ""), "cache_open": cinfo}


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def bits_equal(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def tree_equal(x, y):
    lx, ly = jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y)
    return len(lx) == len(ly) and all(bits_equal(a, b) for a, b in zip(lx, ly))


def tree_maxdiff(x, y):
    m = 0.0
    for a, b in zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y)):
        a = np.asarray(a); b = np.asarray(b)
        if a.dtype.kind == "f":
            d = np.abs(a.astype(np.float64) - b.astype(np.float64)); d = d[np.isfinite(d)]
            m = max(m, float(d.max()) if d.size else 0.0)
        elif not np.array_equal(a, b):
            m = max(m, float("inf"))
    return m


# ------------------------------------------------------------------------------------------------ states
def load_sel():
    Z = np.load(A.goldens)
    pilot_ids = [str(x) for x in Z["meta_state_id"]]
    recs = [(r, "pilot") for r in RC.load_states(A.goldens)]
    if not A.pilot_only and A.shards:
        allm0 = [(sh, sid) for sh in A.shards for sid in RC.list_states(sh)]
        pick = allm0[::max(1, len(allm0) // A.n_m0)][:A.n_m0]
        for sh, sid in pick:
            recs += [(r, "m0") for r in RC.load_states(sh, only=[sid])]
    return Z, pilot_ids, recs


def v2(Z, key, i):
    off = Z["v2_" + key + "__off"]; shp = tuple(int(x) for x in Z["v2_" + key + "__shape"][i])
    a = Z["v2_" + key][off[i]:off[i + 1]]
    return a.reshape(shp) if a.size == np.prod(shp) else a


def parity(Z, ids, sid, o):
    i = ids.index(sid)
    d = {}
    for k in ("esolvM", "esolvS", "esolvM0"):
        ref = v2(Z, k, i).reshape(-1); r = np.abs(getattr(o, k).astype(np.float64) - ref) / np.abs(ref)
        d[k] = dict(max_rel=float(r.max()), n_gt_1e6=int((r > 1e-6).sum()), worst=int(np.argmax(r)))
    sp = v2(Z, "sitpotM", i).reshape(o.sitpotM.shape)
    ds = np.abs(o.sitpotM.astype(np.float64) - sp); scale = np.maximum(np.abs(sp).max(1, keepdims=True), 1e-6)
    d["sitpotM"] = dict(max_abs=float(ds.max()), max_rel_to_tau_scale=float((ds / scale).max()))
    it = v2(Z, "it", i).reshape(-1, 3)
    d["sweeps_equal"] = [float(np.mean(o.sweeps[:, k] == it[:, k])) for k in range(3)]
    d["n_taus"] = int(len(o.esolvM))
    return d


def cfg_for(kind):
    return PB.PBConfig(backend=A.backend, specrad="logged" if kind == "pilot" else "estimate")


# ------------------------------------------------------------------------------------------------ stages
def run_stages():
    Z, ids, recs = load_sel()
    rec = [r for r, k in recs if r.state_id == A.state][0]
    cfg = cfg_for("pilot"); sr = rec.ref["specrad"]; specfn = lambda t, k: float(sr[k, t])
    out = {}
    with GEO.x32():
        g1 = GEO.geometry_state(rec.pos, rec.rad, rec.size_class); g2 = GEO.geometry_state(rec.pos, rec.rad, rec.size_class)
        out["geometry_probes"] = bits_equal(g1.probes, g2.probes)
        bl = list(range(min(cfg.B, rec.n_sites)))
        lay, tl = PB._site_layout(rec, bl, cfg.B, specfn)
        sz = cfg.sizes
        sp1 = PB.site_probes(lay["spos"], lay["srad"], sz.PBs)[0]; sp2 = PB.site_probes(lay["spos"], lay["srad"], sz.PBs)[0]
        out["geometry_sites"] = bits_equal(sp1, sp2)
        j = {k: RC.dev(v, np.float32) for k, v in lay.items()}
        pos_m = jax.device_put(g1.pos); rad_m = jax.device_put(g1.rad)
        m1 = MP.maps_batch(pos_m, rad_m, g1.probes, j["ac"], j["spos"], j["srad"], sp1, sz=sz)
        m2 = MP.maps_batch(pos_m, rad_m, g1.probes, j["ac"], j["spos"], j["srad"], sp1, sz=sz)
        out["maps_batch"] = tree_equal(m1, m2)
        mc, mf, ms, _ = m1
        valid = tl >= 0
        s1a = PB.setup1(mc, ms, j["ac"], j["qp"], j["q"], j["sqp"], j["sq"], j["spec"])
        s1b = PB.setup1(mc, ms, j["ac"], j["qp"], j["q"], j["sqp"], j["sq"], j["spec"])
        out["setup1"] = tree_equal(s1a, s1b); out["setup1_maxdiff"] = tree_maxdiff(s1a, s1b)
        items1, cf1, _ = s1a
        BT = cfg.B * RC.T_MAX
        op1 = np.concatenate([np.repeat(np.arange(cfg.B), RC.T_MAX), cfg.B + np.repeat(np.arange(cfg.B), RC.T_MAX)])
        v1 = np.concatenate([valid.reshape(-1), valid.reshape(-1)])
        # run_pool donates nothing of items1 (lanes_init copies), so it can be called twice on the same inputs
        r1 = SOR.run_pool(items1[0], items1[1], items1[2], RC.dev(op1, np.int32), v1, cf1)
        r2 = SOR.run_pool(items1[0], items1[1], items1[2], RC.dev(op1, np.int32), v1, cf1)
        out["sor_pool"] = tree_equal(r1, r2)
        st1 = r1[0]
        s2a = PB.setup2(mf, j["ac"], j["qp"], j["q"], j["spec"], st1); s2b = PB.setup2(mf, j["ac"], j["qp"], j["q"], j["spec"], st1)
        out["setup2"] = tree_equal(s2a, s2b)
        items2, cf2, _ = s2a
        op2 = np.concatenate([np.repeat(np.arange(cfg.B), RC.T_MAX), np.zeros(BT, np.int64)])
        v2_ = np.concatenate([valid.reshape(-1), np.zeros(BT, bool)])
        q1 = SOR.run_pool(items2[0], items2[1], items2[2], RC.dev(op2, np.int32), v2_, cf2)
        Nall = RC.NATOM_ALL_C[rec.size_class]
        pa = np.full((Nall, 3), RC.FAR, np.float32); pa[:rec.n_atoms] = rec.pos
        pos_all = jax.device_put(pa)
        p1 = PB.post(mc, mf, ms, j["ac"], j["qp"], j["q"], j["sqp"], j["sq"], st1, q1[0], pos_all)
        p2 = PB.post(mc, mf, ms, j["ac"], j["qp"], j["q"], j["sqp"], j["sq"], st1, q1[0], pos_all)
        out["post"] = tree_equal(p1, p2)
        # charge spreading alone, R repetitions, coarse (h 1) grid, all (site, tautomer) charge sets of the batch
        for name, fn in (("spread_scatter", PB.spread_scatter), ("spread", PB.spread)):
            f = jax.jit(jax.vmap(jax.vmap(lambda qp_, q_, a: fn(qp_, q_, a, 1.0, PB.N), in_axes=(0, 0, None))))
            ref = None; nd = 0; md = 0.0
            for _ in range(A.R):
                x = np.asarray(f(j["qp"], j["q"], j["ac"]))
                if ref is None:
                    ref = x
                elif not bits_equal(x, ref):
                    nd += 1; md = max(md, float(np.abs(x - ref).max()))
            out[name + "_runs_differing"] = nd; out[name + "_maxdiff"] = md; out["R"] = A.R
            if name == "spread_scatter":
                sref = ref
        out["spread_vs_scatter_maxabs"] = float(np.abs(ref - sref).max())
        # timing of the two spreads in setup-like batch (diagnostic)
        for name, fn in (("spread_scatter", PB.spread_scatter), ("spread", PB.spread)):
            f = jax.jit(jax.vmap(jax.vmap(lambda qp_, q_, a: fn(qp_, q_, a, 0.25, PB.N), in_axes=(0, 0, None))))
            f(j["qp"], j["q"], j["ac"]).block_until_ready()
            t = time.perf_counter()
            for _ in range(5):
                f(j["qp"], j["q"], j["ac"]).block_until_ready()
            out[name + "_ms"] = (time.perf_counter() - t) / 5 * 1e3
    res["stages"] = out
    print(json.dumps(out, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ passes / fresh
def one_pass(recs, Z, ids, tag, store):
    per = {}
    with CompileCounter() as cc:
        for rec, kind in recs:
            tim = {}
            o = PB.solve_state(rec, cfg_for(kind), timing=tim)
            store[rec.state_id] = {k: np.asarray(getattr(o, k)) for k in KEYS}
            d = dict(kind=kind, t_total=tim["t_total"], t_solves=tim["t_solves"], t_maps=tim["t_maps"], t_geometry=tim["t_geometry"],
                     t_post=tim["t_post"], sor_waste=tim["sor_waste"], n_taus=rec.n_taus,
                     digest={k: digest(store[rec.state_id][k]) for k in KEYS})
            if kind == "pilot":
                d["parity"] = parity(Z, ids, rec.state_id, o)
            per[rec.state_id] = d
            print(tag, rec.state_id, kind, round(tim["t_total"], 2), json.dumps(d.get("parity", {}).get("esolvM", {})), flush=True)
    return per, cc.summary()


def compare(a, b):
    out = {}
    for sid in a:
        out[sid] = {k: bits_equal(a[sid][k], b[sid][k]) for k in KEYS}
        if not all(out[sid].values()):
            out[sid]["maxdiff"] = {k: float(np.abs(a[sid][k].astype(np.float64) - b[sid][k].astype(np.float64)).max()) for k in KEYS
                                   if not out[sid][k]}
    return out, all(all(v[k] for k in KEYS) for v in out.values())


def parity_summary(per):
    P = [v["parity"] for v in per.values() if "parity" in v]
    if not P:
        return {}
    return dict(max_rel_energy=max(max(p[k]["max_rel"] for k in ("esolvM", "esolvS", "esolvM0")) for p in P),
                n_energy_gt_1e6=sum(sum(p[k]["n_gt_1e6"] for k in ("esolvM", "esolvS", "esolvM0")) for p in P),
                sitpot_max_abs=max(p["sitpotM"]["max_abs"] for p in P), sitpot_max_rel=max(p["sitpotM"]["max_rel_to_tau_scale"] for p in P),
                sweeps_equal_min=min(min(p["sweeps_equal"]) for p in P), n_taus=sum(p["n_taus"] for p in P),
                gt_1e6=[(s, k, v["parity"][k]["max_rel"], v["parity"][k]["worst"]) for s, v in per.items() if "parity" in v
                        for k in ("esolvM", "esolvS", "esolvM0") if v["parity"][k]["n_gt_1e6"]])


def run_passes(fresh):
    Z, ids, recs = load_sel()
    res["states"] = [r.state_id for r, _ in recs]
    res["n_taus"] = int(sum(r.n_taus for r, _ in recs))
    res["classes"] = sorted({r.size_class for r, _ in recs})
    res["budget"] = budget(res["classes"], backend=PB.resolve_backend(A.backend))
    refp = os.path.join(A.out, f"{A.tag}_ref.npz")
    if fresh:
        R = np.load(refp)
        ref = {sid: {k: R[f"{sid}__{k}"] for k in KEYS} for sid in res["states"]}
        st = {}
        per, cs = one_pass(recs, Z, ids, "fresh", st)
        cmp_, ok = compare(ref, st)
        res.update(fresh=per, compile=cs, vs_ref=cmp_, bitwise_vs_ref=ok, parity=parity_summary(per))
    else:
        passes = []
        for p in range(A.npasses):
            st = {}
            per, cs = one_pass(recs, Z, ids, f"pass{p + 1}", st)
            passes.append(st)
            res[f"pass{p + 1}"] = per; res[f"compile_pass{p + 1}"] = cs
            if p > 0:
                c, ok = compare(passes[0], st)
                res[f"pass{p + 1}_vs_pass1"] = c; res[f"pass{p + 1}_bitwise"] = ok
                print(f"pass{p + 1} bitwise == pass1: {ok}", flush=True)
        np.savez(refp, **{f"{sid}__{k}": v for sid, d in passes[0].items() for k, v in d.items()})
        res["ref_npz"] = refp
        res["bitwise_all_passes"] = all(res.get(f"pass{p + 1}_bitwise", True) for p in range(1, A.npasses))
        res["parity"] = parity_summary(res["pass1"])
        last = res[f"pass{A.npasses}"]
        res["time_last_pass"] = {k: round(sum(v[k] for v in last.values()), 2) for k in ("t_total", "t_solves", "t_maps", "t_geometry", "t_post")}
        res["time_last_pass_pilot"] = {k: round(sum(v[k] for v in last.values() if v["kind"] == "pilot"), 2)
                                       for k in ("t_total", "t_solves", "t_maps", "t_geometry", "t_post")}


def run_rfprobe():
    Z, ids, recs = load_sel()
    rec = [r for r, k in recs if r.state_id == A.state][0]
    cfg = cfg_for("pilot"); sr = rec.ref["specrad"]; specfn = lambda t, k: float(sr[k, t])
    with GEO.x32():
        g1 = GEO.geometry_state(rec.pos, rec.rad, rec.size_class)
        bl = list(range(min(cfg.B, rec.n_sites)))
        lay, tl = PB._site_layout(rec, bl, cfg.B, specfn)
        sz = cfg.sizes
        sp1 = PB.site_probes(lay["spos"], lay["srad"], sz.PBs)[0]
        j = {k: RC.dev(v, np.float32) for k, v in lay.items()}
        mc, mf, ms, _ = MP.maps_batch(jax.device_put(g1.pos), jax.device_put(g1.rad), g1.probes, j["ac"], j["spos"], j["srad"], sp1, sz=sz)
        valid = tl >= 0; BT = cfg.B * RC.T_MAX
        items1, cf1, _ = PB.setup1(mc, ms, j["ac"], j["qp"], j["q"], j["sqp"], j["sq"], j["spec"])
        op1 = np.concatenate([np.repeat(np.arange(cfg.B), RC.T_MAX), cfg.B + np.repeat(np.arange(cfg.B), RC.T_MAX)])
        st1 = SOR.run_pool(items1[0], items1[1], items1[2], RC.dev(op1, np.int32), np.concatenate([valid.reshape(-1)] * 2), cf1)[0]
        items2, cf2, _ = PB.setup2(mf, j["ac"], j["qp"], j["q"], j["spec"], st1)
        op2 = np.concatenate([np.repeat(np.arange(cfg.B), RC.T_MAX), np.zeros(BT, np.int64)])
        st2 = SOR.run_pool(items2[0], items2[1], items2[2], RC.dev(op2, np.int32), np.concatenate([valid.reshape(-1), np.zeros(BT, bool)]), cf2)[0]
        out = {"st1": np.asarray(st1), "st2": np.asarray(st2)}
        for var, fn in (("jnpsum", jnp.sum), ("tsum", PB.tsum)):
            def pieces(bidx, surf, phi, qp_, q_, a, h):
                rho = PB.spread(qp_, q_, a, h, PB.N)
                e0, e1 = PB.rf_energy(phi, bidx, surf, rho, qp_, q_, h, PB.EPSIN, PB.EPSOUT, sum_=fn)
                lap = (jnp.roll(phi, -1, 0) + jnp.roll(phi, 1, 0) + jnp.roll(phi, -1, 1) + jnp.roll(phi, 1, 1)
                       + jnp.roll(phi, -1, 2) + jnp.roll(phi, 1, 2) - 6 * phi)
                sig0 = (-(h / (4 * jnp.pi * PB.EPKT)) * lap).reshape(-1); bb = jnp.maximum(bidx, 0)
                sv = jnp.where(bidx >= 0, sig0[bb], 0.0) - jnp.where(bidx >= 0, rho.reshape(-1)[bb], 0.0) / PB.EPSIN
                return fn(sv), fn(q_), e0, e1
            sh = st1.shape[1:]; B_, T_ = cfg.B, RC.T_MAX
            grids = {"coarse": (mc, st1[:BT], "qp", "q", 1.0), "focused": (mf, st2[:BT], "qp", "q", 0.25),
                     "model": (ms, st1[BT:], "sqp", "sq", 0.25)}
            for gname, (m, phis, qk, qq, h) in grids.items():
                f = jax.jit(jax.vmap(jax.vmap(lambda bi, su, ph, a_, b_, c_: pieces(bi, su, ph, a_, b_, c_, h),
                                              in_axes=(None, None, 0, 0, 0, None))))
                r = f(m["bidx"], m["surf"], phis.reshape((B_, T_) + sh), j[qk], j[qq], j["ac"])
                for nm, v in zip(("sum_sv", "sum_q", "E_raw", "E_gauss"), r):
                    out[f"{var}_{gname}_{nm}"] = np.asarray(v)
        eM, eM0, eS, sit = PB.post(mc, mf, ms, j["ac"], j["qp"], j["q"], j["sqp"], j["sq"], st1, st2,
                                   jax.device_put(np.full((RC.NATOM_ALL_C[rec.size_class], 3), RC.FAR, np.float32)))
        out.update(post_eM=np.asarray(eM), post_eM0=np.asarray(eM0), post_eS=np.asarray(eS))
    refp = os.path.join(A.out, f"{A.tag}_rfprobe_ref.npz")
    if not os.path.exists(refp):
        np.savez(refp, **out); res["rfprobe"] = {"saved": refp}
    else:
        R = np.load(refp)
        res["rfprobe"] = {k: (True if bits_equal(R[k], out[k]) else float(np.abs(R[k].astype(np.float64) - out[k]).max())) for k in out}
    print(json.dumps(res["rfprobe"], indent=1), flush=True)


if A.phase == "stages":
    run_stages()
elif A.phase == "rfprobe":
    run_rfprobe()
else:
    run_passes(A.phase == "fresh")
if A.cache:
    res["cache_close"] = cache_close(A.cache, cinfo["local_dir"])
json.dump(res, open(os.path.join(A.out, f"{A.tag}_{A.phase}.json"), "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in res if k in ("bitwise_all_passes", "bitwise_vs_ref", "parity", "time_last_pass", "stages")},
                 indent=1, default=str))
