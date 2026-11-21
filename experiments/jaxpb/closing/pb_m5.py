"""M5 closing test (GPU): Pallas SOR backend against the XLA reference on the same states, in one process.
Per state both backends solve all three DelPhi problems of every tautomer (pilot states: logged rho; M0 states: relfac
estimator, as production). Criteria (DESIGN M5): sweep counts >= 99.9% equal to the XLA reference (pooled over all
tautomer calls; exact % reported), energies <= 1e-5 (rel) and site potentials <= 1e-5 (rel to the tautomer's potential
scale) vs the XLA reference, L2 vs DelPhi still passing on the M0 states with the Pallas backend (solvation median rel
<= 1e-3, p99 <= 1e-2; site potentials at other sites p99 <= 0.02 kT/e), SOR time >= 2x faster. Timings exclude any
solve during which a compile happened.
usage: python tests/test_pb_m5.py <out> --goldens pilot.npz --shards s0 s1 s2 [--cache tar] [--limit n] [--tag m5]"""
import os, json, time, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--goldens", required=True); ap.add_argument("--shards", nargs="*", default=[])
ap.add_argument("--cache", default=None); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--tag", default="m5")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
from jaxpb import records as RC
from jaxpb.compile_budget import CompileCounter, cache_open, cache_close
cinfo = cache_open(A.cache) if A.cache else None
import jax
from jaxpb import pb as PB

res = {"jax": jax.__version__, "device": str(jax.devices()[0]), "cache_open": cinfo, "states": {}}
todo = [(A.goldens, sid, "pilot") for sid in RC.list_states(A.goldens)]
todo += [(sh, sid, "m0") for sh in A.shards for sid in RC.list_states(sh)]
if A.limit:
    todo = todo[:A.limit]
pool = {k: [] for k in ("eq_c", "eq_f", "eq_s", "erel", "sit_rel", "sit_abs", "relM", "relS", "sit_other", "dl_c", "dl_f", "dl_s")}


def l2(rec, o):
    """test_pb_m2.l2: Pallas outputs vs DelPhi's recorded ones."""
    R = rec.ref
    relM = np.abs(o.esolvM - R["esolvM"]) / np.abs(R["esolvM"]); relS = np.abs(o.esolvS - R["esolvS"]) / np.abs(R["esolvS"])
    allsite = np.unique(rec.site_atoms)
    errs = []
    for t in range(rec.n_taus):
        other = np.setdiff1d(allsite, rec.site_atom_idx(int(rec.tau_site[t])), assume_unique=True)
        errs.append(np.abs(o.sitpotM[t, other].astype(np.float64) - R["sitpotM"][t, other]))
    lg = R["loops"]
    return relM, relS, np.concatenate(errs), [o.sweeps[:, k] == lg[k] for k in range(3)]


tsum = {"xla": {}, "pallas": {}}
for path, sid, kind in todo:
    rec = RC.load_states(path, only=[sid])[0]
    o = {}; tim = {}; ncomp = {}
    for be in ("xla", "pallas"):
        cfg = PB.PBConfig(backend=be, specrad="logged" if kind == "pilot" else "estimate")
        tim[be] = {}
        with CompileCounter() as cc:
            o[be] = PB.solve_state(rec, cfg, timing=tim[be])
        ncomp[be] = cc.n_compiled
    x, p = o["xla"], o["pallas"]
    eq = [p.sweeps[:, k] == x.sweeps[:, k] for k in range(3)]
    erel = np.concatenate([np.abs(getattr(p, k).astype(np.float64) - getattr(x, k)) / np.abs(getattr(x, k).astype(np.float64))
                           for k in ("esolvM", "esolvS", "esolvM0")])
    ds = np.abs(p.sitpotM.astype(np.float64) - x.sitpotM)
    scale = np.maximum(np.abs(x.sitpotM).max(1, keepdims=True), 1e-6)
    d = dict(kind=kind, n_taus=rec.n_taus, size_class=x.flags["size_class"], sweeps_eq=[float(e.mean()) for e in eq],
             n_sweeps_diff=int(sum((~e).sum() for e in eq)), energy_max_rel=float(erel.max()),
             sitpot_max_abs=float(ds.max()), sitpot_max_rel=float((ds / scale).max()), compiles=ncomp,
             timing={be: {k: tim[be][k] for k in ("t_total", "t_sor", "t_solves", "t_maps", "t_geometry", "t_post", "sor_waste")}
                     for be in tim})
    if d["n_sweeps_diff"]:
        dd = []
        for k in range(3):
            for t in np.nonzero(~eq[k])[0][:5]:
                dd.append(dict(call=k, tau=int(t), xla=int(x.sweeps[t, k]), pallas=int(p.sweeps[t, k]),
                               mc_xla=float(x.maxchange[t, k]), mc_pallas=float(p.maxchange[t, k])))
        d["sweep_diffs"] = dd
    for k, v in (("eq_c", eq[0]), ("eq_f", eq[1]), ("eq_s", eq[2]), ("erel", erel), ("sit_rel", (ds / scale).max(1)), ("sit_abs", ds.max(1))):
        pool[k].append(np.asarray(v, np.float64))
    if kind == "m0":
        relM, relS, e, deq = l2(rec, p)
        for k, v in (("relM", relM), ("relS", relS), ("sit_other", e), ("dl_c", deq[0]), ("dl_f", deq[1]), ("dl_s", deq[2])):
            pool[k].append(np.asarray(v, np.float64))
    if ncomp["xla"] == 0 and ncomp["pallas"] == 0:
        for be in tim:
            for k in ("t_total", "t_sor", "t_solves"):
                tsum[be][k] = tsum[be].get(k, 0.0) + tim[be][k]
            tsum[be]["n"] = tsum[be].get("n", 0) + 1
    res["states"][sid] = d
    print(sid, kind, json.dumps({k: d[k] for k in ("sweeps_eq", "energy_max_rel", "sitpot_max_rel", "compiles")}),
          {be: round(tim[be]["t_sor"], 2) for be in tim}, {be: round(tim[be]["t_total"], 2) for be in tim}, flush=True)

cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in pool.items()}
eq_all = np.concatenate([cat["eq_c"], cat["eq_f"], cat["eq_s"]])
S = dict(n_states=len(res["states"]), n_taus=int(len(cat["eq_c"])), n_calls=int(len(eq_all)),
         sweeps_equal_all=float(eq_all.mean()), n_sweeps_diff=int((eq_all == 0).sum()),
         sweeps_equal_coarse=float(cat["eq_c"].mean()), sweeps_equal_focused=float(cat["eq_f"].mean()),
         sweeps_equal_model=float(cat["eq_s"].mean()),
         energy_max_rel=float(cat["erel"].max()), energy_p99_rel=float(np.percentile(cat["erel"], 99)),
         n_energy_gt_1e5=int((cat["erel"] > 1e-5).sum()),
         sitpot_max_rel=float(cat["sit_rel"].max()), sitpot_max_abs=float(cat["sit_abs"].max()),
         n_taus_sitpot_gt_1e5=int((cat["sit_rel"] > 1e-5).sum()), timing_no_compile=tsum)
if len(cat["relM"]):
    both = np.concatenate([cat["relM"], cat["relS"]])
    S["l2_pallas"] = dict(n_states=sum(1 for v in res["states"].values() if v["kind"] == "m0"), n_taus=int(len(cat["relM"])),
                          solv_all_median_rel=float(np.median(both)), solv_all_p99_rel=float(np.percentile(both, 99)),
                          sitpot_other_p99=float(np.percentile(cat["sit_other"], 99)), sitpot_other_max=float(cat["sit_other"].max()),
                          sweeps_equal_delphi=[float(cat[k].mean()) for k in ("dl_c", "dl_f", "dl_s")])
    L = S["l2_pallas"]
    L["pass"] = L["solv_all_median_rel"] <= 1e-3 and L["solv_all_p99_rel"] <= 1e-2 and L["sitpot_other_p99"] <= 0.02
if tsum["xla"].get("n") and tsum["pallas"].get("t_sor"):
    S["sor_speedup"] = tsum["xla"]["t_sor"] / tsum["pallas"]["t_sor"]
    S["total_speedup"] = tsum["xla"]["t_total"] / tsum["pallas"]["t_total"]
S["pass"] = dict(sweeps=S["sweeps_equal_all"] >= 0.999, energies=S["energy_max_rel"] <= 1e-5, sitpot=S["sitpot_max_rel"] <= 1e-5,
                 l2=S.get("l2_pallas", {}).get("pass"), sor_2x=S.get("sor_speedup", 0) >= 2.0)
res["summary"] = S
if A.cache:
    res["cache_close"] = cache_close(A.cache, cinfo["local_dir"])
json.dump(res, open(os.path.join(A.out, f"{A.tag}.json"), "w"), indent=1, default=float)
print(json.dumps(S, indent=1, default=float))
