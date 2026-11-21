"""M1 closing test (Isambard GPU): records + size classes + argument-pure geometry/maps/PB with the XLA lane pool.
  1. pass 1 over the 5 pilot states (logged spectral radii): energies and site potentials vs jaxpb-pilot-v2 outputs
     (criterion: rel <= 1e-6 energies, site potentials <= 1e-6 rel-to-scale / reported abs), sweep counts 100% equal
  2. compile count of pass 1 <= compile_budget.budget(classes touched); pass 2 (same process) compiles 0
  3. overflow: d0d5 B with a too-small Ka (forced overflow) -> automatic XL retry -> identical outputs, flag set
  4. (phase 'cache') a NEW process with the cache tarball compiles 0 on the 5 states
usage: python tests/test_pb_m1.py <goldens.npz> <out_dir> <phase: main|cache> [--states a,b] [--B 16]"""
import sys, os, json, time, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("goldens"); ap.add_argument("out"); ap.add_argument("phase", choices=["main", "cache"])
ap.add_argument("--states", default=None); ap.add_argument("--B", type=int, default=None); ap.add_argument("--cache", default=None)
ap.add_argument("--skip-overflow", action="store_true")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
from jaxpb import records as RC
from jaxpb.compile_budget import CompileCounter, cache_open, cache_close, budget
cinfo = cache_open(A.cache) if A.cache else None
import jax
from jaxpb import pb as PB, maps as MP

Z = np.load(A.goldens)
ids = [str(x) for x in Z["meta_state_id"]]
sel = A.states.split(",") if A.states else ids
recs = {r.state_id: r for r in RC.load_states(A.goldens, only=sel)}


def v2(key, i):
    off = Z["v2_" + key + "__off"]; shp = tuple(int(x) for x in Z["v2_" + key + "__shape"][i])
    a = Z["v2_" + key][off[i]:off[i + 1]]
    return a.reshape(shp) if a.size == np.prod(shp) else a


cfg = PB.PBConfig() if A.B is None else PB.PBConfig(B=A.B)
res = {"jax": jax.__version__, "device": str(jax.devices()[0]), "B": cfg.B, "phase": A.phase, "cache_open": cinfo, "states": {}}


def compare(sid, o):
    i = ids.index(sid)
    d = {}
    for k in ("esolvM", "esolvS", "esolvM0"):
        ref = v2(k, i).reshape(-1); x = getattr(o, k).astype(np.float64)
        r = np.abs(x - ref) / np.abs(ref)
        d[k] = dict(max_rel=float(r.max()), n_gt_1e6=int((r > 1e-6).sum()))
    sp = v2("sitpotM", i).reshape(o.sitpotM.shape)
    ds = np.abs(o.sitpotM.astype(np.float64) - sp)
    scale = np.maximum(np.abs(sp).max(1, keepdims=True), 1e-6)
    d["sitpotM"] = dict(max_abs=float(ds.max()), max_rel_to_tau_scale=float((ds / scale).max()),
                        p99_abs=float(np.percentile(ds, 99)), n_taus_gt_1e6_rel=int(((ds / scale).max(1) > 1e-6).sum()))
    it = v2("it", i).reshape(-1, 3)
    d["sweeps_equal"] = [float(np.mean(o.sweeps[:, k] == it[:, k])) for k in range(3)]
    d["n_taus"] = int(len(o.esolvM))
    return d


def run_pass(tag):
    out = {}
    with CompileCounter() as cc:
        for sid in sel:
            tim = {}
            o = PB.solve_state(recs[sid], cfg, timing=tim)
            out[sid] = o
            res["states"].setdefault(sid, {})[tag] = dict(timing=tim, flags={k: v for k, v in o.flags.items() if k != "xl_reasons"},
                                                          parity=compare(sid, o))
            print(tag, sid, json.dumps(res["states"][sid][tag]["parity"]), round(tim["t_total"], 2), flush=True)
    return out, cc.summary()


classes = sorted({recs[s].size_class for s in sel})
res["classes"] = classes
res["budget"] = budget(classes)
o1, c1 = run_pass("pass1")
res["compile_pass1"] = c1
if A.phase == "main":
    o2, c2 = run_pass("pass2")
    res["compile_pass2"] = c2
    res["pass2_vs_pass1_max_abs"] = {s: float(max(np.abs(o1[s].esolvM - o2[s].esolvM).max(), np.abs(o1[s].sitpotM - o2[s].sitpotM).max()))
                                     for s in sel}
    res["pass2_sweeps_identical"] = all(np.array_equal(o1[s].sweeps, o2[s].sweeps) for s in sel)
    if not A.skip_overflow:
        s0 = sel[0]
        small = MP.Sizes(Ka=4)
        cfg_ov = PB.PBConfig(B=cfg.B, sizes=small, xl_sizes=MP.Sizes())
        with CompileCounter() as cc3:
            o3 = PB.solve_state(recs[s0], cfg_ov)
        res["overflow_test"] = dict(state=s0, forced_Ka=4, xl_retries=o3.flags["xl_retries"],
                                    reasons=o3.flags.get("xl_reasons"), compiles=cc3.n_compiled,
                                    max_abs_vs_normal=float(max(np.abs(o3.esolvM - o1[s0].esolvM).max(), np.abs(o3.esolvS - o1[s0].esolvS).max(),
                                                                np.abs(o3.sitpotM - o1[s0].sitpotM).max())),
                                    sweeps_equal=bool(np.array_equal(o3.sweeps, o1[s0].sweeps)))
        print("overflow", json.dumps(res["overflow_test"]), flush=True)
P = [res["states"][s]["pass1"]["parity"] for s in sel]
res["summary"] = dict(
    max_rel_energy=max(max(p[k]["max_rel"] for k in ("esolvM", "esolvS", "esolvM0")) for p in P),
    n_energy_gt_1e6=sum(sum(p[k]["n_gt_1e6"] for k in ("esolvM", "esolvS", "esolvM0")) for p in P),
    sitpot_max_abs=max(p["sitpotM"]["max_abs"] for p in P), sitpot_max_rel=max(p["sitpotM"]["max_rel_to_tau_scale"] for p in P),
    sweeps_equal_min=min(min(p["sweeps_equal"]) for p in P),
    compiles_pass1=c1["n_compiled"], budget=res["budget"],
    compiles_pass2=res.get("compile_pass2", {}).get("n_compiled"))
if A.cache:
    res["cache_close"] = cache_close(A.cache, cinfo["local_dir"])
json.dump(res, open(os.path.join(A.out, f"m1_{A.phase}.json"), "w"), indent=1, default=str)
print(json.dumps(res["summary"], indent=1))
