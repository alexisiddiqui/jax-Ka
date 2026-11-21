"""M2 closing test (Isambard GPU): DelPhi-mode PB with the ported spectral-radius estimator (NO logged rho used) on the
~100 M0 golden states, against DelPhi's own recorded outputs.
Criteria: SOR sweep counts equal to DelPhi's logged loops at >= v1 rates (focused 87-91%, model 83-88%; coarse ~96%);
L2: solvation (esolvM + esolvS, all runs) median rel <= 1e-3, p99 <= 1e-2; site potentials at other sites' atoms
p99 <= 0.02 kT/e. Also reports the estimator vs the logged rho (diagnostic only).
usage: python tests/test_pb_m2.py <out_dir> <shard.npz> [<shard.npz> ...] [--cache tar] [--limit n] [--specrad estimate|logged]"""
import sys, os, json, time, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("shards", nargs="+")
ap.add_argument("--cache", default=None); ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--specrad", default="estimate"); ap.add_argument("--tag", default="m2")
ap.add_argument("--index", type=int, default=None, help="process only the i-th state over the shards (array jobs)")
ap.add_argument("--B", type=int, default=None)
ap.add_argument("--aggregate", action="store_true", help="pool the per-state npz files in <out> instead of running")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
from jaxpb import records as RC
from jaxpb.compile_budget import CompileCounter, cache_open, cache_close, budget
cinfo = cache_open(A.cache) if A.cache else None
import jax
from jaxpb import pb as PB

cfg = PB.PBConfig(specrad=A.specrad) if A.B is None else PB.PBConfig(specrad=A.specrad, B=A.B)
res = {"jax": jax.__version__, "device": str(jax.devices()[0]), "specrad": A.specrad, "states": {}}
pool = {k: [] for k in ("relM", "relS", "sit_other", "eq_c", "eq_f", "eq_s", "rho_f_err", "rho_s_err")}
classes = set()


def l2(rec, o):
    R = rec.ref
    relM = np.abs(o.esolvM - R["esolvM"]) / np.abs(R["esolvM"]); relS = np.abs(o.esolvS - R["esolvS"]) / np.abs(R["esolvS"])
    site_of = {}
    for s in range(rec.n_sites):
        site_of[s] = rec.site_atom_idx(s)
    allsite = np.unique(rec.site_atoms)
    errs = []
    for t in range(rec.n_taus):
        own = site_of[int(rec.tau_site[t])]
        other = np.setdiff1d(allsite, own, assume_unique=True)
        errs.append(np.abs(o.sitpotM[t, other].astype(np.float64) - R["sitpotM"][t, other]))
    e = np.concatenate(errs)
    lg = R["loops"]
    eq = [o.sweeps[:, k] == lg[k] for k in range(3)]
    return relM, relS, e, eq


t0 = time.perf_counter()
n = 0
ALL = [(sh, sid) for sh in A.shards for sid in RC.list_states(sh)]
if A.aggregate:
    import glob
    for f in sorted(glob.glob(os.path.join(A.out, f"{A.tag}_state_*.npz"))):
        Zs = np.load(f)
        for k in pool:
            if k in Zs.files:
                pool[k].append(Zs[k])
        d = json.loads(str(Zs["summary"])); res["states"][str(Zs["state_id"])] = d; classes.add(d["size_class"])
    res["n_expected"] = len(ALL)


class _Null:
    n_compiled = 0
    def __enter__(self): return self
    def __exit__(self, *a): pass


with (CompileCounter() if not A.aggregate else _Null()) as cc:
    todo = [] if A.aggregate else (ALL if A.index is None else [ALL[A.index]])
    for sh, sid in todo:
        for rec in RC.load_states(sh, only=[sid]):
            if A.limit and n >= A.limit:
                break
            tim = {}
            o = PB.solve_state(rec, cfg, timing=tim)
            classes.add(o.flags["size_class"])
            relM, relS, e, eq = l2(rec, o)
            d = dict(n_taus=rec.n_taus, size_class=o.flags["size_class"], role=rec.role,
                     esolvM_med=float(np.median(relM)), esolvM_p99=float(np.percentile(relM, 99)),
                     esolvS_med=float(np.median(relS)), esolvS_p99=float(np.percentile(relS, 99)),
                     sit_other_p99=float(np.percentile(e, 99)), sit_other_max=float(e.max()),
                     sweeps_eq=[float(x.mean()) for x in eq], xl_retries=o.flags["xl_retries"], class_bumps=o.flags["class_bumps"],
                     t_total=tim["t_total"], sor_waste=tim["sor_waste"])
            if "rho_est" in o.flags.get("_arrays", {}):
                pass
            rho = getattr(o, "rho", None)
            if rho is not None:
                rf = np.abs(rho[:, 0] - rec.ref["specrad"][1]); rs = np.abs(rho[:, 1] - rec.ref["specrad"][2])
                d.update(rho_f_maxerr=float(rf.max()), rho_s_maxerr=float(rs.max()))
                pool["rho_f_err"].append(rf); pool["rho_s_err"].append(rs)
            res["states"][rec.state_id] = d
            for k, v in (("relM", relM), ("relS", relS), ("sit_other", e), ("eq_c", eq[0]), ("eq_f", eq[1]), ("eq_s", eq[2])):
                pool[k].append(np.asarray(v, np.float64))
            n += 1
            print(rec.state_id, json.dumps(d), flush=True)
            if A.index is not None:
                d["compiles"] = cc.n_compiled
                np.savez_compressed(os.path.join(A.out, f"{A.tag}_state_{rec.state_id}.npz"), state_id=rec.state_id,
                                    summary=json.dumps(d), relM=relM, relS=relS, sit_other=e.astype(np.float32),
                                    eq_c=eq[0], eq_f=eq[1], eq_s=eq[2],
                                    **({"rho_f_err": pool["rho_f_err"][-1], "rho_s_err": pool["rho_s_err"][-1]} if rho is not None else {}))
                sys.exit(0)
cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in pool.items()}
both = np.concatenate([cat["relM"], cat["relS"]])
n = len(res["states"])
S = dict(n_states=n, n_taus=int(len(cat["relM"])),
         solv_all_median_rel=float(np.median(both)), solv_all_p99_rel=float(np.percentile(both, 99)),
         esolvM_median_rel=float(np.median(cat["relM"])), esolvM_p99_rel=float(np.percentile(cat["relM"], 99)),
         esolvS_median_rel=float(np.median(cat["relS"])), esolvS_p99_rel=float(np.percentile(cat["relS"], 99)),
         sitpot_other_p99=float(np.percentile(cat["sit_other"], 99)), sitpot_other_max=float(cat["sit_other"].max()),
         sweeps_equal_coarse=float(cat["eq_c"].mean()), sweeps_equal_focused=float(cat["eq_f"].mean()),
         sweeps_equal_model=float(cat["eq_s"].mean()),
         per_state_focused_min=min(v["sweeps_eq"][1] for v in res["states"].values()),
         per_state_model_min=min(v["sweeps_eq"][2] for v in res["states"].values()),
         xl_retries=sum(v["xl_retries"] for v in res["states"].values()), class_bumps=sum(v["class_bumps"] for v in res["states"].values()),
         compiles=(max(v.get("compiles", 0) for v in res["states"].values()) if A.aggregate else cc.n_compiled), budget=budget(sorted(classes), xl_variants=1), classes=sorted(classes), wall_s=time.perf_counter() - t0)
if len(cat["rho_f_err"]):
    S.update(rho_focused_maxabs_err=float(cat["rho_f_err"].max()), rho_model_maxabs_err=float(cat["rho_s_err"].max()),
             rho_focused_exact_frac=float(np.mean(cat["rho_f_err"] < 5e-7)), rho_model_exact_frac=float(np.mean(cat["rho_s_err"] < 5e-7)))
S["pass"] = dict(sweeps=S["sweeps_equal_focused"] >= 0.87 and S["sweeps_equal_model"] >= 0.83,
                 l2=S["solv_all_median_rel"] <= 1e-3 and S["solv_all_p99_rel"] <= 1e-2 and S["sitpot_other_p99"] <= 0.02)
res["summary"] = S
if A.cache:
    res["cache_close"] = cache_close(A.cache, cinfo["local_dir"])
json.dump(res, open(os.path.join(A.out, f"{A.tag}.json"), "w"), indent=1, default=float)
print(json.dumps(S, indent=1))
