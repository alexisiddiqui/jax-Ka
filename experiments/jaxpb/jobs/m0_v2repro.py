"""M0 (a) check: re-run the jaxpb-pilot-v2 runner (read-only code in $S/_submission/jaxpb-pilot-v2) on the 5 pilot
states inside a CompileCounter, compare with the saved v2 outputs; optionally with the persistent cache tarball.
usage: python m0_v2repro.py <v2_code_dir> <v1_ref_dir> <v2_out_dir> <out_dir> <tag> [--cache tar] [--states s1,s2] [--keep]"""
import sys, os, json, runpy, time, argparse
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("v2code"); ap.add_argument("refdir"); ap.add_argument("v2out"); ap.add_argument("out"); ap.add_argument("tag")
ap.add_argument("--cache", default=None); ap.add_argument("--keep", action="store_true", help="keep <out>/<sid>.<tag>.npz"); ap.add_argument("--states", default="d0d5c619bd23b13d_B,d0d5c619bd23b13d_A,24d02660df3ac695_A,d0d5c619bd23b13d_AB,d6004cf2ee4008d2_AB")
A = ap.parse_args()
from jaxpb.compile_budget import CompileCounter, cache_open, cache_close
import jax
# the v2 runner imports its own module named `jaxpb`; hide this package from sys.modules while it runs
for _k in [k for k in sys.modules if k == "jaxpb" or k.startswith("jaxpb.")]:
    sys.modules.pop(_k)
sys.path.insert(0, A.v2code)
res = {"jax": jax.__version__, "device": str(jax.devices()[0]), "states": {}}
cinfo = cache_open(A.cache) if A.cache else None
res["cache_open"] = cinfo
os.makedirs(A.out, exist_ok=True)
tot = CompileCounter(); tot.__enter__()
for sid in A.states.split(","):
    pref = os.path.join(A.out, f"{sid}.{A.tag}")
    sys.argv = ["jaxrun2.py", os.path.join(A.refdir, sid + ".rec.npz"), pref, "--passes", "1"]
    t = time.perf_counter()
    with CompileCounter() as cc:
        runpy.run_path(os.path.join(A.v2code, "jaxrun2.py"), run_name="__main__")
    wall = time.perf_counter() - t
    O = np.load(pref + ".npz"); V = np.load(os.path.join(A.v2out, sid + ".final.npz"))
    d = {}
    for k in ("esolvM", "esolvM0", "esolvS"):
        r = np.abs(O[k] - V[k]) / np.abs(V[k]); d[k + "_max_rel"] = float(np.nanmax(r)); d[k + "_n_gt_1e6"] = int((r > 1e-6).sum())
    d["sitpotM_max_abs"] = float(np.abs(O["sitpotM"] - V["sitpotM"]).max())
    d["sweeps_equal"] = [float(np.mean(O["it"][:, k] == V["it"][:, k])) for k in range(3)]
    d["compile"] = {k: v for k, v in cc.summary().items() if k not in ("names", "hit_names")}
    d["wall_s"] = wall
    res["states"][sid] = d
    print(sid, json.dumps(d), flush=True)
    if not A.keep:
        os.remove(pref + ".npz")
tot.__exit__()
res["compile_total"] = tot.summary()
res["pass"] = all(v["esolvM_max_rel"] <= 2e-6 and v["esolvS_max_rel"] <= 2e-6 and min(v["sweeps_equal"]) == 1.0 for v in res["states"].values())
if A.cache:
    res["cache_close"] = cache_close(A.cache, cinfo["local_dir"])
json.dump(res, open(os.path.join(A.out, f"v2repro.{A.tag}.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "states"}, indent=1)[:3000])
