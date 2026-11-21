"""jobs/bench_jaxpb.py -- arms B (JAX CPU) and C (A40) of BUILD-BENCH (2026-11-21) + the sanity compare. src/ unchanged.

  --mode src     : print "<sid> <campaign_state_dir>" for STATES[--index] (no jax import; used by bench.sh)
  --mode pb      : PB + assembly (api.energies, production geo reuse, AB before A/B) on --states for --passes passes.
                   Per state: wall, CPU s (getrusage SELF), stage timings (pb.solve_state timing), compile count and
                   compile seconds (all /jax/core/compile/* event durations), host peak RSS (VmHWM reset per state via
                   /proc/self/clear_refs), GPU allocator peak (running; JAX has no reset). Energies -> energies_<tag>.npz
  --mode mc      : titration of saved energies (--energies npz...) at --sweeps for --states (200k subset via
                   --states200). --S prod: api.titrate as shipped (S = 32 per launch; padding lanes are free in Pallas,
                   NOT under the XLA vmap). --S tight: one launch per Mp bucket with S = real states (labels identical by
                   construction: per-state chain keys). --fill 1 (GPU): every bucket cycled to 32 states (aliases sid@k)
                   = production occupancy. A 1k-sweep call on the same shapes compiles first (compile s reported).
  --mode compare : energies CPU vs GPU, pKa CPU vs GPU, jaxpb and native PyPKA vs the campaign teacher.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
pc = time.perf_counter
BASE = "/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench"
TP = BASE + "/tierB/jaxpb-pipeline"
TRIPLES = ["4066edfcdb3ea8e3", "70c9c3eaa126c736", "2db42a83bc0b799f", "29d8e3cd7c0c56e3", "d0d5c619bd23b13d"]
EXTRA = ["c84a01ff31e41ea4_AB", "0dbad2ec4e61686e_AB", "4de4ae106157408f_A", "24d02660df3ac695_A", "d6004cf2ee4008d2_AB"]
STATES = [f"{c}_{s}" for c in TRIPLES for s in ("AB", "A", "B")] + EXTRA          # = results/profile.json states
PILOT_SRC = {"d0d5c619bd23b13d": "d0d5c619bd23b13d/attempt-731014", "24d02660df3ac695": "24d02660df3ac695/attempt-730982",
             "d6004cf2ee4008d2": "d6004cf2ee4008d2/attempt-730974"}
INPUTS = [f"{TP}/m0/shards/m0_goldens_0{k}.npz" for k in range(3)] + [f"{TP}/a40/m6/inputs/pilot5_m6.npz"]


def src_of(sid):
    cid, _, st = sid.rpartition("_")
    if cid in PILOT_SRC:
        return f"{BASE}/campaigns/production-1024-v1/jobs/pypka/{PILOT_SRC[cid]}/{st}"
    for s in json.load(open(f"{TP}/m0/states.json")):
        if s["state_id"] == sid:
            return s["source"]
    raise KeyError(sid)


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def status(key):
    for line in open("/proc/self/status"):
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return -1


def reset_hwm():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def env_info():
    model = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    import jax
    return dict(host=os.uname().nodename, cpu_model=model, affinity=len(os.sched_getaffinity(0)), jax=jax.__version__,
                device=str(jax.devices()[0]), backend=jax.default_backend(), job=os.environ.get("SLURM_JOB_ID"),
                xla_flags=os.environ.get("XLA_FLAGS"), code=os.environ.get("CODE", os.getcwd()))


class CompileTime:
    """sums every /jax/core/compile/* duration (trace, lower, backend compile) -> seconds attributable to compiling."""

    def __init__(self):
        import jax
        self.s = 0.0; self.by = {}
        jax.monitoring.register_event_duration_secs_listener(self._cb)

    def _cb(self, event, duration, **kw):
        if "/jax/core/compile" in event or "compilation_cache" in event:
            self.s += float(duration); self.by[event] = self.by.get(event, 0.0) + float(duration)


# ------------------------------------------------------------------------------------------------ PB + assembly
def mode_pb(a):
    import jax
    from jaxpb import api as API
    from jaxpb.compile_budget import CompileCounter
    import prof_geomaps as PG
    import speed as SP
    ct = CompileTime()
    ids = a.states.split(",") if a.states else STATES
    recs = SP.triple_order(PG.load(INPUTS, ids), True)
    cfg = API.Config()
    res = dict(env=env_info(), cfg=API.cfg_dict(cfg), code_sha=API.code_sha(), pb_backend=API.pb_backend(cfg),
               states=[r.state_id for r in recs], hwm_reset=reset_hwm(), passes=[])
    t_proc0 = pc()
    with CompileCounter() as cc:
        for p in range(a.passes):
            gc = {}; rows = {}; E = {}
            tp0 = pc(); cp0 = cpu_s()
            for r in recs:
                reset_hwm(); c0, s0 = cc.n_compiled, ct.s
                t = pc(); u = cpu_s(); tl = []
                e = API.energies([r], cfg, timing=tl, geo_cache=gc)[0]
                wall = pc() - t
                tm = tl[0]
                rows[r.state_id] = dict(c=r.size_class, n_sites=r.n_sites, n_taus=r.n_taus, n_atoms=r.n_atoms, wall=wall,
                                        cpu=cpu_s() - u, t_pb=tm["t_pb"], t_asm=tm["t_asm"],
                                        stages={k: v for k, v in tm.items() if k.startswith("t_") and k not in ("t_pb", "t_asm")},
                                        compiles=cc.n_compiled - c0, compile_s=ct.s - s0, host_hwm_kb=status("VmHWM"),
                                        threads=status("Threads"), gpu_peak_running=API.gpu_mem()[1],
                                        reuse=e.diag["pb"]["flags"].get("geo_reuse", None))
                E[r.state_id] = e
                print(f"[pb p{p}] {r.state_id} c{r.size_class} wall {wall:.1f} cpu {rows[r.state_id]['cpu']:.1f} pb {tm['t_pb']:.1f}"
                      f" asm {tm['t_asm']:.2f} comp {rows[r.state_id]['compiles']} ({rows[r.state_id]['compile_s']:.1f} s)"
                      f" hwm {rows[r.state_id]['host_hwm_kb'] / 1e6:.2f} GB", flush=True)
            res["passes"].append(dict(rows=rows, wall=pc() - tp0, cpu=cpu_s() - cp0, compiles_total=cc.n_compiled,
                                      compile_s_total=ct.s))
            jdump(res, os.path.join(a.out, f"pb_{a.tag}.json"))
        res["compiles"] = cc.summary(); res["compile_s_by_event"] = ct.by
    res["process_wall"] = pc() - t_proc0
    res["memtrack"] = API.MEMTRACK.table()
    res["gpu_mem_final"] = API.gpu_mem()
    res["ru_maxrss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    jdump(res, os.path.join(a.out, f"pb_{a.tag}.json"))
    arr = {}
    for sid, e in E.items():
        for k in ("g", "occ", "W", "start", "nst", "pairs"):
            arr[f"{sid}/{k}"] = np.asarray(getattr(e, k))
        arr[f"{sid}/pkint"] = np.asarray(e.diag["pkint"], np.float64)
        for k in ("esolvM", "esolvS", "esolvM0"):
            arr[f"{sid}/{k}"] = np.asarray(e.diag["pb"][k])
    np.savez(os.path.join(a.out, f"energies_{a.tag}.npz"), **arr)


def load_energies(paths, ids=None):
    from jaxpb import records as RC
    out = {}
    for p in paths:
        z = np.load(p)
        for sid in sorted({k.split("/")[0] for k in z.files}):
            if ids is None or sid in ids:
                out[sid] = RC.Energies(g=z[f"{sid}/g"], occ=z[f"{sid}/occ"], W=z[f"{sid}/W"], start=z[f"{sid}/start"],
                                       nst=z[f"{sid}/nst"], pairs=z[f"{sid}/pairs"],
                                       diag=dict(pkint=z[f"{sid}/pkint"]))
    return out


# ------------------------------------------------------------------------------------------------ titration
def mode_mc(a):
    import jax
    from jaxpb import api as API, titrate as T
    from jaxpb.compile_budget import CompileCounter
    ct = CompileTime()
    E = load_energies(a.energies)
    cfg = API.Config()
    res = dict(env=env_info(), mc_backend=API.mc_backend(cfg), S_mode=a.S, fill=a.fill, runs=[], hwm_reset=reset_hwm())
    pka = {}

    def bucket(e):
        return T.fit_bucket(len(e.g), len(e.nst), len(e.pairs), T.MP_BUCKETS)[0]

    def launches(ids):
        """list of (Mp, [(sid, alias)]) launches for the S mode."""
        grp = {}
        for sid in ids:
            grp.setdefault(bucket(E[sid]), []).append(sid)
        out = []
        for Mp in sorted(grp):
            m = grp[Mp]
            if a.fill:
                al = [(m[k % len(m)], m[k % len(m)] if k < len(m) else f"{m[k % len(m)]}@{k // len(m)}") for k in range(32)]
                out.append((Mp, al))
            else:
                for b0 in range(0, len(m), 32):
                    out.append((Mp, [(s, s) for s in m[b0:b0 + 32]]))
        return out

    def run(ids, sweeps, label, keep):
        L = launches(ids)
        rows = []
        tot_w = tot_c = 0.0
        for Mp, al in L:
            S = 32 if a.S == "prod" else len(al)
            c = dataclasses.replace(cfg, s_flush=S)
            reset_hwm(); s0 = ct.s
            t = pc(); u = cpu_s()
            R = API.titrate([E[s] for s, _ in al], c, chain_ids=[API.chain_id_of(x) for _, x in al], mc_sweeps=sweeps)
            w = pc() - t; cu = cpu_s() - u
            tot_w += w; tot_c += cu
            n_real = len({x for _, x in al})
            rows.append(dict(Mp=Mp, S=S, n_lanes=len(al), states=sorted({s for s, _ in al}), wall=w, cpu=cu,
                             compile_s=ct.s - s0, s_per_lane=w / len(al), host_hwm_kb=status("VmHWM"),
                             gpu_peak_running=API.gpu_mem()[1], flags=[r.get("flag") for r in R][:3]))
            if keep:
                for (s, x), r in zip(al, R):
                    if s == x:
                        if r.get("pka") is None:
                            res.setdefault("flagged", []).append(dict(state=s, sweeps=sweeps, flag=r.get("flag")))
                            continue
                        pka.setdefault(str(sweeps), {})[s] = np.asarray(r["pka"], np.float64).tolist()
            print(f"[mc {label}] {sweeps} Mp {Mp} S {S} lanes {len(al)} wall {w:.1f} cpu {cu:.1f} comp {rows[-1]['compile_s']:.1f}",
                  flush=True)
        per_state = {}
        for r in rows:
            for s in r["states"]:
                per_state[s] = r["s_per_lane"]
        res["runs"].append(dict(label=label, sweeps=sweeps, n_states=len(ids), launches=rows, wall=tot_w, cpu=tot_c,
                                s_per_state_mean=float(np.mean(list(per_state.values()))) if per_state else None,
                                s_per_state=per_state))
        jdump(res, os.path.join(a.out, f"mc_{a.tag}.json"))

    ids = [s for s in (a.states.split(",") if a.states else STATES) if s in E]
    ids200 = [s for s in (a.states200.split(",") if a.states200 else ids) if s in E]
    with CompileCounter() as cc:
        run(sorted(set(ids) | set(ids200)), 1000, "warm1k", False)
        for sw in [int(x) for x in a.sweeps.split(",")]:
            run(ids200 if sw == 200000 else ids, sw, f"{sw}", not a.fill)
        res["compiles"] = cc.summary()
    res["pka"] = pka
    res["ru_maxrss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    jdump(res, os.path.join(a.out, f"mc_{a.tag}.json"))


# ------------------------------------------------------------------------------------------------ sanity compare
def mode_compare(a):
    sys.path.insert(0, BASE + "/tierB/mf-surrogate-v1/code")
    import common as C
    B = a.out
    out = {}
    teach = {}
    for sid in STATES:
        src = src_of(sid)
        toks = json.load(open(os.path.join(src, "mc-energies.json")))["all_sites"]
        rows = {(r["chain"], r["resnum"], r["group"]): r["pka"] for r in json.load(open(os.path.join(src, "result.json")))["rows"]}
        teach[sid] = (toks, np.array([np.nan if rows[C.token_key(t)] is None else rows[C.token_key(t)] for t in toks]), rows)

    def err(p, t):
        p = np.asarray(p, float); t = np.asarray(t, float)
        both = np.isfinite(p) & np.isfinite(t)
        return np.abs(p[both] - t[both]), int((np.isfinite(p) != np.isfinite(t)).sum())

    def stats(errs, mm):
        x = np.concatenate(errs) if errs else np.zeros(0)
        return dict(n_sites=int(len(x)), mae=float(x.mean()) if len(x) else None, p99=float(np.percentile(x, 99)) if len(x) else None,
                    max=float(x.max()) if len(x) else None, midpoint_mismatches=mm)
    # energies CPU vs GPU
    Ec = load_energies(sorted(glob.glob(os.path.join(B, "jaxcpu", "energies_cpu12_*.npz"))))
    Eg = load_energies(sorted(glob.glob(os.path.join(B, "gpu", "energies_gpu.npz"))))
    ed = {}
    for sid in sorted(set(Ec) & set(Eg)):
        c, g = Ec[sid], Eg[sid]
        ed[sid] = dict(g=float(np.max(np.abs(c.g - g.g))), W=float(np.max(np.abs(c.W - g.W))),
                       pkint=float(np.nanmax(np.abs(c.diag["pkint"] - g.diag["pkint"]))),
                       pairs_equal=bool(np.array_equal(c.pairs, g.pairs)))
    out["energies_cpu12_vs_gpu"] = dict(n_states=len(ed), g_maxabs=max([v["g"] for v in ed.values()], default=None),
                                        W_maxabs=max([v["W"] for v in ed.values()], default=None),
                                        pkint_maxabs=max([v["pkint"] for v in ed.values()], default=None),
                                        pairs_equal=sum(v["pairs_equal"] for v in ed.values()), per_state=ed)
    # pKa
    P = {}
    for tag, path in (("gpu", "gpu/mc_gpu_prod.json"), ("cpu12", "jaxcpu/mc_cpu12.json"), ("cpu1", "jaxcpu/mc_cpu1.json")):
        f = os.path.join(B, path)
        if os.path.exists(f):
            P[tag] = json.load(open(f))["pka"]
    for tag in P:
        for sw in P[tag]:
            errs, mm = [], 0
            for sid, p in P[tag][sw].items():
                e, m = err(p, teach[sid][1]); errs.append(e); mm += m
            out[f"pka_{tag}_{sw}_vs_teacher"] = stats(errs, mm)
            if tag != "gpu" and sw in P.get("gpu", {}):
                errs, mm = [], 0
                for sid, p in P[tag][sw].items():
                    if sid in P["gpu"][sw]:
                        e, m = err(p, P["gpu"][sw][sid]); errs.append(e); mm += m
                out[f"pka_{tag}_vs_gpu_{sw}"] = stats(errs, mm)
                out[f"pka_{tag}_vs_gpu_{sw}"]["states"] = len(errs)
                # same states, GPU vs teacher (so the CPU-vs-teacher number has its GPU counterpart on the subset)
                errs, mm = [], 0
                for sid in P[tag][sw]:
                    if sid in P["gpu"][sw]:
                        e, m = err(P["gpu"][sw][sid], teach[sid][1]); errs.append(e); mm += m
                out[f"pka_gpu_{sw}_vs_teacher_on_{tag}_subset"] = stats(errs, mm)
    # native PyPKA vs campaign
    for n in (1, 12):
        errs, mm, ms = [], 0, 0
        for f in sorted(glob.glob(os.path.join(B, "pypka", f"n{n}", "*.json"))):
            d = json.load(open(f))
            sid = os.path.basename(f)[:-5]
            rows = teach[sid][2]
            p, t = [], []
            for r in d["rows"]:
                k = (r["chain"], r["resnum"], r["group"])
                if k in rows:
                    p.append(np.nan if r["pka"] is None else r["pka"]); t.append(np.nan if rows[k] is None else rows[k])
            e, m = err(p, t); errs.append(e); mm += m; ms += bool(d.get("mc_energies_bitwise_vs_campaign"))
        out[f"pypka_n{n}_vs_campaign"] = dict(stats(errs, mm), states=len(errs), mc_energies_bitwise=ms)
    jdump(out, os.path.join(B, "compare.json"))
    print(json.dumps({k: v for k, v in out.items() if k != "energies_cpu12_vs_gpu"}, indent=1))
    print(json.dumps({k: v for k, v in out["energies_cpu12_vs_gpu"].items() if k != "per_state"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("src", "pb", "mc", "compare"), required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default=".")
    ap.add_argument("--tag", default="x")
    ap.add_argument("--states", default=None)
    ap.add_argument("--states200", default=None)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--energies", nargs="*", default=[])
    ap.add_argument("--sweeps", default="50000,200000")
    ap.add_argument("--S", choices=("prod", "tight"), default="prod")
    ap.add_argument("--fill", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "src":
        sid = STATES[a.index]; print(sid, src_of(sid)); return
    os.makedirs(a.out, exist_ok=True)
    {"pb": mode_pb, "mc": mode_mc, "compare": mode_compare}[a.mode](a)


if __name__ == "__main__":
    main()
