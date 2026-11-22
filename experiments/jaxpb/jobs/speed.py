"""jobs/speed.py -- closing tests + timing for the geometry/maps speed-ups (BUILD-SPEED.md, 2026-11-20).

Runs against whatever jaxpb is on PYTHONPATH (the M6 frozen code for the reference digest, a lever snapshot otherwise).
  --mode parity : warm-up, then per state production pb.solve_state with geometry_state / maps_batch wrapped to record a
                  sha256 of every output (probe buffer; ex/ey/ez/ion/bidx/surf of the coarse, focused and model maps of
                  every site batch; ctr) and of the PBOut arrays (esolvM/S/M0, sitpotM, sweeps, maxchange, rho).
                  --passes 2 repeats the pass in the same process (determinism); digests -> <out>/digest_<tag>.json.
                  Stage timings are recorded with the hashing time subtracted.
  --mode timing : warm-up, then untraced production timing of the states (stage marks via the MemTrack hook);
                  --trace N additionally traces the first N states (GPU idle share per stage, prof_geomaps.parse_trace).
  --mode compare: compare digests (<out> = result json, positional digest files: first = reference).
  --mode labels : driver label shards <ref dir> <run dir> bitwise (labels, energies, PB arrays) + driver timing.
  --reuse 1     : lever 3 (parity / timing): each triple's AB first, its GeoReuse passed to its A and B states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
pc = time.perf_counter


def sha(x):
    a = np.ascontiguousarray(np.asarray(x))
    h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()[:24]


def state_timings(summary):
    """Stream old summaries or the written shards' timing records (OPT1 bounded driver summaries)."""
    if "per_state" in summary:
        yield from summary["per_state"]
        return
    for k in summary.get("shards_written", []):
        path = os.path.join(summary["out"], f"labels_{k:05d}.npz")
        with np.load(path) as z:
            rows = json.loads(str(z["timing_json"]))["states"]
        yield from rows


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def triple_order(recs, reuse):
    """lever 3: each complete triple's AB first, then its A and B (other states keep their place)."""
    if not reuse:
        return recs
    grp = {}
    for r in recs:
        cid, _, role = r.state_id.rpartition("_")
        grp.setdefault(cid, {})[role] = r
    out, seen = [], set()
    for r in recs:
        cid, _, role = r.state_id.rpartition("_")
        g = grp.get(cid, {})
        if "AB" in g and ("A" in g or "B" in g) and role in ("AB", "A", "B"):
            if cid not in seen:
                seen.add(cid); out += [g[k] for k in ("AB", "A", "B") if k in g]
        else:
            out.append(r)
    return out


class Reuse:
    """per-state solve_state kwargs for lever 3 (AB stores its GeoReuse; A / B get it as parent)."""

    def __init__(self, on):
        self.on = on; self.cache = {}

    def kw(self, r):
        if not self.on:
            return {}, None
        cid, _, role = r.state_id.rpartition("_")
        if role == "AB":
            g = {}
            return {"geo_out": g}, (cid, g)
        if role in ("A", "B") and cid in self.cache:
            return {"geo_parent": self.cache[cid]}, None
        return {}, None

    def done(self, post):
        if post is not None and post[1].get("reuse") is not None:
            self.cache[post[0]] = post[1]["reuse"]


def all_ids(inputs):
    from jaxpb import records as RC
    return [sid for p in inputs for sid in RC.list_states(p)]


def setup(a):
    import jax
    from jaxpb import api as API                 # noqa: F401  process-wide x64 ON, as in production
    from jaxpb.compile_budget import cache_open
    cinfo = cache_open(a.cc_tar, a.cc_dir) if (a.cc_tar or a.cc_dir) else None
    return jax, cinfo


def mode_parity(a):
    jax, cinfo = setup(a)
    import prof_geomaps as PG
    from jaxpb import geometry as GEO, maps as MP, pb as PB, records as RC
    from jaxpb.compile_budget import CompileCounter
    ids = a.states.split(",") if a.states else all_ids(a.inputs)
    recs = triple_order(PG.load(a.inputs, ids), a.reuse)
    cfg = PG.pbcfg()
    RU = Reuse(a.reuse)
    cur = {}
    hs = {"geo": 0.0, "maps": 0.0}
    g0, m0 = GEO.geometry_state, MP.maps_batch

    def geo_w(*args, **kw):
        g = g0(*args, **kw)
        t = pc()
        cur["geo"] = dict(probes=sha(g.probes), n_probes=int(g.n_probes), c=int(g.c),
                          occ=int(g.flags.get("geo_cell_occ", -1)), K=int(g.flags.get("geo_K", -1)),
                          ntr=int(g.flags.get("knn_truncation_changed", -1)), na=int(g.flags.get("n_arc_pts", -1)))
        hs["geo"] += pc() - t
        return g

    def maps_w(*args, **kw):
        out = m0(*args, **kw)
        t = pc()
        mc, mf, ms, ctr = out
        d = {}
        for gname, m in (("c", mc), ("f", mf), ("s", ms)):
            for k in ("ex", "ey", "ez", "ion", "bidx", "surf"):
                d[gname + k] = sha(m[k])
        ch = np.asarray(ctr)
        d["ctr"] = sha(ch); d["ctr_max"] = ch.max(0).tolist()
        cur.setdefault("maps", []).append(d)
        hs["maps"] += pc() - t
        return out
    GEO.geometry_state = geo_w; MP.maps_batch = maps_w
    res = dict(meta=dict(job=os.environ.get("SLURM_JOB_ID"), code=os.environ.get("CODE", os.getcwd()), jax=jax.__version__,
                         device=str(jax.devices()[0]), cache=cinfo, tag=a.tag, n_states=len(recs)), passes=[])
    with CompileCounter() as cc:
        t = pc(); tw = PG.warm(recs, cfg) if a.warm else 0.0
        res["warm_s"] = tw; res["compiles_after_warm"] = cc.n_compiled
        for p in range(a.passes):
            dig = {}; tim = {}
            for r in recs:
                cur.clear(); hs["geo"] = hs["maps"] = 0.0
                c0 = cc.n_compiled
                tm = {}
                kw, post = RU.kw(r)
                t = pc()
                out = PB.solve_state(r, cfg, timing=tm, **kw)
                wall = pc() - t
                RU.done(post)
                t1 = pc()
                pbd = {k: sha(getattr(out, k)) for k in ("esolvM", "esolvS", "esolvM0", "sitpotM", "sweeps", "maxchange", "rho")}
                hpb = pc() - t1
                dig[r.state_id] = dict(cur, pb=pbd)
                tim[r.state_id] = dict(c=r.size_class, wall=wall - hs["geo"] - hs["maps"], t_geometry=tm["t_geometry"] - hs["geo"],
                                       t_maps=tm["t_maps"] - hs["maps"], t_sor=tm["t_sor"], t_solves=tm["t_solves"],
                                       t_post=tm["t_post"], compiles=cc.n_compiled - c0, hash_s=hs["geo"] + hs["maps"] + hpb,
                                       n_sites=r.n_sites, n_taus=r.n_taus, reuse=list(kw), flags={k: v for k, v in out.flags.items()
                                                                                    if k.startswith(("geo_", "ctr_", "class", "xl"))})
                print(f"[parity p{p}] {r.state_id} c{r.size_class} wall {tim[r.state_id]['wall']:.2f} geo {tim[r.state_id]['t_geometry']:.2f}"
                      f" maps {tim[r.state_id]['t_maps']:.2f} occ {cur['geo']['occ']} comp {tim[r.state_id]['compiles']}", flush=True)
            res["passes"].append(dict(digest=dig, timing=tim, compiles_total=cc.n_compiled, compile_names=list(cc.names)))
            jdump(res, os.path.join(a.out, f"parity_{a.tag}.json"))
        res["compiles"] = cc.summary()
    if a.passes > 1:
        res["pass_equal"] = [res["passes"][k]["digest"] == res["passes"][0]["digest"] for k in range(a.passes)]
    jdump(res, os.path.join(a.out, f"parity_{a.tag}.json"))
    jdump(res["passes"][0]["digest"], os.path.join(a.out, f"digest_{a.tag}.json"))
    if a.cc_tar_out:
        from jaxpb.compile_budget import cache_close
        cache_close(a.cc_tar_out, a.cc_dir)


def mode_timing(a):
    jax, cinfo = setup(a)
    import prof_geomaps as PG
    from jaxpb import pb as PB
    from jaxpb.compile_budget import CompileCounter
    ids = a.states.split(",") if a.states else PG.DEFAULT_STATES
    recs = triple_order(PG.load(a.inputs, ids), a.reuse)
    cfg = PG.pbcfg()
    RU = Reuse(a.reuse)

    def solve(r, annotate=False):
        h = PG.Hook(annotate); tim = {"memtrack": h}
        kw, post = RU.kw(r)
        h.open("geometry"); t = pc()
        PB.solve_state(r, cfg, timing=tim, **kw)
        wall = pc() - t
        RU.done(post)
        return {k: v for k, v in tim.items() if isinstance(v, (int, float))}, h.finish(), wall
    res = dict(meta=dict(job=os.environ.get("SLURM_JOB_ID"), code=os.environ.get("CODE", os.getcwd()), tag=a.tag, reuse=a.reuse,
                         device=str(jax.devices()[0]), states=[r.state_id for r in recs]))
    with CompileCounter() as cc:
        res["warm_s"] = PG.warm(recs, cfg)
        res["compiles_warm"] = cc.n_compiled
        A = {}
        for rep in range(a.repeat):
            for r in recs:
                c0 = cc.n_compiled
                tim, segs, wall = solve(r)
                A.setdefault(r.state_id, []).append(dict(tim=tim, seg=PG.seg_sums(segs), wall=wall, compiles=cc.n_compiled - c0))
                print(f"[timing r{rep}] {r.state_id} wall {wall:.2f} geo {tim['t_geometry']:.2f} maps {tim['t_maps']:.2f}", flush=True)
        res["A"] = A
        res["compiles_total"] = cc.n_compiled
    if a.trace:
        B = {}
        for r in recs[:a.trace]:
            td = os.path.join(a.trace_dir, r.state_id); shutil.rmtree(td, ignore_errors=True)
            jax.profiler.start_trace(td)
            _, segs, wall = solve(r, annotate=True)
            jax.profiler.stop_trace()
            try:
                out, _ = PG.parse_trace(td, {s["k"]: s["name"] for s in segs})
                B[r.state_id] = dict(seg={k: {kk: vv for kk, vv in v.items() if kk in ("wall_ns", "busy_ns", "kern_ns")}
                                          for k, v in out.get("seg", {}).items()}, wall=wall)
            except Exception as e:  # noqa: BLE001
                B[r.state_id] = {"error": repr(e)[:300]}
            shutil.rmtree(td, ignore_errors=True)
        res["trace"] = B
        idle = {}
        for st in ("geometry", "maps", "sor"):
            w = sum(v["seg"].get(st, {}).get("wall_ns", 0) for v in B.values() if "seg" in v)
            b = sum(v["seg"].get(st, {}).get("busy_ns", 0) for v in B.values() if "seg" in v)
            idle[st] = (1 - b / w) if w else None
        res["gpu_idle_share"] = idle
    jdump(res, os.path.join(a.out, f"timing_{a.tag}.json"))


def mode_compare(a):
    ref = json.load(open(a.digests[0]))
    out = {"ref": a.digests[0], "cmp": {}}
    for p in a.digests[1:]:
        d = json.load(open(p))
        keys = sorted(set(ref) & set(d))
        bad = {}
        ngeo = nmap = npb = 0
        for sid in keys:
            r, x = ref[sid], d[sid]
            b = []
            if r["geo"]["probes"] != x["geo"]["probes"] or r["geo"]["n_probes"] != x["geo"]["n_probes"]:
                b.append("probes")
            else:
                ngeo += 1
            if len(r["maps"]) != len(x["maps"]):
                b.append("n_batches")
            else:
                okm = True
                for bi, (mr, mx) in enumerate(zip(r["maps"], x["maps"])):
                    for k in mr:
                        if k in ("ctr", "ctr_max"):
                            continue
                        if mr[k] != mx.get(k):
                            b.append(f"b{bi}:{k}"); okm = False
                nmap += okm
            pbb = [k for k in r["pb"] if r["pb"][k] != x["pb"].get(k)]
            b += ["pb:" + k for k in pbb]; npb += not pbb
            if b:
                bad[sid] = b[:20]
        out["cmp"][p] = dict(n_ref=len(ref), n=len(d), n_common=len(keys), probes_equal=ngeo, maps_equal=nmap, pb_equal=npb,
                             all_equal=(not bad and len(keys) == len(ref) == len(d)), bad=bad)
        print(p, json.dumps({k: v for k, v in out["cmp"][p].items() if k != "bad"}), "bad:", list(bad)[:10], flush=True)
    jdump(out, a.out)


def mode_labels(a):
    """driver label shards of a run vs a reference run (per alias): every label/energy/PB array and sitpot sha bitwise;
    diag/provenance/timing are reported separately (flags and code_sha change by design)."""
    import glob
    from jaxpb import api as API
    def load(d):
        out = {}
        for p in sorted(glob.glob(os.path.join(d, "labels_*.npz"))):
            for r in API.load_shard(p, full_W=False):
                out[r["alias"]] = r
        return out
    R, X = load(a.digests[0]), load(a.digests[1])
    keys = list(API.RAGGED) + ["sitpot_sha", "status"]
    bad = {}
    for k in sorted(set(R) | set(X)):
        if k not in R or k not in X:
            bad[k] = ["missing"]; continue
        b = [f for f in keys if not np.array_equal(np.asarray(R[k][f]), np.asarray(X[k][f]),
                                                    equal_nan=np.asarray(R[k][f]).dtype.kind == "f")]   # pKa NaN = no midpoint
        if b:
            bad[k] = b
    runs = sorted(glob.glob(os.path.join(a.digests[1], "run_*.json")))
    summ = json.load(open(runs[-1])) if runs else {}
    nwarm = 0; wt = dict(t_pb=0.0, t_asm=0.0, t_geometry=0.0, t_maps=0.0)
    for p in state_timings(summ):
        if not p.get("compiles") and not p.get("failed"):
            nwarm += 1
            for k in wt:
                wt[k] += p.get(k, 0.0)
    tb = summ.get("titration_batches", [])
    res = dict(ref=a.digests[0], run=a.digests[1], n_ref=len(R), n=len(X), arrays=keys, all_equal=not bad, bad=bad,
               compiles=summ.get("compiles", {}).get("n_compiled"), wall_s=summ.get("wall_s"),
               warm_states=nwarm, pb_s=wt["t_pb"] / nwarm if nwarm else None,
               asm_s=wt["t_asm"] / nwarm if nwarm else None,
               titr_s_per_state=(sum(b["seconds"] for b in tb if not b["compiles"]) / max(1, sum(b["states"] for b in tb if not b["compiles"]))),
               geo_s=wt["t_geometry"] / nwarm if nwarm else None,
               maps_s=wt["t_maps"] / nwarm if nwarm else None)
    print(json.dumps({k: v for k, v in res.items() if k != "bad"}), "bad:", list(bad)[:10], flush=True)
    jdump(res, a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("parity", "timing", "compare", "labels"), required=True)
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--states", default=None)
    ap.add_argument("--tag", default="x")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--trace", type=int, default=0)
    ap.add_argument("--trace-dir", default=f"/tmp/jpa-speed-trace-{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    ap.add_argument("--cc-tar", default=None)
    ap.add_argument("--cc-dir", default=None)
    ap.add_argument("--cc-tar-out", default=None)
    ap.add_argument("--reuse", type=int, default=0)
    ap.add_argument("digests", nargs="*")
    a = ap.parse_args()
    if a.mode not in ("compare", "labels"):
        os.makedirs(a.out, exist_ok=True)
    {"parity": mode_parity, "timing": mode_timing, "compare": mode_compare, "labels": mode_labels}[a.mode](a)


if __name__ == "__main__":
    main()
