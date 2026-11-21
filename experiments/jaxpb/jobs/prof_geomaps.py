"""jobs/prof_geomaps.py -- measurement-only profile of the PB geometry + maps stages (BUILD-PROFILE.md, 2026-11-20).

src/ is imported unchanged. Every finer-grained timing uses COPIES of src pieces as separate jits (block_until_ready
timed); their outputs are compared with the production programs (parity reported) and then discarded.

  --mode main : warm-up; A production timing (untraced, stage marks via the MemTrack hook); B traced pass (jax.profiler,
                per-stage GPU busy / per-module kernel time); C geometry decomposition (f64) + FP32 timing-only loop;
                D maps per batch (production maps_batch) + 4^3/8^3 block culling stats on every batch, batch-0
                decomposition (coarse/focused/model stages) + probe-test fractions; E AB/A/B probe reuse; F HLO stats
                (FP64 element share of geo_vertex/geo_arc, maps_batch kernel time by source line).
  --mode thru : concurrency test (N processes on one GPU, barrier-synchronised timed loop of solve_state).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import inspect
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from jaxpb import api as API                 # process-wide x64 ON, as in production
from jaxpb import records as RC
from jaxpb import geometry as GEO
from jaxpb import maps as MP
from jaxpb import pb as PB
from jaxpb.compile_budget import cache_open, CompileCounter

pc = time.perf_counter
TRIPLES = ["4066edfcdb3ea8e3", "70c9c3eaa126c736", "2db42a83bc0b799f", "29d8e3cd7c0c56e3", "d0d5c619bd23b13d"]
EXTRA = ["c84a01ff31e41ea4_AB", "0dbad2ec4e61686e_AB", "4de4ae106157408f_A", "24d02660df3ac695_A", "d6004cf2ee4008d2_AB"]
DEFAULT_STATES = [f"{c}_{s}" for c in TRIPLES for s in ("AB", "A", "B")] + EXTRA
KEEP_TRACES = ["2db42a83bc0b799f_AB", "2db42a83bc0b799f_B"]
THRU = ["4066edfcdb3ea8e3_AB", "c84a01ff31e41ea4_AB", "70c9c3eaa126c736_B", "2db42a83bc0b799f_A", "29d8e3cd7c0c56e3_A",
        "2db42a83bc0b799f_AB"]


def jsafe(o):
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return jsafe(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def load(inputs, ids):
    recs = {}
    for p in inputs:
        have = sorted(set(RC.list_states(p)) & set(ids))
        if have:
            for r in RC.load_states(p, only=have, with_ref=False):
                recs[r.state_id] = r
    missing = [i for i in ids if i not in recs]
    if missing:
        print("[prof] missing states", missing, flush=True)
    return [recs[i] for i in ids if i in recs]


def blk(x):
    jax.block_until_ready(x)
    return x


class Hook:
    """MemTrack-compatible stage hook: pb._solve_state calls mark() after geometry, maps (per batch), sor, post."""
    NEXT = {"geometry": "maps", "maps": "sor", "sor": "post", "post": "maps"}

    def __init__(self, annotate=False):
        self.annotate = annotate; self.segs = []; self.cur = None; self.k = 0; self.name = None

    def open(self, name):
        self.name = name; self.t0 = pc()
        if self.annotate:
            self.cur = jax.profiler.TraceAnnotation(f"seg{self.k}:{name}"); self.cur.__enter__()

    def close(self):
        if self.cur is not None:
            self.cur.__exit__(None, None, None); self.cur = None
        self.segs.append(dict(k=self.k, name=self.name, t0=self.t0, t1=pc())); self.k += 1

    def mark(self, stage, ctx=None):
        self.close(); self.open(self.NEXT.get(stage, "other"))

    def finish(self):
        self.close()
        if self.segs and self.segs[-1]["name"] == "maps":
            self.segs[-1]["name"] = "tail"
        return self.segs


def seg_sums(segs):
    d = defaultdict(float)
    for s in segs:
        d[s["name"]] += s["t1"] - s["t0"]
    return dict(d)


def pbcfg():
    return API._pbcfg(API.Config())


def solve(rec, cfg, annotate=False, sites=None):
    h = Hook(annotate); tim = {"memtrack": h}
    h.open("geometry")
    t = pc()
    out = PB.solve_state(rec, cfg, timing=tim, sites=sites)
    wall = pc() - t
    segs = h.finish()
    tim = {k: v for k, v in tim.items() if isinstance(v, (int, float))}
    return out, tim, segs, wall


def warm(recs, cfg):
    t = pc(); done = set()
    for r in recs:
        c = r.size_class
        if c in done:
            continue
        solve(r, cfg, sites=list(range(min(RC.B, r.n_sites))))
        done.add(c)
    return pc() - t


# ------------------------------------------------------------------------------------------------ trace parsing
SEG_RE = re.compile(r"^seg(\d+):(\w+)$")
GRID_RE = re.compile(r"grid:(\d+),(\d+),(\d+)")
BLOCK_RE = re.compile(r"block:(\d+),(\d+),(\d+)")
OCC_RE = re.compile(r"occ(?:upancy)?_pct:([\d.]+)")
N_SM = 84   # A40


def _stats(ev):
    try:
        return {str(k): v for k, v in ev.stats}
    except Exception:  # noqa: BLE001
        return {}


def parse_trace(tdir, segnames, want_ops=False, diag=False):
    files = glob.glob(os.path.join(tdir, "**", "*.xplane.pb"), recursive=True)
    if not files:
        return {"error": "no xplane"}, None
    pdata = jax.profiler.ProfileData.from_file(files[0])
    host = {}; K = []; D = {"planes": [], "lines": defaultdict(int), "stat_keys": set(), "sample_events": []}
    for plane in pdata.planes:
        pn = plane.name
        D["planes"].append(pn)
        if pn.startswith("/host"):
            for line in plane.lines:
                for ev in line.events:
                    m = SEG_RE.match(ev.name)
                    if m:
                        host[int(m.group(1))] = (ev.start_ns, ev.start_ns + ev.duration_ns)
        elif pn.startswith("/device:GPU"):
            for line in plane.lines:
                ln = line.name
                D["lines"][ln] += 0
                if "Stream" not in ln:
                    continue
                for ev in line.events:
                    st = _stats(ev)
                    D["lines"][ln] += 1
                    if diag and len(D["sample_events"]) < 12:
                        D["sample_events"].append(dict(line=ln, name=ev.name[:80], stats={k: str(v)[:120] for k, v in st.items()}))
                    D["stat_keys"].update(st.keys())
                    kd = str(st.get("kernel_details", ""))
                    g = GRID_RE.search(kd); b = BLOCK_RE.search(kd); o = OCC_RE.search(kd)
                    nm = ev.name
                    kind = "copy" if ("emcpy" in nm or "emset" in nm or "memcpy_details" in st) else "kernel"
                    K.append((ev.start_ns, ev.start_ns + ev.duration_ns, str(st.get("hlo_module", "")), str(st.get("hlo_op", "")),
                              kind, (int(g.group(1)) * int(g.group(2)) * int(g.group(3))) if g else -1,
                              (int(b.group(1)) * int(b.group(2)) * int(b.group(3))) if b else -1,
                              float(o.group(1)) if o else -1.0, nm))
    D["lines"] = dict(D["lines"]); D["stat_keys"] = sorted(D["stat_keys"])
    if not K or not host:
        return {"error": f"kernels {len(K)} host segs {len(host)}", "diag": D}, None
    K.sort(key=lambda r: r[0])
    st = np.array([k[0] for k in K], np.int64); en = np.array([k[1] for k in K], np.int64)
    isk = np.array([k[4] == "kernel" for k in K])
    # union of busy intervals (kernels + copies; and kernels only)
    def merge(mask):
        s = st[mask]; e = en[mask]
        if len(s) == 0:
            return np.zeros((0, 2), np.int64)
        out = []; cs, ce = s[0], e[0]
        for a, b in zip(s[1:], e[1:]):
            if a <= ce:
                ce = max(ce, b)
            else:
                out.append((cs, ce)); cs, ce = a, b
        out.append((cs, ce))
        return np.array(out, np.int64)
    Mall = merge(np.ones(len(K), bool)); Mk = merge(isk)

    def busy(M, a, b):
        if len(M) == 0:
            return 0
        return int(np.sum(np.clip(np.minimum(M[:, 1], b) - np.maximum(M[:, 0], a), 0, None)))
    keys = sorted(host)
    seg_t0 = np.array([host[k][0] for k in keys], np.int64)
    res = defaultdict(lambda: defaultdict(float))
    mod = defaultdict(lambda: defaultdict(float))
    ops = defaultdict(float)
    span = (host[keys[0]][0], host[keys[-1]][1])
    for k in keys:
        nm = segnames.get(k, "other"); a, b = host[k]
        r = res[nm]
        r["wall_ns"] += b - a; r["busy_ns"] += busy(Mall, a, b); r["kbusy_ns"] += busy(Mk, a, b); r["n_seg"] += 1
    in_span = 0; tot_k = 0
    for (s, e, m, op, kind, grid, blkn, occ, nm) in K:
        d = e - s
        i = int(np.searchsorted(seg_t0, s, side="right")) - 1
        if i < 0 or s > host[keys[i]][1]:
            seg = "outside"
        else:
            seg = segnames.get(keys[i], "other"); in_span += d
        tot_k += d
        r = res[seg]
        if kind == "kernel":
            r["kern_ns"] += d; r["n_kern"] += 1
            if 0 < grid < N_SM:
                r["kern_small_grid_ns"] += d
            if occ >= 0:
                r["kern_occ_wsum"] += occ * d; r["kern_occ_w"] += d
        else:
            r["copy_ns"] += d; r["n_copy"] += 1
        mod[seg][m or ("<copy>" if kind != "kernel" else "<none>")] += d
        if want_ops and m.endswith("maps_batch") and kind == "kernel":
            ops[nm] += d          # kernels run inside CUDA command buffers (hlo_op = command_buffer_N): key on kernel name
    out = dict(seg={k: dict(v) for k, v in res.items()}, module={k: dict(v) for k, v in mod.items()},
               span_ns=span[1] - span[0], busy_span_ns=busy(Mall, *span), kbusy_span_ns=busy(Mk, *span),
               kernel_ns_in_segments=in_span, kernel_ns_total=tot_k, n_events=len(K))
    if diag:
        out["diag"] = D
    return out, (dict(ops) if want_ops else None)


# ------------------------------------------------------------------------------------------------ geometry copies
@partial(jax.jit, static_argnames=("CH", "nvert"))
def d_vgen(s, NV, P, Rj, valid, U, CH=RC.GEO_CHUNK, nvert=RC.NVERT):
    g = s + jnp.arange(CH, dtype=jnp.int64)
    ok = g < NV
    na = P.shape[0]
    a = jnp.minimum(g // nvert, na - 1); vv = g % nvert
    pts = P[a] + Rj[a][:, None] * U[vv]
    return pts, jnp.stack([a, a], 1), ok & valid[a]


@partial(jax.jit, static_argnames=("CH",))
def d_agen(s, NA, Cum, Mm, Cen, Rho, E1, E2, Pi, Pj, CH=RC.GEO_CHUNK):
    g = s + jnp.arange(CH, dtype=jnp.int64)
    ok = g < NA
    p = jnp.minimum(jnp.searchsorted(Cum, g, side="right"), Cum.shape[0] - 1)
    k = g - (Cum[p] - Mm[p])
    th = 2 * np.pi * k / jnp.maximum(Mm[p], 1)
    pts = Cen[p] + Rho[p][:, None] * (jnp.cos(th)[:, None] * E1[p] + jnp.sin(th)[:, None] * E2[p])
    return pts, jnp.stack([Pi[p], Pj[p]], 1), ok


@partial(jax.jit, static_argnames=("K",))
def d_cand(pts, lo, cell, scid, order, K=RC.GEO_K):
    return GEO._cand(pts, lo, cell, scid, order, K)


@partial(jax.jit, static_argnames=("knn",))
def d_burial(pts, own, cand, P, Rj, bound, knn=24):
    return GEO._burial(pts, own, cand, P, Rj, bound, knn)


d_finish = jax.jit(GEO._finish)


@jax.jit
def d_eq(a, b):
    return jnp.all(a == b)


def geo_run(rec, c, prec="f64", decomp=False):
    """copy of geometry.geometry_state with per-op timers (block_until_ready). prec 'f32' = timing-only experiment:
    the same jitted functions traced with float32/int32 inputs inside x32 (results discarded)."""
    from scipy.spatial import cKDTree
    T = defaultdict(float); par = defaultdict(int)
    t = pc(); Pf, Rf, n = GEO.surface_atoms(rec.pos, rec.rad, c); T["host_surface_atoms"] += pc() - t
    N = RC.NA_C[c]; NPmax = RC.NP_C[c]; NPAIR = RC.PAIRS_PER_ATOM * N; CH = RC.GEO_CHUNK; K = RC.GEO_K; nvert = RC.NVERT
    t = pc()
    pos = Pf[:n].astype(np.float64); R = Rf[:n].astype(np.float64) + RC.RADPRB
    Rmax = float(R.max()); bound = Rmax + 1e-3
    tree = cKDTree(pos); T["host_kdtree_build"] += pc() - t
    t = pc(); pairs = tree.query_pairs(2 * Rmax, output_type="ndarray"); T["host_query_pairs"] += pc() - t
    t = pc()
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(pos[j] - pos[i], axis=1)
    ok = (d < R[i] + R[j]) & (d > np.abs(R[i] - R[j])) & (d > 1e-6)
    i, j, d = i[ok], j[ok], d[ok]
    if len(i) > NPAIR:
        raise OverflowError(f"pairs {len(i)} > {NPAIR}")
    pi = np.zeros(NPAIR, np.int64); pj = np.zeros(NPAIR, np.int64); pd = np.ones(NPAIR); pv = np.zeros(NPAIR, bool)
    pi[:len(i)] = i; pj[:len(i)] = j; pd[:len(i)] = d; pv[:len(i)] = True
    cell = bound + 1e-6
    lo = pos.min(0) - cell
    Pd = np.full((N, 3), RC.FAR, np.float64); Pd[:n] = pos
    Rd = np.zeros(N, np.float64); Rd[:n] = R
    valid = np.zeros(N, bool); valid[:n] = True
    T["host_pair_filter_pad"] += pc() - t
    ctx = GEO.x64() if prec == "f64" else GEO.x32()
    fd = np.float64 if prec == "f64" else np.float32
    idt = np.int64 if prec == "f64" else np.int32
    with ctx:
        t = pc()
        P_ = RC.dev(Pd, fd); R_ = RC.dev(Rd, fd); V_ = jax.device_put(valid)
        lo_ = RC.dev(lo, fd); cell_ = RC.dev(cell, fd); b_ = RC.dev(bound, fd)
        U = RC.dev(GEO.fib_sphere(nvert), fd)
        Pi_ = RC.dev(pi, idt); Pj_ = RC.dev(pj, idt); pd_ = RC.dev(pd, fd); pv_ = jax.device_put(pv)
        buf = jax.device_put(np.full((NPmax + CH, 3), RC.FAR, np.float32))
        blk((P_, R_, V_, lo_, cell_, b_, U, Pi_, Pj_, pd_, pv_, buf)); T["h2d"] += pc() - t
        t = pc(); scid, order = blk(GEO._cell_hash(P_, V_, lo_, cell_)); T["dev_cell_hash"] += pc() - t
        t = pc()
        cen, rho, e1, e2, m, cum = GEO.geo_pairs(P_, R_, Pi_, Pj_, pd_, pv_, arc_step=RC.ARC_STEP)
        NA = int(np.asarray(cum)[-1]); T["dev_geo_pairs"] += pc() - t
        NV = n * nvert
        off = 0; occmax = 0; ntr = 0; nvc = 0; nac = 0
        for s in range(0, NV, CH):
            t = pc()
            pts, cnt, tr, occ = GEO.geo_vertex(RC.dev(s, idt), RC.dev(NV, idt), P_, R_, V_, U, lo_, cell_, b_, scid, order,
                                               CH=CH, nvert=nvert, K=K)
            cnt = int(cnt); occmax = max(occmax, int(occ)); ntr += int(tr); T["dev_geo_vertex"] += pc() - t
            t = pc()
            if off + cnt <= NPmax:
                buf = blk(GEO.geo_pack(buf, pts, RC.dev(off, idt)))
            T["dev_geo_pack"] += pc() - t
            off += cnt; nvc += 1
            if decomp:
                t = pc(); q, own, okv = blk(d_vgen(RC.dev(s, idt), RC.dev(NV, idt), P_, R_, V_, U)); T["vx_gen"] += pc() - t
                t = pc(); cand, oc = blk(d_cand(q, lo_, cell_, scid, order)); T["vx_cand"] += pc() - t
                t = pc(); bb, ball = blk(d_burial(q, own, cand, P_, R_, b_)); T["vx_burial"] += pc() - t
                t = pc(); o2, n2, tr2, oc2 = blk(d_finish(q, okv, bb, ball, oc)); T["vx_finish"] += pc() - t
                par["vx_chunks"] += 1; par["vx_eq"] += int(bool(d_eq(o2, pts)) and int(n2) == cnt)
                del cand
        for s in range(0, NA, CH):
            t = pc()
            pts, cnt, tr, occ = GEO.geo_arc(RC.dev(s, idt), RC.dev(NA, idt), cum, m, cen, rho, e1, e2, Pi_, Pj_, P_, R_,
                                            lo_, cell_, b_, scid, order, CH=CH, K=K)
            cnt = int(cnt); occmax = max(occmax, int(occ)); ntr += int(tr); T["dev_geo_arc"] += pc() - t
            t = pc()
            if off + cnt <= NPmax:
                buf = blk(GEO.geo_pack(buf, pts, RC.dev(off, idt)))
            T["dev_geo_pack"] += pc() - t
            off += cnt; nac += 1
            if decomp:
                t = pc(); q, own, oka = blk(d_agen(RC.dev(s, idt), RC.dev(NA, idt), cum, m, cen, rho, e1, e2, Pi_, Pj_)); T["arc_gen"] += pc() - t
                t = pc(); cand, oc = blk(d_cand(q, lo_, cell_, scid, order)); T["arc_cand"] += pc() - t
                t = pc(); bb, ball = blk(d_burial(q, own, cand, P_, R_, b_)); T["arc_burial"] += pc() - t
                t = pc(); o2, n2, tr2, oc2 = blk(d_finish(q, oka, bb, ball, oc)); T["arc_finish"] += pc() - t
                par["arc_chunks"] += 1; par["arc_eq"] += int(bool(d_eq(o2, pts)) and int(n2) == cnt)
                del cand
        if off > NPmax or occmax > K:
            raise OverflowError(f"probes {off}/{NPmax} occ {occmax}/{K}")
        probes = np.asarray(buf[:off]) if prec == "f64" else None
    info = dict(n_atoms=n, n_pairs=int(len(i)), NV=NV, NA=NA, n_probes=off, vertex_chunks=nvc, arc_chunks=nac, knn_tr=ntr,
                cell_occ=occmax, c=c)
    return dict(T), info, dict(par), probes


def geo_flop_share(rec, c):
    """static element counts of arithmetic HLO ops by dtype in the optimised geo_vertex / geo_arc / geometry_sites
    programs (f64 path) + XLA cost_analysis. Approximate (no loop trip counts; geo_* have none)."""
    OPS = ("add", "subtract", "multiply", "divide", "sqrt", "rsqrt", "compare", "minimum", "maximum", "select", "floor",
           "convert", "cosine", "sine", "abs", "negate", "power", "exponential", "ceil", "and", "or", "not", "clamp")
    rx = re.compile(r"= (f64|f32|s64|s32|u32|u64|pred|s8|u8|s16)\[([0-9,]*)\][^ ]* (" + "|".join(OPS) + r")\(")

    def count(txt):
        acc = defaultdict(float)
        for m in rx.finditer(txt):
            dims = [int(x) for x in m.group(2).split(",") if x]
            acc[m.group(1)] += float(np.prod(dims)) if dims else 1.0
        tot = sum(acc.values())
        return dict(by_dtype=dict(acc), f64_share=(acc.get("f64", 0) / tot if tot else None),
                    float_f64_share=(acc.get("f64", 0) / max(acc.get("f64", 0) + acc.get("f32", 0), 1)))
    out = {}
    Pf, Rf, n = GEO.surface_atoms(rec.pos, rec.rad, c)
    N = RC.NA_C[c]; CH = RC.GEO_CHUNK
    with GEO.x64():
        f, i = jnp.float64, jnp.int64
        S = jax.ShapeDtypeStruct
        args = (S((), i), S((), i), S((N, 3), f), S((N,), f), S((N,), bool), S((RC.NVERT, 3), f), S((3,), f), S((), f), S((), f),
                S((N,), i), S((N,), i))
        comp = GEO.geo_vertex.lower(*args, CH=CH, nvert=RC.NVERT, K=RC.GEO_K).compile()
        out["geo_vertex"] = count(comp.as_text())
        try:
            ca = comp.cost_analysis(); ca = ca[0] if isinstance(ca, (list, tuple)) else ca
            out["geo_vertex"]["cost_flops"] = float(ca.get("flops", -1)); out["geo_vertex"]["cost_bytes"] = float(ca.get("bytes accessed", -1))
        except Exception as e:  # noqa: BLE001
            out["geo_vertex"]["cost_err"] = repr(e)[:200]
        NPAIR = RC.PAIRS_PER_ATOM * N
        args = (S((), i), S((), i), S((NPAIR,), i), S((NPAIR,), i), S((NPAIR, 3), f), S((NPAIR,), f), S((NPAIR, 3), f), S((NPAIR, 3), f),
                S((NPAIR,), i), S((NPAIR,), i), S((N, 3), f), S((N,), f), S((3,), f), S((), f), S((), f), S((N,), i), S((N,), i))
        comp = GEO.geo_arc.lower(*args, CH=CH, K=RC.GEO_K).compile()
        out["geo_arc"] = count(comp.as_text())
        try:
            ca = comp.cost_analysis(); ca = ca[0] if isinstance(ca, (list, tuple)) else ca
            out["geo_arc"]["cost_flops"] = float(ca.get("flops", -1)); out["geo_arc"]["cost_bytes"] = float(ca.get("bytes accessed", -1))
        except Exception as e:  # noqa: BLE001
            out["geo_arc"]["cost_err"] = repr(e)[:200]
    return out


# ------------------------------------------------------------------------------------------------ maps copies
@dataclass(frozen=True)
class GPar:
    h: float
    n: int
    Ka: int
    Kp: int
    nca: int
    ncp: int
    NS: int
    NB: int
    PB: int
    brute: bool
    hstat: float


def gpars(sz, PBc):
    n = RC.GSIZE
    nca_c, ncp_c = MP.ncells(1.0, n, "a"), MP.ncells(1.0, n, "p")
    nca_f, ncp_f = MP.ncells(0.25, n, "a"), MP.ncells(0.25, n, "p")
    return {"c": GPar(1.0, n, sz.Ka, sz.Kp, nca_c, ncp_c, sz.NSc, sz.NBc, PBc, False, 1.0),
            "f": GPar(0.25, n, sz.Ka, sz.Kp, nca_f, ncp_f, sz.NSf, sz.NBf, sz.PBf, False, 0.25),
            "s": GPar(0.25, n, sz.Ka, sz.Kp, nca_f, ncp_f, sz.NSs, sz.NBs, sz.PBs, True, 0.25)}


def _lo(acent, gp):
    rmax = 2.0; L = (gp.n - 1) * gp.h
    return acent - L / 2 - (rmax + MP.EXRAD + 0.5), rmax + MP.EXRAD + 0.01, acent - L / 2 - (MP.RP + 0.5)


def _grid(acent, gp):
    x, y, z = MP.grid_coords(acent, gp.h, gp.n)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    return jnp.stack([X, Y, Z], -1).reshape(-1, 3)


def _qmin(pos, tab_a, acent, gp):
    lo, ca, _ = _lo(acent, gp)
    if gp.brute:
        return lambda Q, w: MP.brute_min(Q, pos, w)
    return lambda Q, w: MP.query_min(Q, pos, w, tab_a, lo, gp.nca, ca)


def st_cells_a(pos, rad, acent, gp):
    lo, ca, _ = _lo(acent, gp)
    return MP.build_cells(pos, rad, lo, gp.nca, ca, gp.Ka)


def st_cells_p(probes, pos, acent, gp):
    _, _, lop = _lo(acent, gp)
    zp = jnp.zeros(probes.shape[0], pos.dtype)
    return MP.build_cells(probes, zp, lop, gp.ncp, MP.PCELL, gp.Kp)


def st_region(probes, acent, gp):
    return MP.region_probes(probes, acent, gp.h, gp.n, gp.PB)


def st_ion(pos, rad, tab_a, acent, gp):
    da, ia = _qmin(pos, tab_a, acent, gp)(_grid(acent, gp), rad)
    return (da >= MP.EXRAD).reshape(gp.n, gp.n, gp.n), ia


def st_mid(pos, rad, tab_a, acent, ax, gp):
    M = _grid(acent, gp) + jnp.zeros(3, pos.dtype).at[ax].set(gp.h / 2)
    return _qmin(pos, tab_a, acent, gp)(M, rad)


def st_contact(pos, rad, tab_a, acent, dv, iv, ax, gp):
    rp = MP.RP
    M = _grid(acent, gp) + jnp.zeros(3, pos.dtype).at[ax].set(gp.h / 2)
    shell = (dv >= 0) & (dv < rp)
    si = jnp.nonzero(shell, size=gp.NS, fill_value=0)[0]
    Ms = M[si]
    ia = jnp.maximum(iv[si], 0); u = Ms - pos[ia]; u = u / jnp.maximum(jnp.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    cpr = pos[ia] + (rad[ia] + rp)[:, None] * u
    dcp, _ = _qmin(pos, tab_a, acent, gp)(cpr, rad + rp)
    contact = dcp >= -1e-5
    need = jnp.zeros_like(shell).at[si].max(~contact) & shell
    return shell, need, si, contact, jnp.sum(shell), jnp.sum(need)


def st_reach(preg, pval, acent, ax, gp):
    return MP.reach_mask(preg, pval, acent, gp.h, gp.n, ax, gp.hstat)


def st_combine(dv, shell, need, si, contact, reachm, gp):
    reach = jnp.zeros(dv.shape[0], bool).at[si].max(contact) | (need & reachm)
    solvent = (dv >= MP.RP) | (shell & reach)
    return (~solvent).reshape(gp.n, gp.n, gp.n)


def st_boundary(ex, ey, ez, gp):
    nb = [ex, jnp.roll(ex, 1, 0), ey, jnp.roll(ey, 1, 1), ez, jnp.roll(ez, 1, 2)]
    s = sum(a.astype(jnp.int32) for a in nb)
    bnd = (s > 0) & (s < 6)
    bnd = bnd.at[0].set(False).at[-1].set(False).at[:, 0].set(False).at[:, -1].set(False).at[:, :, 0].set(False).at[:, :, -1].set(False)
    bflat = bnd.reshape(-1)
    return jnp.nonzero(bflat, size=gp.NB, fill_value=-1)[0], jnp.sum(bflat)


def st_proj_a(pos, rad, tab_a, acent, bidx, ia_pts, gp):
    rp = MP.RP
    Gb = _grid(acent, gp)[jnp.maximum(bidx, 0)]
    ia = jnp.maximum(ia_pts[jnp.maximum(bidx, 0)], 0)
    v = Gb - pos[ia]; nv = jnp.linalg.norm(v, axis=1, keepdims=True)
    sa = pos[ia] + rad[ia][:, None] * v / jnp.maximum(nv, 1e-6)
    cpr = pos[ia] + (rad[ia] + rp)[:, None] * v / jnp.maximum(nv, 1e-6)
    dcp, _ = _qmin(pos, tab_a, acent, gp)(cpr, rad + rp)
    needb = (dcp < -1e-5) & (bidx >= 0)
    bi2 = jnp.nonzero(needb, size=gp.NB, fill_value=0)[0]
    return Gb, sa, dcp, needb, bi2, jnp.sum(needb)


def st_proj_p(probes, pos, tab_p, acent, Gb, sa, dcp, needb, bi2, gp):
    rp = MP.RP
    _, _, lop = _lo(acent, gp)
    zp = jnp.zeros(probes.shape[0], pos.dtype)
    _, ip2 = MP.query_min(Gb[bi2], probes, zp, tab_p, lop, gp.ncp, MP.PCELL, reach=MP.PREACH)
    ip = jnp.full(Gb.shape[0], -1, ip2.dtype).at[bi2].set(jnp.where(needb[bi2], ip2, -1))
    ipc = probes[jnp.maximum(ip, 0)]
    vp = Gb - ipc; nvp = jnp.linalg.norm(vp, axis=1, keepdims=True)
    sp = ipc + rp * vp / jnp.maximum(nvp, 1e-6)
    use_atom = (dcp >= -1e-5) | (ip < 0)
    return jnp.where(use_atom[:, None], sa, sp)


_JC = {}


def J(name, fn, gp, axes):
    """jit(vmap(fn)) per (stage, grid params, in_axes) -- cached so each compiles once."""
    key = (name, gp, axes)
    if key not in _JC:
        _JC[key] = jax.jit(jax.vmap(partial(fn, gp=gp), in_axes=axes))
    return _JC[key]


_WARM = set()


def timed(T, key, f, *a):
    wk = (key, id(f), tuple((tuple(x.shape), str(x.dtype)) if hasattr(x, "shape") else None for x in a))
    if wk not in _WARM:      # first call per program and shape compiles: run once untimed
        blk(f(*a)); _WARM.add(wk)
    t = pc(); r = blk(f(*a)); T[key] += pc() - t
    return r


def maps_decomp(g, gp, pos, rad, probes, acent, shared):
    """one grid of build_maps as separate stage programs vmapped over B. shared: atoms/probes not batched (c, f)."""
    T = defaultdict(float); A = None if shared else 0
    out = {}
    if not gp.brute:
        tab_a, _ = timed(T, "cells_atoms", J("ca", st_cells_a, gp, (A, A, 0)), pos, rad, acent)
    else:
        tab_a = jnp.zeros((acent.shape[0], 1, 1), jnp.int32)
    tab_p, _ = timed(T, "cells_probes", J("cp", st_cells_p, gp, (A, A, 0)), probes, pos, acent)
    preg, pval, npreg = timed(T, "region_probes", J("rp", st_region, gp, (A, 0)), probes, acent)
    qn = "brute_min" if gp.brute else "query_min"
    ion, ia_pts = timed(T, f"ion_{qn}", J("ion", st_ion, gp, (A, A, 0, 0)), pos, rad, tab_a, acent)
    E = []; shells = []; needs = []
    for ax in range(3):
        axd = jnp.asarray(ax, jnp.int32)
        dv, iv = timed(T, f"mid_{qn}", J("mid", st_mid, gp, (A, A, 0, 0, None)), pos, rad, tab_a, acent, axd)
        shell, need, si, contact, nsh, nne = timed(T, "shell_contact", J("con", st_contact, gp, (A, A, 0, 0, 0, 0, None)),
                                                   pos, rad, tab_a, acent, dv, iv, axd)
        rm = timed(T, "reach_mask", J("rm", st_reach, gp, (0, 0, 0, None)), preg, pval, acent, axd)
        e = timed(T, "combine", J("cb", st_combine, gp, (0, 0, 0, 0, 0, 0)), dv, shell, need, si, contact, rm)
        E.append(e); shells.append(np.asarray(nsh)); needs.append(np.asarray(nne))
    bidx, nbd = timed(T, "boundary", J("bd", st_boundary, gp, (0, 0, 0)), *E)
    Gb, sa, dcp, needb, bi2, nbp = timed(T, f"proj_atoms_{qn}", J("pa", st_proj_a, gp, (A, A, 0, 0, 0, 0)), pos, rad, tab_a, acent, bidx, ia_pts)
    surf = timed(T, "proj_probes_query", J("pp", st_proj_p, gp, (A, A, 0, 0, 0, 0, 0, 0, 0)), probes, pos, tab_p, acent, Gb, sa, dcp, needb, bi2)
    out = dict(ex=E[0], ey=E[1], ez=E[2], ion=ion, bidx=bidx, surf=surf)
    cnt = dict(shell=np.stack(shells, 1), need=np.stack(needs, 1), npreg=np.asarray(npreg), nbd=np.asarray(nbd), nbproj=np.asarray(nbp))
    return dict(T), out, cnt


@partial(jax.jit, static_argnames=("k",))
def block_stats(ex, ey, ez, ion, k):
    """4^3 / 8^3 block classes of one grid: inside (all midpoints eps_in), outside (all eps_out and ion constant), mixed.
    Edge blocks are padded by edge replication (no new values). Returns block counts and original-point counts."""
    n = ex.shape[0]; p = (-n) % k; m = (n + p) // k
    pe = lambda a: jnp.pad(a, ((0, p),) * 3, mode="edge")
    Eb = jnp.stack([pe(ex), pe(ey), pe(ez)]).reshape(3, m, k, m, k, m, k)
    Ib = pe(ion).reshape(m, k, m, k, m, k)
    allin = jnp.all(Eb, axis=(0, 2, 4, 6)); anyin = jnp.any(Eb, axis=(0, 2, 4, 6))
    ia = jnp.all(Ib, axis=(1, 3, 5)); ny = jnp.any(Ib, axis=(1, 3, 5))
    ionc = ia | ~ny
    inside = allin; outside = ~anyin & ionc; mixed = ~inside & ~outside
    W = jnp.pad(jnp.ones((n, n, n), jnp.int32), ((0, p),) * 3).reshape(m, k, m, k, m, k).sum((1, 3, 5))
    outside_eps = ~anyin
    return jnp.stack([inside.sum(), outside.sum(), mixed.sum(), (W * inside).sum(), (W * outside).sum(), (W * mixed).sum(),
                      (inside & ionc).sum(), outside_eps.sum(), (W * outside_eps).sum()])


BS_NAMES = ("blk_inside", "blk_outside", "blk_mixed", "pts_inside", "pts_outside", "pts_mixed", "blk_inside_ionconst",
            "blk_outside_eps_only", "pts_outside_eps_only")
@partial(jax.jit, static_argnames=("k",))
def block_stats_b(ex, ey, ez, ion, k):
    return jax.vmap(lambda a, b, c, d: block_stats(a, b, c, d, k=k))(ex, ey, ez, ion)


# ------------------------------------------------------------------------------------------------ triple reuse
def triple_reuse(cid, recs, probes, geoT, cls):
    from scipy.spatial import cKDTree
    ab, a, b = (recs[f"{cid}_{s}"] for s in ("AB", "A", "B"))

    def surf(r):
        k = r.rad > 0
        return r.pos[k].astype(np.float32), r.rad[k].astype(np.float32)

    def rows(x):
        x = np.ascontiguousarray(x, np.float32)
        return x.view(np.dtype((np.void, x.dtype.itemsize * x.shape[1]))).ravel()
    Pab, Rab = surf(ab)
    ab_atoms = rows(np.concatenate([Pab, Rab[:, None]], 1))
    ab_probes = rows(probes[f"{cid}_AB"])
    out = {}
    for s, r, other in (("A", a, b), ("B", b, a)):
        P, Rr = surf(r)
        atoms = rows(np.concatenate([P, Rr[:, None]], 1))
        am = np.isin(atoms, ab_atoms)
        # partner atoms = AB surface atoms not in this chain (exact (pos, rad) match)
        pm = ~np.isin(ab_atoms, atoms)
        partner = Pab[pm].astype(np.float64)
        R = Rr.astype(np.float64) + RC.RADPRB
        Rmax_ab = float(Rab.max()) + RC.RADPRB
        bound = Rmax_ab + 1e-3
        tr = cKDTree(partner)
        near = np.array(tr.query_ball_point(P.astype(np.float64), R + bound, return_length=True)) > 0
        # pair list exactly as production (A-only), arc counts m per pair (host f64)
        pos = P.astype(np.float64)
        pairs = cKDTree(pos).query_pairs(2 * float(R.max()), output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.linalg.norm(pos[j] - pos[i], axis=1)
        ok = (d < R[i] + R[j]) & (d > np.abs(R[i] - R[j])) & (d > 1e-6)
        i, j, d = i[ok], j[ok], d[ok]
        aa = (d ** 2 + R[i] ** 2 - R[j] ** 2) / (2 * d)
        rho = np.sqrt(np.maximum(R[i] ** 2 - aa ** 2, 0))
        mm = np.maximum(8, np.ceil(2 * np.pi * rho / RC.ARC_STEP)).astype(np.int64)
        far_pair = ~(near[i] & near[j])
        nv_tot = len(P) * RC.NVERT; nv_far = int((~near).sum()) * RC.NVERT
        na_tot = int(mm.sum()); na_far = int(mm[far_pair].sum())
        pr = probes[f"{cid}_{s}"]
        inab = np.isin(rows(pr), ab_probes)
        dmin, _ = tr.query(pr.astype(np.float64), distance_upper_bound=bound)
        farp = ~np.isfinite(dmin) | (dmin >= bound)
        T = geoT[f"{cid}_{s}"]
        tv = T.get("dev_geo_vertex", 0.0); ta = T.get("dev_geo_arc", 0.0)
        out[s] = dict(n_probes=len(pr), probes_in_AB=float(inab.mean()), atoms_in_AB=float(am.mean()),
                      n_partner_atoms=int(pm.sum()), atoms_near_partner=float(near.mean()),
                      vertex_cand_far=nv_far / nv_tot, arc_cand_far=na_far / max(na_tot, 1),
                      arc_cand_host_vs_device=None,
                      probes_far_from_partner=float(farp.mean()), far_probes_in_AB=float(inab[farp].mean()) if farp.any() else None,
                      near_probes_in_AB=float(inab[~farp].mean()) if (~farp).any() else None,
                      geo_kernel_s=tv + ta, reusable_kernel_s=tv * nv_far / nv_tot + ta * na_far / max(na_tot, 1),
                      c=cls[f"{cid}_{s}"])
    tab = geoT[f"{cid}_AB"]
    tot = sum(geoT[f"{cid}_{s}"].get("dev_geo_vertex", 0) + geoT[f"{cid}_{s}"].get("dev_geo_arc", 0) for s in ("AB", "A", "B"))
    totall = sum(sum(v for k, v in geoT[f"{cid}_{s}"].items() if not k.startswith(("vx_", "arc_"))) for s in ("AB", "A", "B"))
    reuse = out["A"]["reusable_kernel_s"] + out["B"]["reusable_kernel_s"]
    out["triple"] = dict(vertex_arc_kernel_s=tot, reusable_kernel_s=reuse, reusable_share_of_vertex_arc=reuse / tot if tot else None,
                         reusable_share_of_geometry_ops=reuse / totall if totall else None, AB_c=cls[f"{cid}_AB"])
    return out


# ------------------------------------------------------------------------------------------------ main modes
def mode_main(a):
    T0 = pc()
    os.makedirs(a.out, exist_ok=True)
    R = dict(meta=dict(host=os.uname().nodename, jax=jax.__version__, device=str(jax.devices()[0]), job=os.environ.get("SLURM_JOB_ID"),
                       code=os.getcwd(), states=None))
    path = os.path.join(a.out, "profile_raw.json")

    def save(tag):
        R["meta"]["elapsed_s"] = pc() - T0; R["meta"]["last_pass"] = tag
        with open(path + ".tmp", "w") as f:
            json.dump(jsafe(R), f, indent=1)
        os.replace(path + ".tmp", path)
        print(f"[prof] saved after {tag} ({pc() - T0:.0f}s)", flush=True)
    R["cache"] = cache_open(a.cc_tar, a.cc_dir) if a.cc_tar else None
    ids = a.states.split(",") if a.states else DEFAULT_STATES
    recs = load(a.inputs, ids)
    recs.sort(key=lambda r: (r.size_class, r.n_atoms))
    RD = {r.state_id: r for r in recs}
    cls = {r.state_id: r.size_class for r in recs}
    R["meta"]["states"] = [dict(id=r.state_id, c=r.size_class, n_atoms=r.n_atoms, n_surface=r.n_surface_atoms, n_sites=r.n_sites,
                                n_taus=r.n_taus) for r in recs]
    cfg = pbcfg()
    R["meta"]["pbcfg"] = dict(backend=cfg.backend, specrad=cfg.specrad, B=cfg.B, resolved=PB.resolve_backend(cfg.backend))
    with CompileCounter() as cc:
        R["warm_s"] = warm(recs, cfg)
    R["warm_compiles"] = cc.n_compiled
    save("warm")
    # ---------------------------------------------------------------- A: production timing (untraced)
    A = {}
    for r in recs:
        _, tim, segs, wall = solve(r, cfg)
        A[r.state_id] = dict(tim=tim, seg=seg_sums(segs), wall=wall, nseg=len(segs))
        print(f"[A] {r.state_id} c{cls[r.state_id]} wall {wall:.2f} geo {tim['t_geometry']:.2f} maps {tim['t_maps']:.2f} sor {tim['t_sor']:.2f}", flush=True)
    R["A"] = A; save("A")
    # ---------------------------------------------------------------- B: traced pass
    B = {}; tbase = a.trace_dir
    keep = a.keep_traces.split(",") if a.keep_traces else KEEP_TRACES
    first = True
    for r in recs:
        td = os.path.join(tbase, r.state_id)
        shutil.rmtree(td, ignore_errors=True)
        h = Hook(True); tim = {"memtrack": h}
        jax.profiler.start_trace(td)
        h.open("geometry"); t = pc()
        PB.solve_state(r, cfg, timing=tim)
        wall = pc() - t
        segs = h.finish()
        jax.profiler.stop_trace()
        names = {s["k"]: s["name"] for s in segs}
        want_ops = r.state_id in keep
        t = pc()
        try:
            res, ops = parse_trace(td, names, want_ops=want_ops, diag=first)
        except Exception as e:  # noqa: BLE001
            res, ops = {"error": repr(e)[:400]}, None
        res["parse_s"] = pc() - t; res["wall_traced"] = wall; res["seg_host"] = seg_sums(segs)
        if ops is not None:
            res["maps_ops_ns"] = dict(sorted(ops.items(), key=lambda kv: -kv[1])[:400])
        B[r.state_id] = res
        first = False
        if r.state_id in keep:
            f = glob.glob(os.path.join(td, "**", "*.xplane.pb"), recursive=True)
            if f:
                os.makedirs(os.path.join(a.out, "traces"), exist_ok=True)
                with open(f[0], "rb") as fi, gzip.open(os.path.join(a.out, "traces", f"{r.state_id}.xplane.pb.gz"), "wb") as fo:
                    shutil.copyfileobj(fi, fo)
        shutil.rmtree(td, ignore_errors=True)
        print(f"[B] {r.state_id} traced wall {wall:.2f} parse {res['parse_s']:.1f}s {res.get('error', '')}", flush=True)
    R["B"] = B; save("B")
    # ---------------------------------------------------------------- C: geometry decomposition + FP32 timing experiment
    C = {}; probes = {}; geoT = {}
    for r in recs:
        c = r.size_class
        while True:
            try:
                if ("f64dec", c) not in _WARM:       # compile the decomposition copies for this class (untimed)
                    geo_run(r, c, "f64", decomp=True); _WARM.add(("f64dec", c))
                T64, info, par, pr = geo_run(r, c, "f64", decomp=True)
                break
            except OverflowError:
                c += 1
        probes[r.state_id] = pr; geoT[r.state_id] = T64
        # production-pattern loops (no decomposition) f64 vs f32; first state of a class compiles the f32 variant
        Tp64, _, _, _ = geo_run(r, c, "f64", decomp=False)
        try:
            if ("f32geo", c) not in _WARM:
                geo_run(r, c, "f32"); _WARM.add(("f32geo", c))
            Tp32, info32, _, _ = geo_run(r, c, "f32")
            e32 = None
        except Exception as e:  # noqa: BLE001
            Tp32, info32, e32 = {}, {}, repr(e)[:300]
        C[r.state_id] = dict(decomp=T64, info=info, parity=par, prod_f64=Tp64, prod_f32=Tp32, f32_n_probes=info32.get("n_probes"),
                             f32_err=e32)
        print(f"[C] {r.state_id} c{c} f64 vx {Tp64.get('dev_geo_vertex', 0):.2f} arc {Tp64.get('dev_geo_arc', 0):.2f} | f32 vx "
              f"{Tp32.get('dev_geo_vertex', 0):.2f} arc {Tp32.get('dev_geo_arc', 0):.2f} | parity {par}", flush=True)
    R["C"] = C; save("C")
    # ---------------------------------------------------------------- D: maps per batch + culling + batch-0 decomposition
    D = {}; keep_args = None
    with GEO.x32():
        for r in recs:
            c = r.size_class
            geo = GEO.geometry_state(r.pos, r.rad, c)
            par_probes = bool(np.array_equal(np.asarray(geo.probes[:geo.n_probes]), probes[r.state_id]))
            pos_m = jax.device_put(geo.pos); rad_m = jax.device_put(geo.rad)
            sz = cfg.sizes; GP = gpars(sz, geo.probes.shape[0])
            sl = list(range(r.n_sites)); nb = int(np.ceil(len(sl) / cfg.B))
            TT = defaultdict(float); stats = {g: {k: np.zeros(len(BS_NAMES), np.int64) for k in (4, 8)} for g in "cfs"}
            ctrs = []; dec = None
            for bi in range(nb):
                bl = sl[bi * cfg.B:(bi + 1) * cfg.B]; nreal = len(bl)
                t = pc(); lay, tl = PB._site_layout(r, bl, cfg.B, lambda tt, kk: 0.99); TT["host_site_layout"] += pc() - t
                t = pc(); sprb, scnt, smx, _ = PB.site_probes(lay["spos"], lay["srad"], sz.PBs); blk(sprb); TT["site_probes_f64"] += pc() - t
                t = pc(); jj = {k: RC.dev(v, np.float32) for k, v in lay.items()}; blk(jj); TT["h2d"] += pc() - t
                t = pc(); mc, mf, ms, ctr = blk(MP.maps_batch(pos_m, rad_m, geo.probes, jj["ac"], jj["spos"], jj["srad"], sprb, sz=sz))
                TT["maps_batch"] += pc() - t
                t = pc(); ctr_h = np.asarray(ctr); MP.overflow(ctr_h, sz); TT["ctr_check"] += pc() - t
                ctrs.append(ctr_h[:nreal])
                # FP32 site probes (timing only)
                sv = lay["srad"] > 0; sR = np.where(sv, lay["srad"] + RC.RADPRB, 0.0)
                key = ("f32sites",)
                U32 = RC.dev(GEO.fib_sphere(RC.NVERT), np.float32)
                args32 = (RC.dev(lay["spos"], np.float32), RC.dev(sR, np.float32), jax.device_put(sv), U32)
                if key not in _WARM:
                    blk(GEO.geometry_sites(*args32, OUT=sz.PBs)); _WARM.add(key)
                t = pc(); blk(GEO.geometry_sites(*args32, OUT=sz.PBs)); TT["site_probes_f32"] += pc() - t
                for g, mm in (("c", mc), ("f", mf), ("s", ms)):
                    for k in (4, 8):
                        bs = np.asarray(block_stats_b(mm["ex"], mm["ey"], mm["ez"], mm["ion"], k=k))[:nreal]
                        stats[g][k] += bs.sum(0)
                if bi == 0:
                    dec = {}
                    for g in "cfs":
                        gp = GP[g]
                        if g == "s":
                            T_, out_, cnt_ = maps_decomp(g, gp, jj["spos"], jj["srad"], sprb, jj["ac"], shared=False)
                        else:
                            T_, out_, cnt_ = maps_decomp(g, gp, pos_m, rad_m, geo.probes, jj["ac"], shared=True)
                        ref = {"c": mc, "f": mf, "s": ms}[g]
                        eq = {k: float(np.mean(np.asarray(out_[k])[:nreal] == np.asarray(ref[k])[:nreal])) for k in out_}
                        nn = RC.GSIZE ** 3
                        dec[g] = dict(T=T_, parity=eq, shell_frac=float(cnt_["shell"][:nreal].mean() / nn),
                                      need_frac=float(cnt_["need"][:nreal].mean() / nn),
                                      shell_frac_max=float(cnt_["shell"][:nreal].max() / nn), need_frac_max=float(cnt_["need"][:nreal].max() / nn),
                                      npreg_mean=float(cnt_["npreg"][:nreal].mean()), PB_cap=gp.PB, nbd_mean=float(cnt_["nbd"][:nreal].mean()),
                                      nbproj_mean=float(cnt_["nbproj"][:nreal].mean()), nreal=nreal)
                        del out_
                    # fused per-grid timing: maps_batch has all three; time the batch once more for the batch-0 reference
                    t = pc(); blk(MP.maps_batch(pos_m, rad_m, geo.probes, jj["ac"], jj["spos"], jj["srad"], sprb, sz=sz)); dec["fused_batch0_s"] = pc() - t
                    if r.size_class == 2 and keep_args is None:
                        keep_args = (pos_m, rad_m, geo.probes, jj["ac"], jj["spos"], jj["srad"], sprb, sz)
                del mc, mf, ms
            C_ = np.concatenate(ctrs)
            D[r.state_id] = dict(T=dict(TT), nbatches=nb, n_sites=r.n_sites, probes_equal_geo_run=par_probes,
                                 blocks={g: {str(k): dict(zip(BS_NAMES, stats[g][k].tolist())) for k in (4, 8)} for g in "cfs"},
                                 npreg={g: float(C_[:, gi, 5].mean() + (GP[g].PB)) for gi, g in enumerate("cfs")},
                                 n_bnd={g: float(C_[:, gi, 6].mean()) for gi, g in enumerate("cfs")}, decomp=dec)
            print(f"[D] {r.state_id} maps_batch {TT['maps_batch']:.2f}s/{nb} sites f64 {TT['site_probes_f64']:.2f} f32 "
                  f"{TT['site_probes_f32']:.2f} probes_eq {par_probes}", flush=True)
            del geo
    R["D"] = D; save("D")
    # ---------------------------------------------------------------- E: AB/A/B reuse
    E = {}
    for cid in TRIPLES:
        if all(f"{cid}_{s}" in probes for s in ("AB", "A", "B")):
            try:
                E[cid] = triple_reuse(cid, RD, probes, geoT, cls)
            except Exception as e:  # noqa: BLE001
                E[cid] = {"error": repr(e)[:300]}
    R["E"] = E; save("E")
    # ---------------------------------------------------------------- F: HLO stats
    F = {}
    try:
        r2 = next((r for r in recs if r.size_class == 2), recs[-1])
        F["geo_flops_state"] = r2.state_id
        F["geo"] = geo_flop_share(r2, r2.size_class)
    except Exception as e:  # noqa: BLE001
        F["geo_err"] = repr(e)[:400]
    if keep_args is not None:
        try:
            with GEO.x32():
                comp = MP.maps_batch.lower(*keep_args[:7], sz=keep_args[7]).compile()
                txt = comp.as_text()
            meta = {}
            rx = re.compile(r"^\s*%?([\w.\-]+) = .*?metadata=\{([^}]*)\}", re.M)
            for m in rx.finditer(txt):
                md = m.group(2); sl = re.search(r"source_line=(\d+)", md); sf = re.search(r'source_file="([^"]+)"', md)
                v = (os.path.basename(sf.group(1)) if sf else "?", int(sl.group(1)) if sl else -1)
                meta[m.group(1)] = v; meta[m.group(1).replace(".", "_")] = v
            funcs = {}
            for nm in ("build_cells", "query_min", "reach_offsets", "region_probes", "reach_mask", "brute_min", "build_maps", "maps_batch", "grid_coords"):
                src, st = inspect.getsourcelines(getattr(MP, nm)); funcs[nm] = (st, st + len(src) - 1)

            def fn_of(f, line):
                if f != "maps.py":
                    return f"{f}"
                for nm, (s0, s1) in funcs.items():
                    if s0 <= line <= s1:
                        return nm
                return "maps.py:?"
            kid = next((s_ for s_ in keep if cls.get(s_) == 2), None)
            kst = B.get(kid, {}) if kid else {}
            ops = kst.get("maps_ops_ns", {}) if isinstance(kst, dict) else {}
            byfn = defaultdict(float); byline = defaultdict(float); matched = 0.0; tot = 0.0
            for op, ns in ops.items():
                tot += ns
                if op in meta:
                    f, line = meta[op]; matched += ns
                    byfn[fn_of(f, line)] += ns; byline[f"{f}:{line}"] += ns
            F["maps_source"] = dict(state=kid, total_ns=tot, matched_ns=matched,
                                    by_function=dict(sorted(byfn.items(), key=lambda kv: -kv[1])),
                                    by_line=dict(sorted(byline.items(), key=lambda kv: -kv[1])[:40]), n_meta=len(meta))
        except Exception as e:  # noqa: BLE001
            F["maps_source_err"] = repr(e)[:400]
    R["F"] = F; save("F")
    R["meta"]["total_s"] = pc() - T0
    save("done")


def mode_thru(a):
    os.makedirs(a.out, exist_ok=True)
    cache_open(a.cc_tar, a.cc_dir) if a.cc_tar else None
    ids = a.states.split(",") if a.states else THRU
    recs = load(a.inputs, ids)
    cfg = pbcfg()
    tw = warm(recs, cfg)
    os.makedirs(a.barrier, exist_ok=True)
    open(os.path.join(a.barrier, f"ready_{a.tag}"), "w").close()
    while len(glob.glob(os.path.join(a.barrier, "ready_*"))) < a.nproc:
        time.sleep(0.02)
    t0e = time.time(); t0 = pc(); per = []
    for rep in range(a.repeat):
        for r in recs:
            _, tim, segs, wall = solve(r, cfg)
            per.append(dict(id=r.state_id, rep=rep, wall=wall, seg=seg_sums(segs), t_geometry=tim["t_geometry"], t_maps=tim["t_maps"],
                            t_sor=tim["t_sor"], t_solves=tim["t_solves"], t_post=tim["t_post"]))
    wall = pc() - t0
    res = dict(tag=a.tag, nproc=a.nproc, repeat=a.repeat, warm_s=tw, wall=wall, t0_epoch=t0e, t1_epoch=time.time(), per=per,
               stage_sum={k: sum(p["seg"].get(k, 0) for p in per) for k in ("geometry", "maps", "sor", "post", "tail")})
    with open(os.path.join(a.out, f"thru_{a.tag}.json"), "w") as f:
        json.dump(jsafe(res), f, indent=1)
    print(f"[thru] {a.tag} nproc {a.nproc} wall {wall:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("main", "thru"), required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--states", default=None)
    ap.add_argument("--keep-traces", default=None)
    ap.add_argument("--trace-dir", default=f"/tmp/jpa-prof-trace-{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    ap.add_argument("--cc-tar", default=None)
    ap.add_argument("--cc-dir", default=None)
    ap.add_argument("--tag", default="p0")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--barrier", default=None)
    a = ap.parse_args()
    if a.mode == "main":
        mode_main(a)
        shutil.rmtree(a.trace_dir, ignore_errors=True)
    else:
        mode_thru(a)


if __name__ == "__main__":
    main()
