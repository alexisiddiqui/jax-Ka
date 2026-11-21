"""jaxpb.geometry -- probe centres (exposed SAS points) for the whole molecule and the model compounds.

Same construction as jaxpb-pilot v1 (jaxpb.probe_centres) / v2 (maps2.probe_centres_gpu, site_probes_gpu): Fibonacci
vertices (NVERT) on every r + 1.4 sphere plus points every ARC_STEP A on every pairwise SAS circle, dropped when strictly
inside (tol 1e-6) a non-own expanded sphere among the 24 nearest atoms within Rmax + 1e-3; float64 inside, float32 out.
Order of the output = v2's (vertices atom-major, then arcs pair-major), which keeps nearest-probe tie breaking equal.

Shape discipline (records.py): all static shapes come from the class c (NA_C[c] atoms, NP_C[c] probes, PAIRS_PER_ATOM,
GEO_CHUNK, GEO_K) and the site batch (B, MA, MP_MODEL, MX). Burial candidates come from a hashed cell list (sorted cell
ids + searchsorted on a fixed 1024^3 virtual grid) instead of v2's dense, data-sized nc^3 table. The overlapping-pair
list is built on the host (scipy cKDTree, as v1/v2) and padded to PAIRS_PER_ATOM * NA_C[c]. Every capacity has a counter;
GeoState.flags reports them (overflow -> caller retries in the next class / XL sizes).
float64 is enabled only around these calls (x64() context, as v2): process-wide x64 would silently promote the f32
map/PB code through weak-typed Python scalars and break parity with the f32 prototypes (see DEVIATIONS in pb.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp

from . import records as RC



def _x64ctx(flag: bool):
    if hasattr(jax, "enable_x64"):
        try:
            return jax.enable_x64(flag)
        except TypeError:
            pass
    from jax import experimental as _jx
    return _jx.enable_x64() if flag else _jx.disable_x64()


def x64():
    """float64 context (JAX 0.11: jax.enable_x64(True); older: jax.experimental.enable_x64())."""
    return _x64ctx(True)


def x32():
    """float64 OFF context: the PB float32 programs must not see process-wide x64 (weak-typed scalars would promote)."""
    return _x64ctx(False)

NCV = 1024                       # virtual cell grid per axis (hashed cells)
MX = 720                         # arc points per pair, model compounds (v2: ceil(2 pi 3.4 / 0.03) + 1 = 702)
OFFS27 = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], np.int64)


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


@dataclass
class GeoState:
    pos: np.ndarray          # [NA_C[c],3] f32 surface atoms (radius > 0) in original order, FAR-padded
    rad: np.ndarray          # [NA_C[c]] f32, 0 for padding
    n_atoms: int
    probes: jax.Array        # [NP_C[c],3] f32 device, FAR-padded
    n_probes: int
    c: int
    flags: Dict[str, int] = field(default_factory=dict)
    reuse: Optional["GeoReuse"] = None


@dataclass
class GeoReuse:
    """SPEED lever 3: what the A / B states of a triple need from their AB geometry (kept by the driver until both ran)."""
    pos: np.ndarray          # [n,3] f32 surface atoms of AB
    rad: np.ndarray          # [n] f32
    bound: float
    pi: np.ndarray           # [P] host pair list (cKDTree order)
    pj: np.ndarray
    cum: jax.Array           # [NPAIR] device arc-count cumsum / counts of AB's padded pair list
    m: jax.Array
    kidx: jax.Array          # [NP_C + CH] device AB candidate index of every kept probe (ascending, KBIG padded)
    nvert: int


# ------------------------------------------------------------------------------------------------ shared pieces
def _burial(pts, own, cand, cpos, cR, bound, knn):
    """v2 maps2._burial: dropped iff strictly inside (tol 1e-6) a non-own expanded sphere that is among the knn nearest
    candidate atoms within `bound` (cKDTree k=24 semantics)."""
    ii = jnp.maximum(cand, 0)
    dd = jnp.sqrt(jnp.sum((pts[:, None, :] - cpos[ii]) ** 2, axis=-1))
    v = (cand >= 0) & (dd < bound)
    inside = v & (dd < cR[ii] - 1e-6) & (ii != own[:, :1]) & (ii != own[:, 1:2])
    dmin_in = jnp.min(jnp.where(inside, dd, jnp.inf), axis=1)
    rank = jnp.sum(v & (dd < dmin_in[:, None]), axis=1)
    anyin = jnp.any(inside, axis=1)
    return anyin & (rank < knn), anyin


def _cell_of(pts, lo, cell):
    c = jnp.floor((pts - lo) / cell).astype(jnp.int64)
    return jnp.clip(c, 0, NCV - 1)


def _hash_cells(P, valid, lo, cell):
    """sorted cell ids of the atoms (invalid atoms -> sentinel NCV^3), and the permutation."""
    c = _cell_of(P, lo, cell)
    cid = jnp.where(valid, (c[:, 0] * NCV + c[:, 1]) * NCV + c[:, 2], NCV ** 3)
    order = jnp.argsort(cid, stable=True)
    return cid[order], order


def _cand(pts, lo, cell, scid, order, K):
    """candidate atom indices (m, 27K) from the 27 neighbouring hashed cells, -1 padded; overflow count (max cell occ)."""
    c = _cell_of(pts, lo, cell)[:, None, :] + jnp.asarray(OFFS27)[None]
    okc = jnp.all((c >= 0) & (c < NCV), axis=2)
    cid = jnp.where(okc, (c[..., 0] * NCV + c[..., 1]) * NCV + c[..., 2], -1)
    st = jnp.searchsorted(scid, cid, side="left"); en = jnp.searchsorted(scid, cid, side="right")
    kk = st[..., None] + jnp.arange(K)
    ok = (kk < en[..., None]) & okc[..., None]
    idx = jnp.where(ok, order[jnp.minimum(kk, order.shape[0] - 1)], -1)
    return idx.reshape(pts.shape[0], -1), jnp.max(jnp.where(okc, en - st, 0))


def _pair_geom(pos, R, i, j, d, arc_step):
    """v2 maps2._pair_geom (float64)."""
    nrm = (pos[j] - pos[i]) / d[:, None]
    a_ = (d ** 2 + R[i] ** 2 - R[j] ** 2) / (2 * d)
    rho = jnp.sqrt(jnp.maximum(R[i] ** 2 - a_ ** 2, 0))
    cen = pos[i] + a_[:, None] * nrm
    tmp = jnp.where(jnp.abs(nrm[:, :1]) < 0.9, jnp.array([[1.0, 0, 0]], jnp.float64), jnp.array([[0, 1.0, 0]], jnp.float64))
    e1 = jnp.cross(nrm, tmp); e1 = e1 / jnp.linalg.norm(e1, axis=1, keepdims=True); e2 = jnp.cross(nrm, e1)
    m = jnp.maximum(8, jnp.ceil(2 * np.pi * rho / arc_step)).astype(jnp.int64)
    return cen, rho, e1, e2, m


# ------------------------------------------------------------------------------------------------ whole molecule
@partial(jax.jit, static_argnames=("arc_step",))
def geo_pairs(P, Rj, pi, pj, pd, pvalid, arc_step=RC.ARC_STEP):
    """per padded pair: circle geometry and arc-point count (0 for padding); cumulative counts."""
    dsafe = jnp.where(pvalid, pd, 1.0)
    cen, rho, e1, e2, m = _pair_geom(P, Rj, pi, pj, dsafe, arc_step)
    m = jnp.where(pvalid, m, 0)
    return cen, rho, e1, e2, m, jnp.cumsum(m)


KBIG = np.int64(1) << 62          # kidx padding (keeps the kept-candidate index buffer sorted)


def _vertex_gen(g, P, Rj, U, nvert):
    a = jnp.minimum(g // nvert, P.shape[0] - 1); vv = g % nvert
    return P[a] + Rj[a][:, None] * U[vv], a, vv


def _arc_gen(g, Cum, Mm, Cen, Rho, E1, E2):
    p = jnp.minimum(jnp.searchsorted(Cum, g, side="right"), Cum.shape[0] - 1)
    k = g - (Cum[p] - Mm[p])
    th = 2 * np.pi * k / jnp.maximum(Mm[p], 1)
    return Cen[p] + Rho[p][:, None] * (jnp.cos(th)[:, None] * E1[p] + jnp.sin(th)[:, None] * E2[p]), p, k


def _vertex_chunk(s, NV, P, Rj, valid, U, lo, cell, bound, scid, order, CH, nvert, K, knn):
    g = s + jnp.arange(CH, dtype=jnp.int64)
    ok = g < NV
    pts, a, vv = _vertex_gen(g, P, Rj, U, nvert)
    cand, occ = _cand(pts, lo, cell, scid, order, K)
    b, ball = _burial(pts, jnp.stack([a, a], 1), cand, P, Rj, bound, knn)
    return _finish(pts, ok & valid[a], b, ball, occ, g)


def _arc_chunk(s, NA, Cum, Mm, Cen, Rho, E1, E2, Pi, Pj, P, Rj, lo, cell, bound, scid, order, CH, K, knn):
    g = s + jnp.arange(CH, dtype=jnp.int64)
    ok = g < NA
    pts, p, k = _arc_gen(g, Cum, Mm, Cen, Rho, E1, E2)
    cand, occ = _cand(pts, lo, cell, scid, order, K)
    b, ball = _burial(pts, jnp.stack([Pi[p], Pj[p]], 1), cand, P, Rj, bound, knn)
    return _finish(pts, ok, b, ball, occ, g)


def _finish(pts, valid, b_knn, b_all, occ, g):
    keep = valid & ~b_knn
    n = jnp.sum(keep)
    sel = jnp.nonzero(keep, size=pts.shape[0], fill_value=0)[0]
    first = (jnp.arange(pts.shape[0]) < n)
    out = jnp.where(first[:, None], pts[sel].astype(jnp.float32), jnp.float32(RC.FAR))
    return out, n, jnp.sum(valid & (b_knn != b_all)), occ, jnp.where(first, g[sel], KBIG)


def _near_burial(pts, own, near, lo, cell, scid, order, P, Rj, bound, K, knn, SC):
    """SPEED lever 3: burial of the rows with `near` only (compacted, sub-chunks of SC rows, dynamic trip count); other
    rows: (False, False). Per row the same _cand / _burial as the full chunk."""
    CH = pts.shape[0]
    nn = jnp.sum(near)
    idx = jnp.nonzero(near, size=CH, fill_value=0)[0]

    def body(j, c):
        bk, ba, occ = c
        sub = jax.lax.dynamic_slice_in_dim(idx, j * SC, SC)
        live = (j * SC + jnp.arange(SC)) < nn
        ps = pts[sub]
        cand, oc = _cand(ps, lo, cell, scid, order, K)
        b, ball = _burial(ps, own[sub], cand, P, Rj, bound, knn)
        tgt = jnp.where(live, sub, CH)
        return bk.at[tgt].set(b, mode="drop"), ba.at[tgt].set(ball, mode="drop"), jnp.maximum(occ, oc.astype(occ.dtype))
    z = jnp.zeros((), jnp.int64)
    return jax.lax.fori_loop(z, (nn + SC - 1) // SC, body, (jnp.zeros(CH, bool), jnp.zeros(CH, bool), z))


def _in_sorted(kidx, q):
    pos = jnp.minimum(jnp.searchsorted(kidx, q, side="left"), kidx.shape[0] - 1)
    return kidx[pos] == q


@partial(jax.jit, static_argnames=("CH", "nvert", "K", "knn"))
def geo_vertex(s, NV, P, Rj, valid, U, lo, cell, bound, scid, order, CH=RC.GEO_CHUNK, nvert=RC.NVERT, K=RC.GEO_K, knn=24):
    """one vertex chunk (M6 host-loop program; kept for the profiling copies, not used by geometry_state)."""
    return _vertex_chunk(s, NV, P, Rj, valid, U, lo, cell, bound, scid, order, CH, nvert, K, knn)[:4]


@partial(jax.jit, static_argnames=("CH", "K", "knn"))
def geo_arc(s, NA, Cum, Mm, Cen, Rho, E1, E2, Pi, Pj, P, Rj, lo, cell, bound, scid, order, CH=RC.GEO_CHUNK, K=RC.GEO_K, knn=24):
    """one arc chunk (M6 host-loop program; kept for the profiling copies, not used by geometry_state)."""
    return _arc_chunk(s, NA, Cum, Mm, Cen, Rho, E1, E2, Pi, Pj, P, Rj, lo, cell, bound, scid, order, CH, K, knn)[:4]


def _probe_loops(NV, NA, CH, NPmax, vchunk, achunk):
    """SPEED lever 1: every vertex chunk then every arc chunk in two device loops (dynamic trip counts), each chunk
    compacted and written at the running offset of the class buffer [NPmax + CH, 3] exactly as the M6 host loop did
    (geo_pack), so the probe order and values are unchanged. The write is unconditional at min(off, NPmax) (M6 skipped
    it once off + cnt > NPmax; both then raise the probe overflow, so a non-overflowing state gets the same buffer).
    Also kept-candidate indices kidx [NPmax + CH] (ascending, KBIG padded) for lever 3."""
    i64 = jnp.int64
    z = jnp.zeros((), i64)

    def put(carry, out, base):
        buf, kid, off, occ, ntr = carry
        pts, cnt, tr, oc, kk = out
        kk = jnp.where(kk == KBIG, KBIG, kk + base)          # arc candidates are numbered after the NV vertices
        o = jnp.minimum(off, NPmax)
        buf = jax.lax.dynamic_update_slice(buf, pts, (o, z))
        kid = jax.lax.dynamic_update_slice(kid, kk, (o,))
        return buf, kid, off + cnt.astype(i64), jnp.maximum(occ, oc.astype(i64)), ntr + tr.astype(i64)
    init = (jnp.full((NPmax + CH, 3), jnp.float32(RC.FAR), jnp.float32), jnp.full((NPmax + CH,), KBIG, i64), z, z, z)
    carry = jax.lax.fori_loop(z, (NV + CH - 1) // CH, lambda k, c: put(c, vchunk(k * CH), z), init)
    return jax.lax.fori_loop(z, (NA + CH - 1) // CH, lambda k, c: put(c, achunk(k * CH), NV), carry)


@partial(jax.jit, static_argnames=("CH", "nvert", "K", "knn", "NPmax"))
def geo_probes(NV, P, Rj, valid, U, lo, cell, bound, scid, order, cum, m, cen, rho, e1, e2, Pi, Pj,
               CH=RC.GEO_CHUNK, nvert=RC.NVERT, K=RC.GEO_K, knn=24, NPmax=None):
    """whole-molecule probes (chunk bodies _vertex_chunk / _arc_chunk, as M6). Returns buf, kidx, n_probes, max cell
    occupancy, knn changes, NA."""
    NA = cum[-1]
    out = _probe_loops(NV, NA, CH, NPmax,
                       lambda s: _vertex_chunk(s, NV, P, Rj, valid, U, lo, cell, bound, scid, order, CH, nvert, K, knn),
                       lambda s: _arc_chunk(s, NA, cum, m, cen, rho, e1, e2, Pi, Pj, P, Rj, lo, cell, bound, scid, order, CH, K, knn))
    return out + (NA,)


@partial(jax.jit, static_argnames=("CH", "nvert", "K", "knn", "NPmax", "SC"))
def geo_probes_reuse(NV, P, Rj, valid, U, lo, cell, bound, scid, order, cum, m, cen, rho, e1, e2, Pi, Pj,
                     amap, anear, pmap, pnear, cumP, mP, kidxP, NVP, CH=RC.GEO_CHUNK, nvert=RC.NVERT, K=RC.GEO_K, knn=24,
                     NPmax=None, SC=RC.GEO_CHUNK // 8):
    """SPEED lever 3: geo_probes of a chain state (A or B) of a triple, reusing the burial decisions of its AB state.
    Candidates are generated exactly as from scratch (own atoms, own pair list, own order). A candidate whose generating
    atom (vertex) or one of whose two atoms (arc) is farther than R + bound from every partner atom has no partner atom
    among its burial candidates, so its decision equals that of the same candidate in AB: kept iff its AB candidate index
    (amap[a] * nvert + v, or NVP + AB arc offset of pair pmap[p] + k) is in AB's kept list kidxP. Only the other ('near')
    candidates get _cand + _burial (with this state's own cell list). Same buffer / order / values as geo_probes.
    knn-change and occupancy counters cover the near candidates only."""
    NA = cum[-1]

    def vchunk(s):
        g = s + jnp.arange(CH, dtype=jnp.int64)
        ok = (g < NV) & valid[jnp.minimum(g // nvert, P.shape[0] - 1)]
        pts, a, vv = _vertex_gen(g, P, Rj, U, nvert)
        near = anear[a]
        bk, ba, occ = _near_burial(pts, jnp.stack([a, a], 1), near & ok, lo, cell, scid, order, P, Rj, bound, K, knn, SC)
        far_b = ~_in_sorted(kidxP, amap[a] * nvert + vv)
        bk = jnp.where(near, bk, far_b); ba = jnp.where(near, ba, far_b)
        return _finish(pts, ok, bk, ba, occ, g)

    def achunk(s):
        g = s + jnp.arange(CH, dtype=jnp.int64)
        ok = g < NA
        pts, p, k = _arc_gen(g, cum, m, cen, rho, e1, e2)
        near = pnear[p]
        bk, ba, occ = _near_burial(pts, jnp.stack([Pi[p], Pj[p]], 1), near & ok, lo, cell, scid, order, P, Rj, bound, K, knn, SC)
        q = pmap[p]
        far_b = ~_in_sorted(kidxP, NVP + cumP[q] - mP[q] + k)
        bk = jnp.where(near, bk, far_b); ba = jnp.where(near, ba, far_b)
        return _finish(pts, ok, bk, ba, occ, g)
    out = _probe_loops(NV, NA, CH, NPmax, vchunk, achunk)
    return out + (NA,)


@partial(jax.jit, donate_argnums=(0,))
def geo_pack(buf, pts, off):
    """write a compacted chunk at offset `off` of the class buffer (the chunk's FAR tail is overwritten by the next)."""
    return jax.lax.dynamic_update_slice(buf, pts, (off, jnp.int64(0)))


@partial(jax.jit, static_argnames=("NP",))
def _cell_hash(P, valid, lo, cell, NP=None):
    return _hash_cells(P, valid, lo, cell)


def surface_atoms(pos_all, rad_all, c):
    """host: radius > 0 atoms in original order, padded to NA_C[c] at FAR with radius 0."""
    keep = np.asarray(rad_all) > 0
    p = np.asarray(pos_all, np.float32)[keep]; r = np.asarray(rad_all, np.float32)[keep]
    n = len(p); N = RC.NA_C[c]
    if n > N:
        raise OverflowError(f"{n} surface atoms > class {c} ({N})")
    P = np.full((N, 3), RC.FAR, np.float32); R = np.zeros(N, np.float32)
    P[:n] = p; R[:n] = r
    return P, R, n


def geometry_state(pos_all, rad_all, c, rp=RC.RADPRB, nvert=RC.NVERT, arc_step=RC.ARC_STEP, CH=RC.GEO_CHUNK, K=RC.GEO_K,
                   parent: Optional[GeoReuse] = None, want_reuse: bool = False):
    """Whole-molecule probe centres for one state in class c. Returns GeoState (probes on device, FAR-padded).
    parent: the GeoReuse of this chain state's AB state (lever 3; same probes, fewer burial tests; falls back to the
    from-scratch path, flag geo_reuse_fallback, if the triple does not match). want_reuse: attach GeoReuse (AB states)."""
    from scipy.spatial import cKDTree
    Pf, Rf, n = surface_atoms(pos_all, rad_all, c)
    N = RC.NA_C[c]; NPmax = RC.NP_C[c]; NPAIR = RC.PAIRS_PER_ATOM * N
    flags = {}
    pos = Pf[:n].astype(np.float64); R = Rf[:n].astype(np.float64) + rp
    Rmax = float(R.max()); bound = Rmax + 1e-3
    pairs = cKDTree(pos).query_pairs(2 * Rmax, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(pos[j] - pos[i], axis=1)
    ok = (d < R[i] + R[j]) & (d > np.abs(R[i] - R[j])) & (d > 1e-6)
    i, j, d = i[ok], j[ok], d[ok]
    flags["pairs"] = int(len(i)); flags["pairs_cap"] = NPAIR
    if len(i) > NPAIR:
        raise OverflowError(f"pairs {len(i)} > {NPAIR}")
    pi = np.zeros(NPAIR, np.int64); pj = np.zeros(NPAIR, np.int64); pd = np.ones(NPAIR); pv = np.zeros(NPAIR, bool)
    pi[:len(i)] = i; pj[:len(i)] = j; pd[:len(i)] = d; pv[:len(i)] = True
    cell = bound + 1e-6
    lo = pos.min(0) - cell
    Pd = np.full((N, 3), RC.FAR, np.float64); Pd[:n] = pos
    Rd = np.zeros(N, np.float64); Rd[:n] = R
    valid = np.zeros(N, bool); valid[:n] = True
    plan = None
    if parent is not None:
        plan = _reuse_plan(parent, Pf[:n], Rf[:n], pos, R, bound, i, j, N, NPAIR, nvert)
        if isinstance(plan, int):
            flags["reuse"] = 0; flags["reuse_fallback"] = plan; plan = None
        else:
            flags["reuse"] = 1; flags["reuse_fallback"] = 0
            flags["reuse_near_atoms"] = int(plan["anear"][:n].sum()); flags["reuse_near_pairs"] = int(plan["pnear"][:len(i)].sum())
            flags["reuse_unmatched_atoms"] = plan["n_unmatched"]
    hold = dict(pos=Pf[:n].copy(), rad=Rf[:n].copy(), bound=bound, pi=i, pj=j) if want_reuse else None
    with x64():
        try:
            return _geometry_state_x64(Pf, Rf, n, c, Pd, Rd, valid, lo, cell, bound, pi, pj, pd, pv, flags, nvert, arc_step, CH, K,
                                       NPmax, parent, plan, hold)
        except CellOverflow as e:            # SPEED lever 2: K = GEO_K (16) too small -> same class with XL_GEO_K (logged)
            if K >= RC.XL_GEO_K:
                raise
            flags["geo_k_retry"] = 1; flags["geo_k_retry_reason"] = str(e)
            return _geometry_state_x64(Pf, Rf, n, c, Pd, Rd, valid, lo, cell, bound, pi, pj, pd, pv, flags, nvert, arc_step, CH,
                                       RC.XL_GEO_K, NPmax, parent, plan, hold)


REUSE_FALLBACK = {1: "bound (largest SAS radius) differs from AB", 5: "nvert differs"}


def _reuse_plan(parent: GeoReuse, pos_f, rad_f, pos, R, bound, i, j, N, NPAIR, nvert):
    """host: map this chain state's surface atoms and SAS pairs onto its AB state; near flags. int = fallback code.
    Chain states are not exact atom subsets of AB (a few hydrogens are re-placed, ~1 A; atom order can differ), so:
    an atom is reusable only if an AB atom has bit-identical position + radius; perturbers = AB atoms not matched (the
    partner chain and AB's versions of moved atoms) + this state's unmatched atoms; a candidate is reused only if its
    generator maps exactly (vertex: matched atom; arc: matched pair with the same i < j orientation) and its generating
    atom is farther than R + bound from every perturber. Everything else is 'near' (recomputed)."""
    from scipy.spatial import cKDTree
    if parent.nvert != nvert:
        return 5
    if parent.bound != bound:
        return 1
    nP = len(parent.pos)
    rows = lambda p_, r_: [x.tobytes() for x in np.ascontiguousarray(np.concatenate([p_, r_[:, None]], 1), np.float32)]
    look = {k: t for t, k in enumerate(rows(parent.pos, parent.rad))}
    amap = np.array([look.get(k, -1) for k in rows(pos_f, rad_f)], np.int64)
    used = np.zeros(nP, bool); used[amap[amap >= 0]] = True
    pert = np.concatenate([parent.pos[~used].astype(np.float64), pos[amap < 0]], 0)
    if len(pert):
        dpart, _ = cKDTree(pert).query(pos, k=1)
    else:
        dpart = np.full(len(pos), np.inf)
    anear = (dpart < R + bound + 1e-3) | (amap < 0)
    kP = parent.pi.astype(np.int64) * nP + parent.pj.astype(np.int64)
    sk = np.argsort(kP, kind="stable"); kPs = kP[sk]
    both = (amap[i] >= 0) & (amap[j] >= 0)
    kA = np.where(both, amap[i] * nP + amap[j], -1)
    loc = np.minimum(np.searchsorted(kPs, kA), max(len(kP) - 1, 0))
    found = both & (len(kP) > 0) & (kPs[loc] == kA) if len(kA) else np.zeros(0, bool)
    pmap = np.where(found, sk[loc], 0) if len(kA) else np.zeros(0, np.int64)
    pnear = (anear[i] & anear[j]) | ~found
    A = np.zeros(N, np.int64); A[:len(amap)] = np.maximum(amap, 0)
    AN = np.ones(N, bool); AN[:len(anear)] = anear
    Pm = np.zeros(NPAIR, np.int64); Pm[:len(pmap)] = pmap
    PN = np.ones(NPAIR, bool); PN[:len(pnear)] = pnear
    return dict(amap=A, anear=AN, pmap=Pm, pnear=PN, n_unmatched=int((amap < 0).sum()))


class CellOverflow(OverflowError):
    """geometry hashed-cell occupancy > K (burial candidates truncated)."""


def _geometry_state_x64(Pf, Rf, n, c, Pd, Rd, valid, lo, cell, bound, pi, pj, pd, pv, flags, nvert, arc_step, CH, K, NPmax,
                        parent=None, plan=None, hold=None):
    f64 = jnp.float64
    P_ = RC.dev(Pd, np.float64); R_ = RC.dev(Rd, np.float64); V_ = jax.device_put(valid)
    lo_ = RC.dev(lo, np.float64); cell_ = RC.dev(cell, np.float64); b_ = RC.dev(bound, np.float64)
    scid, order = _cell_hash(P_, V_, lo_, cell_)
    U = RC.dev(fib_sphere(nvert), np.float64)
    cen, rho, e1, e2, m, cum = geo_pairs(P_, R_, RC.dev(pi, np.int64), RC.dev(pj, np.int64), RC.dev(pd, np.float64), jax.device_put(pv), arc_step=arc_step)
    NV = n * nvert
    args = (RC.dev(NV, np.int64), P_, R_, V_, U, lo_, cell_, b_, scid, order, cum, m, cen, rho, e1, e2, RC.dev(pi, np.int64),
            RC.dev(pj, np.int64))
    if plan is None:
        buf, kidx, off_, occ_, ntr_, NA_ = geo_probes(*args, CH=CH, nvert=nvert, K=K, NPmax=NPmax)
    else:
        buf, kidx, off_, occ_, ntr_, NA_ = geo_probes_reuse(*args, RC.dev(plan["amap"], np.int64), jax.device_put(plan["anear"]),
                                                            RC.dev(plan["pmap"], np.int64), jax.device_put(plan["pnear"]),
                                                            parent.cum, parent.m, parent.kidx, RC.dev(len(parent.pos) * nvert, np.int64), CH=CH, nvert=nvert, K=K, NPmax=NPmax)
    off, occmax, ntr, NA = (int(x) for x in jax.device_get((off_, occ_, ntr_, NA_)))      # the one host sync per state
    flags.update(n_probes=off, probes_cap=NPmax, geo_cell_occ=occmax, geo_K=K, knn_truncation_changed=ntr,
                 n_vertex_pts=NV, n_arc_pts=NA)
    if off > NPmax:
        raise OverflowError(f"probes {off} > {NPmax}")
    if occmax > K:
        raise CellOverflow(f"geometry cell occupancy {occmax} > K {K}")
    reuse = GeoReuse(cum=cum, m=m, kidx=kidx, nvert=nvert, **hold) if hold is not None else None
    return GeoState(pos=Pf, rad=Rf, n_atoms=n, probes=buf, n_probes=off, c=c, flags=flags, reuse=reuse)


# ------------------------------------------------------------------------------------------------ model compounds
def _site_probes(sp, sR, sv, U, nvert, MXs, OUT, knn, arc_step):
    """v2 maps2._site_probes for one model compound (padded to MA atoms, sv = valid)."""
    na = sp.shape[0]
    Rmax = jnp.max(jnp.where(sv, sR, -jnp.inf)); bound = Rmax + 1e-3
    cand_all = jnp.where(sv, jnp.arange(na), -1)
    a = jnp.repeat(jnp.arange(na), nvert); vv = jnp.tile(jnp.arange(nvert), na)
    vpts = sp[a] + sR[a][:, None] * U[vv]
    vvalid = sv[a]
    iu, ju = np.triu_indices(na, 1)
    i = jnp.asarray(iu); j = jnp.asarray(ju)
    d = jnp.linalg.norm(sp[j] - sp[i], axis=1)
    pok = sv[i] & sv[j] & (d < sR[i] + sR[j]) & (d > jnp.abs(sR[i] - sR[j])) & (d > 1e-6)
    ds = jnp.where(pok, d, 1.0)
    cen, rho, e1, e2, m = _pair_geom(sp, sR, i, j, ds, arc_step)
    k = jnp.arange(MXs)
    th = 2 * np.pi * k[None, :] / m[:, None]
    apts = (cen[:, None, :] + rho[:, None, None] * (jnp.cos(th)[..., None] * e1[:, None, :] + jnp.sin(th)[..., None] * e2[:, None, :])).reshape(-1, 3)
    avalid = (pok[:, None] & (k[None, :] < m[:, None])).reshape(-1)
    aown = jnp.stack([jnp.repeat(i, MXs), jnp.repeat(j, MXs)], 1)
    pts = jnp.concatenate([vpts, apts]); valid = jnp.concatenate([vvalid, avalid])
    own = jnp.concatenate([jnp.stack([a, a], 1), aown])
    b, ball = _burial(pts, own, jnp.broadcast_to(cand_all, (pts.shape[0], na)), sp, sR, bound, knn)
    keep = valid & ~b
    sel = jnp.nonzero(keep, size=OUT, fill_value=0)[0]
    n = jnp.sum(keep)
    return (jnp.where((jnp.arange(OUT) < n)[:, None], pts[sel].astype(jnp.float32), jnp.float32(RC.FAR)), n,
            jnp.max(jnp.where(pok, m, 0)), jnp.sum(valid & (b != ball)))


@partial(jax.jit, static_argnames=("nvert", "MXs", "OUT", "knn", "arc_step"))
def geometry_sites(sp, sR, sv, U, nvert=RC.NVERT, MXs=MX, OUT=RC.MP_MODEL, knn=24, arc_step=RC.ARC_STEP):
    """Model-compound probe centres for a batch of B sites. sp [B,MA,3] f64, sR [B,MA] f64 (= r + 1.4), sv [B,MA] bool.
    Returns probes [B,OUT,3] f32 (FAR-padded), counts [B], max arc count [B] (overflow if > MXs), knn changes [B]."""
    return jax.vmap(lambda a, b, c: _site_probes(a, b, c, U, nvert, MXs, OUT, knn, arc_step))(sp, sR, sv)
