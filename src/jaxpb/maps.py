"""jaxpb.maps -- DelPhi dielectric / ion / boundary maps for the coarse, focused and model-compound grids.

Numerics are jaxpb-pilot v1 build_maps / v2 maps2.build_maps2 unchanged (binary epsilon on grid midpoints from the
probe-sphere molecular surface; re-entrant fill by probe scatter; ion map at r + 2 A; boundary points projected onto the
molecular surface). Only the shape discipline changed: every size is a static argument taken from records.py (Ka, Kp,
NS, NB, PB, n), atoms/probes arrive FAR-padded at class sizes, and nothing depends on the data's shape.
All arithmetic is float32 with x64 disabled (as in the prototypes).

maps_batch(...)  -- one jitted, vmapped program per (class c, Sizes): coarse + focused + model maps for B sites.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from . import records as RC

RP = RC.RADPRB
EXRAD = RC.EXRAD
PCELL = RP + 0.01      # probe cell size
PREACH = 1


@dataclass(frozen=True)
class Sizes:
    """Static capacities of one map/PB program variant (normal or XL)."""
    Ka: int = RC.KA
    Kp: int = RC.KP
    NSc: int = RC.SZ["c"]["NS"]
    NBc: int = RC.SZ["c"]["NB"]
    NSf: int = RC.SZ["f"]["NS"]
    NBf: int = RC.SZ["f"]["NB"]
    NSs: int = RC.SZ["s"]["NS"]
    NBs: int = RC.SZ["s"]["NB"]
    PBf: int = RC.FINE_PROBES
    PBs: int = RC.MP_MODEL

    @staticmethod
    def xl():
        return Sizes(Ka=RC.XL_KA, Kp=RC.XL_KP, NSc=RC.XL_SZ["c"]["NS"], NBc=RC.XL_SZ["c"]["NB"], NSf=RC.XL_SZ["f"]["NS"],
                     NBf=RC.XL_SZ["f"]["NB"], NSs=RC.XL_SZ["s"]["NS"], NBs=RC.XL_SZ["s"]["NB"], PBf=RC.XL_FINE_PROBES,
                     PBs=RC.XL_MP_MODEL)


def ncells(h, n, kind):
    L = (n - 1) * h
    return int(np.ceil((L + 2 * 4.5) / 4.01)) + 1 if kind == "a" else int(np.ceil((L + 2 * 1.9) / PCELL)) + 1


def grid_coords(acent, h, n):
    ax = (jnp.arange(n) - (n - 1) / 2) * h
    return acent[0] + ax, acent[1] + ax, acent[2] + ax


def build_cells(P, w, lo, ncell, cell, K):
    """Bin points P (n,3) into a dense (ncell^3, K) index table over lo + [0, ncell*cell). Returns table, max occupancy."""
    c = jnp.floor((P - lo) / cell).astype(jnp.int32)
    ok = jnp.all((c >= 0) & (c < ncell), axis=1)
    cid = jnp.where(ok, (c[:, 0] * ncell + c[:, 1]) * ncell + c[:, 2], ncell ** 3)
    order = jnp.argsort(cid); sc = cid[order]
    start = jnp.searchsorted(sc, jnp.arange(ncell ** 3), side="left")
    stop = jnp.searchsorted(sc, jnp.arange(ncell ** 3), side="right")
    kk = jnp.arange(K)
    idx = start[:, None] + kk[None, :]
    table = jnp.where(idx < stop[:, None], order[jnp.minimum(idx, P.shape[0] - 1)], -1)
    return table, jnp.max(stop - start)


def query_min(Q, P, w, table, lo, ncell, cell, chunk=None, reach=1):
    """min_j |q - p_j| - w_j over points in the (2*reach+1)^3 neighbouring cells; (dmin, argmin) (inf / -1 if none)."""
    rr = range(-reach, reach + 1)
    offs = jnp.array([(a, b, c) for a in rr for b in rr for c in rr], jnp.int32)
    chunk = chunk or max(1024, (1 << 22) // (offs.shape[0] * table.shape[1]))
    nq = Q.shape[0]; pad = (-nq) % chunk
    Qp = jnp.concatenate([Q, jnp.zeros((pad, 3), Q.dtype)]).reshape(-1, chunk, 3)

    dm, am = jax.lax.map(lambda q: _q_one(q, P, w, table, lo, ncell, cell, offs), Qp)
    return dm.reshape(-1)[:nq], am.reshape(-1)[:nq]


def _offs(reach):
    rr = range(-reach, reach + 1)
    return jnp.array([(a, b, c) for a in rr for b in rr for c in rr], jnp.int32)


def _q_chunk(reach, K):
    return max(1024, (1 << 22) // ((2 * reach + 1) ** 3 * K))


def _pick(q, P, w, idx):
    """nearest candidate (min |q - p| - w; first in candidate order on ties) of each query row."""
    pp = P[jnp.maximum(idx, 0)]
    d = jnp.linalg.norm(q[:, None, :] - pp, axis=2) - w[jnp.maximum(idx, 0)]
    d = jnp.where(idx >= 0, d, jnp.inf)
    a = jnp.argmin(d, axis=1)
    return jnp.take_along_axis(d, a[:, None], 1)[:, 0], jnp.take_along_axis(idx, a[:, None], 1)[:, 0]


def _q_one(q, P, w, table, lo, ncell, cell, offs):
    """query_min on one chunk of rows (dense cell table)."""
    c = jnp.floor((q - lo) / cell).astype(jnp.int32)
    cc = c[:, None, :] + offs[None]
    ok = jnp.all((cc >= 0) & (cc < ncell), axis=2)
    cid = jnp.where(ok, (cc[..., 0] * ncell + cc[..., 1]) * ncell + cc[..., 2], 0)
    idx = jnp.where(ok[..., None], table[cid], -1).reshape(q.shape[0], -1)
    return _pick(q, P, w, idx)


def _q_one_hashed(q, P, w, sc, order, lo, ncell, cell, offs, K):
    """SPEED lever 6: _q_one on a hashed cell list (sorted cell ids sc + permutation, no dense ncell^3 x K table). The
    candidate list is the dense table's row for row: slot k of cell cid = order[start(cid) + k] if < stop(cid) else -1,
    with start/stop = searchsorted of the same stable sort, so distances, minima and tie-breaks are identical."""
    c = jnp.floor((q - lo) / cell).astype(jnp.int32)
    cc = c[:, None, :] + offs[None]
    ok = jnp.all((cc >= 0) & (cc < ncell), axis=2)
    cid = jnp.where(ok, (cc[..., 0] * ncell + cc[..., 1]) * ncell + cc[..., 2], 0)
    st = jnp.searchsorted(sc, cid, side="left"); en = jnp.searchsorted(sc, cid, side="right")
    kk = st[..., None] + jnp.arange(K)
    idx = jnp.where(ok[..., None] & (kk < en[..., None]), order[jnp.minimum(kk, order.shape[0] - 1)], -1).reshape(q.shape[0], -1)
    return _pick(q, P, w, idx)


def _b_one(q, P, w):
    """brute_min on one chunk of rows."""
    d = jnp.linalg.norm(q[:, None, :] - P[None], axis=2) - w[None]
    a = jnp.argmin(d, axis=1)
    return jnp.take_along_axis(d, a[:, None], 1)[:, 0], a.astype(jnp.int32)


def reach_offsets(h, rp=RP):
    Rr = int(np.ceil(rp / h)) + 1
    o = np.array([(a, b, c) for a in range(-Rr, Rr + 2) for b in range(-Rr, Rr + 2) for c in range(-Rr, Rr + 2)])
    lo = np.where(o > 1, (o - 1), np.where(o < 0, -o, 0)) * h
    return o[(lo ** 2).sum(1) < rp ** 2 + 1e-9]


def region_probes(probes, acent, h, n, PB, rp=RP):
    L = (n - 1) * h
    inr = jnp.all((probes > acent - L / 2 - rp - 1.0) & (probes < acent + L / 2 + rp + 1.0), axis=1)
    pi = jnp.nonzero(inr, size=PB, fill_value=-1)[0]
    return probes[jnp.maximum(pi, 0)], pi >= 0, jnp.sum(inr)


def reach_mask(p, pvalid, acent, h, n, ax, hstat, rp=RP, budget=1 << 22):
    dt = p.dtype
    off = jnp.zeros(3, dt).at[ax].set(h / 2)
    O = jnp.asarray(reach_offsets(hstat, rp), jnp.int32)
    x, y, z = grid_coords(acent, h, n)
    ch = max(256, budget // O.shape[0]); npb = p.shape[0]; pad = (-npb) % ch
    pp = jnp.concatenate([p, jnp.zeros((pad, 3), dt)]).reshape(-1, ch, 3)
    vv = jnp.concatenate([pvalid, jnp.zeros(pad, bool)]).reshape(-1, ch)

    def body(c, mask):
        pc = pp[c]; ok0 = vv[c]
        i0 = jnp.floor((pc - acent - off) / h + (n - 1) / 2).astype(jnp.int32)
        idx = i0[:, None, :] + O[None]
        ok = jnp.all((idx >= 0) & (idx < n), axis=2) & ok0[:, None]
        ic = jnp.clip(idx, 0, n - 1)
        M = jnp.stack([x[ic[..., 0]] + off[0], y[ic[..., 1]] + off[1], z[ic[..., 2]] + off[2]], -1)
        d = jnp.linalg.norm(M - pc[:, None, :], axis=-1) - jnp.zeros((), dt)
        hit = ok & (d < rp)
        flat = jnp.where(hit, (ic[..., 0] * n + ic[..., 1]) * n + ic[..., 2], n ** 3)
        return mask.at[flat.reshape(-1)].set(True, mode="drop")

    return jax.lax.fori_loop(0, pp.shape[0], body, jnp.zeros(n ** 3, jnp.bool_))


def brute_min(Q, P, w, chunk=65536):
    nq = Q.shape[0]; pad = (-nq) % chunk
    Qp = jnp.concatenate([Q, jnp.zeros((pad, 3), Q.dtype)]).reshape(-1, chunk, 3)

    dm, am = jax.lax.map(lambda q: _b_one(q, P, w), Qp)
    return dm.reshape(-1)[:nq], am.reshape(-1)[:nq]


def build_maps(pos, rad, probes, acent, h, n, Ka, Kp, ncell_a, ncell_p, NS, NB, PB, brute, hstat, rp=RP, exrad=EXRAD):
    """v2 maps2.build_maps2 (identical numerics). Outputs ex, ey, ez, ion (bool n^3), bidx (NB,), surf (NB,3) + counters."""
    dt = pos.dtype
    rmax = 2.0; L = (n - 1) * h
    lo = acent - L / 2 - (rmax + exrad + 0.5); ca = rmax + exrad + 0.01
    if brute:
        qmin = lambda Q, w: brute_min(Q, pos, w)
        ov_a = jnp.asarray(0)
    else:
        tab_a, ov_a = build_cells(pos, rad, lo, ncell_a, ca, Ka)
        qmin = lambda Q, w: query_min(Q, pos, w, tab_a, lo, ncell_a, ca)
    cp = PCELL
    lop = acent - L / 2 - (rp + 0.5)
    zp = jnp.zeros(probes.shape[0], dt)
    tab_p, ov_p = build_cells(probes, zp, lop, ncell_p, cp, Kp)
    preg, pval, npreg = region_probes(probes, acent, h, n, PB, rp)
    x, y, z = grid_coords(acent, h, n)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    G = jnp.stack([X, Y, Z], -1).reshape(-1, 3)
    out = {}
    da_pts, ia_pts = qmin(G, rad)
    out["ion"] = (da_pts >= exrad).reshape(n, n, n)
    ov_s = 0
    for ax, name in enumerate(("ex", "ey", "ez")):
        M = G + jnp.zeros(3, dt).at[ax].set(h / 2)
        dv, iv = qmin(M, rad)
        shell = (dv >= 0) & (dv < rp)
        ov_s = jnp.maximum(ov_s, jnp.sum(shell))
        si = jnp.nonzero(shell, size=NS, fill_value=0)[0]
        Ms = M[si]
        ia = jnp.maximum(iv[si], 0); u = Ms - pos[ia]; u = u / jnp.maximum(jnp.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        cpr = pos[ia] + (rad[ia] + rp)[:, None] * u
        dcp, _ = qmin(cpr, rad + rp)
        contact = dcp >= -1e-5
        need = jnp.zeros_like(shell).at[si].max(~contact) & shell
        reach = jnp.zeros(M.shape[0], bool).at[si].max(contact) | (need & reach_mask(preg, pval, acent, h, n, ax, hstat, rp))
        solvent = (dv >= rp) | (shell & reach)
        out[name] = (~solvent).reshape(n, n, n)
    ex, ey, ez = out["ex"], out["ey"], out["ez"]
    nb = [ex, jnp.roll(ex, 1, 0), ey, jnp.roll(ey, 1, 1), ez, jnp.roll(ez, 1, 2)]
    s = sum(a.astype(jnp.int32) for a in nb)
    bnd = (s > 0) & (s < 6)
    bnd = bnd.at[0].set(False).at[-1].set(False).at[:, 0].set(False).at[:, -1].set(False).at[:, :, 0].set(False).at[:, :, -1].set(False)
    bflat = bnd.reshape(-1)
    nbd = jnp.sum(bflat)
    bidx = jnp.nonzero(bflat, size=NB, fill_value=-1)[0]
    Gb = G[jnp.maximum(bidx, 0)]
    ia = jnp.maximum(ia_pts[jnp.maximum(bidx, 0)], 0)
    v = Gb - pos[ia]; nv = jnp.linalg.norm(v, axis=1, keepdims=True)
    sa = pos[ia] + rad[ia][:, None] * v / jnp.maximum(nv, 1e-6)
    cpr = pos[ia] + (rad[ia] + rp)[:, None] * v / jnp.maximum(nv, 1e-6)
    dcp, _ = qmin(cpr, rad + rp)
    needb = (dcp < -1e-5) & (bidx >= 0)
    nbproj = jnp.sum(needb)
    bi2 = jnp.nonzero(needb, size=NB, fill_value=0)[0]
    _, ip2 = query_min(Gb[bi2], probes, zp, tab_p, lop, ncell_p, cp, reach=PREACH)
    ip = jnp.full(Gb.shape[0], -1, ip2.dtype).at[bi2].set(jnp.where(needb[bi2], ip2, -1))
    ipc = probes[jnp.maximum(ip, 0)]
    vp = Gb - ipc; nvp = jnp.linalg.norm(vp, axis=1, keepdims=True)
    sp = ipc + rp * vp / jnp.maximum(nvp, 1e-6)
    use_atom = (dcp >= -1e-5) | (ip < 0)
    res = dict(ex=ex, ey=ey, ez=ez, ion=out["ion"], bidx=bidx, surf=jnp.where(use_atom[:, None], sa, sp))
    # counters: value > capacity means overflow (ov_a vs Ka, ov_p vs Kp; the compaction ones are reported as excess)
    ctr = jnp.stack([ov_a.astype(jnp.int32), ov_p.astype(jnp.int32), (nbproj - NB).astype(jnp.int32), (ov_s - NS).astype(jnp.int32),
                     (nbd - NB).astype(jnp.int32), (npreg - PB).astype(jnp.int32), nbd.astype(jnp.int32)])
    return res, ctr


CTR_NAMES = ("cell_atoms", "cell_probes", "bproj_excess", "shell_excess", "bnd_excess", "preg_excess", "n_bnd")


def overflow(ctr, sz: Sizes):
    """host check of a [..., 3 grids, 7] counter array -> dict of overflow booleans (any site)."""
    c = np.asarray(ctr).reshape(-1, 3, len(CTR_NAMES))
    ka = np.array([sz.Ka, sz.Ka, 0])     # model grid uses brute force (no atom cells)
    return {"cell_atoms": bool((c[:, :, 0] > ka).any()), "cell_probes": bool((c[:, :, 1] > sz.Kp).any()),
            "compaction": bool((c[:, :, 2:6] > 0).any())}


# ------------------------------------------------------------------------------------------------ SPEED (2026-11-20)
# build_maps_b = build_maps for a batch of B sites written batch-explicitly: every per-site stage is jax.vmap'ed (as
# maps_batch's vmap did) and every point query runs in a chunk loop OUTSIDE the vmap, whose trip count is the largest
# number of rows any site of the batch needs (rows past a site's count are never read). Chunk sizes and chunk bodies
# (_q_one / _b_one, vmapped over the batch) are those of query_min / brute_min, so each row gets the same value.
#  lever 4  block culling: 4^3 blocks of grid points (+ their h/2-shifted midpoints) whose distance function
#           dv(q) = min_j |q - p_j| - w_j is provably < 0 everywhere (inside: eps_in, no ion) or >= max(rp, exrad)
#           everywhere (outside: solvent, ion) are classified without per-point queries; only mixed blocks are queried.
#           Bounds from the block-centre value (1-Lipschitz): reach-2 cell query (upper bound; lower bound
#           min(v, 2 * cell - max w) because unsearched atoms are > 2 cells away) or brute force (model grid, exact),
#           +- half-diagonal + 2e-3 A margin. Cell-limited queries agree with the true dv on every class threshold
#           (an atom within rp/exrad + w < 4.01 A of a point is in its 27 cells), so the classes are exact.
#           The contact test runs only over the shell rows (dynamic count); the full-grid nearest-atom index used by
#           the boundary projection is recomputed on the NB boundary rows.
#  lever 6  probe cell list hashed (sorted ids + searchsorted; no dense ncell^3 x Kp table) and the boundary contact /
#           probe projection queries run only over the rows needed (dynamic counts).
#  lever 7  reach_mask loops only over chunks holding valid region probes (scatter-set of True: order-free).
# Padding rows of the compacted lists (nonzero fill 0) are given the value of the row they duplicate, so every
# scatter they take part in writes what M6 wrote.
CULL = True          # lever 4
PROBE_HASH = False   # lever 6: hashed probe cells measured SLOWER on the A40 (+~1.3 s/state maps); kept behind this flag
DYN_BPROJ = True     # lever 6
DYN_REACH = True     # lever 7
BLK = 4
MARG = 2e-3


def probe_cells_hashed(P, lo, ncell, cell):
    """hashed version of build_cells(P, ...): (sorted cell ids, permutation, max occupancy over the ncell^3 cells)."""
    c = jnp.floor((P - lo) / cell).astype(jnp.int32)
    ok = jnp.all((c >= 0) & (c < ncell), axis=1)
    cid = jnp.where(ok, (c[:, 0] * ncell + c[:, 1]) * ncell + c[:, 2], ncell ** 3)
    order = jnp.argsort(cid); sc = cid[order]
    run = jnp.searchsorted(sc, sc, side="right") - jnp.searchsorted(sc, sc, side="left")
    return sc, order, jnp.max(jnp.where(sc < ncell ** 3, run, 0))


def _dyn_rows(fn, Q, cnt, chunk, args, axes):
    """rows [0, max(cnt)) of fn over chunks of `chunk` rows, batch-explicit: Q [B, nq, 3], cnt [B]; fn(q, *args) on one
    site's chunk, vmapped with in_axes (0,) + axes. Rows not computed: (inf, -1)."""
    B, nq = Q.shape[:2]; pad = (-nq) % chunk
    Qp = jnp.concatenate([Q, jnp.zeros((B, pad, 3), Q.dtype)], 1)
    nr = Qp.shape[1]
    f = jax.vmap(fn, in_axes=(0,) + tuple(axes))

    def body(i, c):
        dm, am = c
        q = jax.lax.dynamic_slice_in_dim(Qp, i * chunk, chunk, axis=1)
        d, a = f(q, *args)
        return (jax.lax.dynamic_update_slice_in_dim(dm, d, i * chunk, axis=1),
                jax.lax.dynamic_update_slice_in_dim(am, a, i * chunk, axis=1))
    init = (jnp.full((B, nr), jnp.inf, Q.dtype), jnp.full((B, nr), -1, jnp.int32))
    nch = (jnp.max(jnp.minimum(cnt, nq)) + chunk - 1) // chunk
    dm, am = jax.lax.fori_loop(0, nch, body, init)
    return dm[:, :nq], am[:, :nq]


def _fill_from(x, cnt, src):
    """rows >= cnt of x [B, n] take the value of row src [B] (the row they duplicate)."""
    v = jnp.take_along_axis(x, src[:, None], 1)
    return jnp.where(jnp.arange(x.shape[1])[None] < cnt[:, None], x, v)


def build_maps_b(pos, rad, probes, acent, h, n, Ka, Kp, ncell_a, ncell_p, NS, NB, PB, brute, hstat, shared, rp=RP, exrad=EXRAD):
    """build_maps (identical outputs) for B sites at once; pos/rad/probes shared by the sites (shared) or per site [B,...]."""
    pa = None if shared else 0
    dt = pos.dtype
    rmax = 2.0; L = (n - 1) * h; ca = rmax + exrad + 0.01; cp = PCELL
    N3 = n ** 3; nbk = -(-n // BLK); NBLK = nbk ** 3
    offs1 = _offs(1); offsP = _offs(PREACH)
    w_c = rad + rp                                              # contact radii (as qmin(cpr, rad + rp))
    V = lambda f, *ax: jax.vmap(f, in_axes=ax)

    # -------------------------------------------------- per-site set-up: cells, region probes, grid, block classes
    def stage0(pos, rad, probes, a):
        lo = a - L / 2 - (rmax + exrad + 0.5)
        if brute:
            tab_a = jnp.zeros((1, 1), jnp.int32); ov_a = jnp.asarray(0)
        else:
            tab_a, ov_a = build_cells(pos, rad, lo, ncell_a, ca, Ka)
        lop = a - L / 2 - (rp + 0.5)
        if PROBE_HASH:
            psc, pord, ov_p = probe_cells_hashed(probes, lop, ncell_p, cp)
            ptab = (psc, pord)
        else:
            tab_p, ov_p = build_cells(probes, jnp.zeros(probes.shape[0], dt), lop, ncell_p, cp, Kp)
            ptab = (tab_p,)
        preg, pval, npreg = region_probes(probes, a, h, n, PB, rp)
        x, y, z = grid_coords(a, h, n)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        G = jnp.stack([X, Y, Z], -1).reshape(-1, 3)
        ii = jnp.arange(n) // BLK
        pblk = ((ii[:, None, None] * nbk + ii[None, :, None]) * nbk + ii[None, None, :]).reshape(-1)
        if CULL:
            cax = (jnp.arange(nbk) * BLK + 1.75 - (n - 1) / 2) * h
            CX, CY, CZ = jnp.meshgrid(a[0] + cax, a[1] + cax, a[2] + cax, indexing="ij")
            C = jnp.stack([CX, CY, CZ], -1).reshape(-1, 3)
            r = 1.75 * h * np.sqrt(3.0) + MARG
            if brute:
                vc, _ = brute_min(C, pos, rad); lbc = vc
            else:
                vc, _ = query_min(C, pos, rad, tab_a, lo, ncell_a, ca, reach=2)
                lbc = jnp.minimum(vc, 2 * ca - jnp.max(rad))
            inb = vc + r < 0
            outb = (lbc - r >= max(rp, exrad)) & ~inb
        else:
            inb = outb = jnp.zeros(NBLK, bool)
        mixed = ~(inb | outb)
        nmix = jnp.sum(mixed)
        mb = jnp.nonzero(mixed, size=NBLK, fill_value=0)[0]
        b3 = jnp.stack([mb // (nbk * nbk), (mb // nbk) % nbk, mb % nbk], -1)
        loc = jnp.asarray(np.array([(p, q, t) for p in range(BLK) for q in range(BLK) for t in range(BLK)], np.int32))
        i3 = b3[:, None, :] * BLK + loc[None]
        okp = jnp.all(i3 < n, axis=2) & (jnp.arange(NBLK) < nmix)[:, None]
        flat = jnp.where(okp, (i3[..., 0] * n + i3[..., 1]) * n + i3[..., 2], N3).reshape(-1)
        Gm = G[jnp.minimum(flat, N3 - 1)]
        pcls = jnp.where(inb, 0, jnp.where(outb, 1, 2))[pblk]
        return dict(lo=lo, tab_a=tab_a, ov_a=ov_a, lop=lop, ptab=ptab, ov_p=ov_p, preg=preg, pval=pval, npreg=npreg, G=G,
                    flat=flat, Gm=Gm, nmix=nmix, pcls=pcls)
    S = V(stage0, pa, pa, pa, 0)(pos, rad, probes, acent)

    def qrows(Q, cnt, w, wax):
        """nearest-atom query (cell list or brute) of rows [0, cnt) of Q [B, nq, 3]."""
        if brute:
            return _dyn_rows(_b_one, Q, cnt, 65536, (pos, w), (pa, wax))
        return _dyn_rows(lambda q, P, w_, tab, lo: _q_one(q, P, w_, tab, lo, ncell_a, ca, offs1), Q, cnt,
                         _q_chunk(1, Ka), (pos, w, S["tab_a"], S["lo"]), (pa, wax, 0, 0))

    ncnt = S["nmix"] * (BLK ** 3)
    da_m, _ = qrows(S["Gm"], ncnt, rad, pa)
    ion = V(lambda pc, f, d: (pc == 1).at[f].set(d >= exrad, mode="drop"), 0, 0, 0)(S["pcls"], S["flat"], da_m)
    out = {"ion": ion}
    ov_s = jnp.zeros(acent.shape[0], jnp.int32)
    for ax, name in enumerate(("ex", "ey", "ez")):
        offv = jnp.zeros(3, dt).at[ax].set(h / 2)
        dv_m, iv_m = qrows(S["Gm"] + offv, ncnt, rad, pa)

        def stage1(pos, rad, pc, f, d, iv, G):
            gerp = (pc == 1).at[f].set(d >= rp, mode="drop")
            shell = jnp.zeros(N3, bool).at[f].set((d >= 0) & (d < rp), mode="drop")
            ivf = jnp.zeros(N3, jnp.int32).at[f].set(iv, mode="drop")
            nsh = jnp.sum(shell)
            si = jnp.nonzero(shell, size=NS, fill_value=0)[0]
            Ms = G[si] + offv
            ia = jnp.maximum(ivf[si], 0); u = Ms - pos[ia]; u = u / jnp.maximum(jnp.linalg.norm(u, axis=1, keepdims=True), 1e-9)
            cpr = pos[ia] + (rad[ia] + rp)[:, None] * u
            return gerp, shell, nsh, si, cpr
        gerp, shell, nsh, si, cpr = V(stage1, pa, pa, 0, 0, 0, 0, 0)(pos, rad, S["pcls"], S["flat"], dv_m, iv_m, S["G"])
        ov_s = jnp.maximum(ov_s, nsh)
        cnt = jnp.minimum(nsh, NS)
        dcp, _ = qrows(cpr, cnt, w_c, pa)
        dcp = _fill_from(dcp, cnt, jnp.zeros_like(cnt))           # padding rows (si = 0) duplicate row 0 when shell[0]
        contact = dcp >= -1e-5
        need = V(lambda sh, si_, c: jnp.zeros_like(sh).at[si_].max(~c) & sh, 0, 0, 0)(shell, si, contact)
        rm = reach_mask_b(S["preg"], S["pval"], S["npreg"], acent, h, n, ax, hstat, PB, rp)

        def stage2(shell, si_, c, need, rm, gerp):
            reach = jnp.zeros(N3, bool).at[si_].max(c) | (need & rm)
            return (~(gerp | (shell & reach))).reshape(n, n, n)
        out[name] = V(stage2, 0, 0, 0, 0, 0, 0)(shell, si, contact, need, rm, gerp)

    # -------------------------------------------------- dielectric boundary + surface projection
    def stage3(ex, ey, ez, G):
        nb = [ex, jnp.roll(ex, 1, 0), ey, jnp.roll(ey, 1, 1), ez, jnp.roll(ez, 1, 2)]
        s_ = sum(a_.astype(jnp.int32) for a_ in nb)
        bnd = (s_ > 0) & (s_ < 6)
        bnd = bnd.at[0].set(False).at[-1].set(False).at[:, 0].set(False).at[:, -1].set(False).at[:, :, 0].set(False).at[:, :, -1].set(False)
        bflat = bnd.reshape(-1)
        nbd = jnp.sum(bflat)
        bidx = jnp.nonzero(bflat, size=NB, fill_value=-1)[0]
        return nbd, bidx, G[jnp.maximum(bidx, 0)]
    nbd, bidx, Gb = V(stage3, 0, 0, 0, 0)(out["ex"], out["ey"], out["ez"], S["G"])
    nbr = jnp.minimum(nbd + 1, NB)                    # boundary rows + the first padding row (G[0]); later rows copy it
    src = jnp.minimum(nbd, NB - 1)
    _, ia_b = qrows(Gb, nbr, rad, pa)
    ia_b = _fill_from(ia_b, nbr, src)

    def stage4(pos, rad, Gb, iab):
        ia = jnp.maximum(iab, 0)
        v = Gb - pos[ia]; nv = jnp.linalg.norm(v, axis=1, keepdims=True)
        sa = pos[ia] + rad[ia][:, None] * v / jnp.maximum(nv, 1e-6)
        cpr = pos[ia] + (rad[ia] + rp)[:, None] * v / jnp.maximum(nv, 1e-6)
        return sa, cpr
    sa, cpr_b = V(stage4, pa, pa, 0, 0)(pos, rad, Gb, ia_b)
    nbr2 = nbr if DYN_BPROJ else jnp.full_like(nbr, NB)
    dcp_b, _ = qrows(cpr_b, nbr2, w_c, pa)
    dcp_b = _fill_from(dcp_b, nbr2, src)

    def stage5(dcp, bidx, Gb):
        needb = (dcp < -1e-5) & (bidx >= 0)
        nbproj = jnp.sum(needb)
        bi2 = jnp.nonzero(needb, size=NB, fill_value=0)[0]
        return needb, nbproj, bi2, Gb[bi2]
    needb, nbproj, bi2, Gq = V(stage5, 0, 0, 0)(dcp_b, bidx, Gb)
    zp = jnp.zeros(probes.shape[-2], dt)
    npq = jnp.minimum(nbproj, NB) if DYN_BPROJ else jnp.full_like(nbproj, NB)
    if PROBE_HASH:
        fq = lambda q, P, w_, sc, order, lo: _q_one_hashed(q, P, w_, sc, order, lo, ncell_p, cp, offsP, Kp)
        _, ip2 = _dyn_rows(fq, Gq, npq, _q_chunk(PREACH, Kp), (probes, zp) + S["ptab"] + (S["lop"],), (pa, None, 0, 0, 0))
    else:
        fq = lambda q, P, w_, tab, lo: _q_one(q, P, w_, tab, lo, ncell_p, cp, offsP)
        _, ip2 = _dyn_rows(fq, Gq, npq, _q_chunk(PREACH, Kp), (probes, zp) + S["ptab"] + (S["lop"],), (pa, None, 0, 0))
    ip2 = _fill_from(ip2, npq, jnp.zeros_like(npq))     # padding rows (bi2 = 0) duplicate row 0 when needb[0]

    def stage6(probes, Gb, needb, bi2, ip2, dcp, sa):
        ip = jnp.full(Gb.shape[0], -1, ip2.dtype).at[bi2].set(jnp.where(needb[bi2], ip2, -1))
        ipc = probes[jnp.maximum(ip, 0)]
        vp = Gb - ipc; nvp = jnp.linalg.norm(vp, axis=1, keepdims=True)
        sp = ipc + rp * vp / jnp.maximum(nvp, 1e-6)
        use_atom = (dcp >= -1e-5) | (ip < 0)
        return jnp.where(use_atom[:, None], sa, sp)
    surf = V(stage6, pa, 0, 0, 0, 0, 0, 0)(probes, Gb, needb, bi2, ip2, dcp_b, sa)
    res = dict(ex=out["ex"], ey=out["ey"], ez=out["ez"], ion=out["ion"].reshape(-1, n, n, n), bidx=bidx, surf=surf)
    ctr = jnp.stack([S["ov_a"].astype(jnp.int32), S["ov_p"].astype(jnp.int32), (nbproj - NB).astype(jnp.int32),
                     (ov_s - NS).astype(jnp.int32), (nbd - NB).astype(jnp.int32), (S["npreg"] - PB).astype(jnp.int32),
                     nbd.astype(jnp.int32)], -1)
    return res, ctr


def reach_mask_b(p, pvalid, npreg, acent, h, n, ax, hstat, PB, rp=RP, budget=1 << 22):
    """reach_mask for B sites (p [B,PB,3] compacted region probes, valid first). Lever 7: the chunk loop stops after the
    last chunk holding a valid probe of any site (chunks past it only scatter to the dropped index)."""
    dt = p.dtype
    off = jnp.zeros(3, dt).at[ax].set(h / 2)
    O = jnp.asarray(reach_offsets(hstat, rp), jnp.int32)
    ch = max(256, budget // O.shape[0]); B, npb = p.shape[:2]; pad = (-npb) % ch
    pp = jnp.concatenate([p, jnp.zeros((B, pad, 3), dt)], 1).reshape(B, -1, ch, 3)
    vv = jnp.concatenate([pvalid, jnp.zeros((B, pad), bool)], 1).reshape(B, -1, ch)

    def one(pc, ok0, a, mask):
        x, y, z = grid_coords(a, h, n)
        i0 = jnp.floor((pc - a - off) / h + (n - 1) / 2).astype(jnp.int32)
        idx = i0[:, None, :] + O[None]
        ok = jnp.all((idx >= 0) & (idx < n), axis=2) & ok0[:, None]
        ic = jnp.clip(idx, 0, n - 1)
        M = jnp.stack([x[ic[..., 0]] + off[0], y[ic[..., 1]] + off[1], z[ic[..., 2]] + off[2]], -1)
        d = jnp.linalg.norm(M - pc[:, None, :], axis=-1) - jnp.zeros((), dt)
        hit = ok & (d < rp)
        flat = jnp.where(hit, (ic[..., 0] * n + ic[..., 1]) * n + ic[..., 2], n ** 3)
        return mask.at[flat.reshape(-1)].set(True, mode="drop")
    f = jax.vmap(one)

    def body(c, mask):
        return f(pp[:, c], vv[:, c], acent, mask)
    nch = pp.shape[1]
    if DYN_REACH:
        nch = (jnp.max(jnp.minimum(npreg, PB)) + ch - 1) // ch
    return jax.lax.fori_loop(0, nch, body, jnp.zeros((B, n ** 3), jnp.bool_))


@partial(jax.jit, static_argnames=("sz",))
def maps_batch(pos_m, rad_m, probes, acent, spos, srad, sprb, sz: Sizes):
    """Coarse (h 1), focused (h 0.25) and model-compound (h 0.25) maps for B sites (batch-explicit build_maps_b; the M6
    per-site vmap of build_maps is maps_batch_ref).
    pos_m/rad_m [NA_C[c]] class arrays (FAR / 0 padded); probes [NP_C[c],3]; acent [B,3]; spos [B,MA,3], srad [B,MA]
    model atoms (FAR / 0 padded); sprb [B,MP_MODEL,3] model probes. Returns (mc, mf, ms) dicts with leading B, ctr [B,3,7]."""
    n = RC.GSIZE
    nca_c, ncp_c = ncells(1.0, n, "a"), ncells(1.0, n, "p")
    nca_f, ncp_f = ncells(0.25, n, "a"), ncells(0.25, n, "p")
    PBc = probes.shape[0]
    mc, cc = build_maps_b(pos_m, rad_m, probes, acent, 1.0, n, sz.Ka, sz.Kp, nca_c, ncp_c, sz.NSc, sz.NBc, PBc, False, 1.0, True)
    mf, cf = build_maps_b(pos_m, rad_m, probes, acent, 0.25, n, sz.Ka, sz.Kp, nca_f, ncp_f, sz.NSf, sz.NBf, sz.PBf, False, 0.25, True)
    ms, cs = build_maps_b(spos, srad, sprb, acent, 0.25, n, sz.Ka, sz.Kp, nca_f, ncp_f, sz.NSs, sz.NBs, sz.PBs, True, 0.25, False)
    return mc, mf, ms, jnp.stack([cc, cf, cs], 1)


@partial(jax.jit, static_argnames=("sz",))
def maps_batch_ref(pos_m, rad_m, probes, acent, spos, srad, sprb, sz: Sizes):
    """Coarse (h 1), focused (h 0.25) and model-compound (h 0.25) maps for B sites.
    pos_m/rad_m [NA_C[c]] class arrays (FAR / 0 padded); probes [NP_C[c],3]; acent [B,3]; spos [B,MA,3], srad [B,MA]
    model atoms (FAR / 0 padded); sprb [B,MP_MODEL,3] model probes. Returns (mc, mf, ms) dicts with leading B, ctr [B,3,7]."""
    n = RC.GSIZE
    nca_c, ncp_c = ncells(1.0, n, "a"), ncells(1.0, n, "p")
    nca_f, ncp_f = ncells(0.25, n, "a"), ncells(0.25, n, "p")
    PBc = probes.shape[0]

    def one(a, sp, sr, spb):
        mc, cc = build_maps(pos_m, rad_m, probes, a, 1.0, n, sz.Ka, sz.Kp, nca_c, ncp_c, sz.NSc, sz.NBc, PBc, False, 1.0)
        mf, cf = build_maps(pos_m, rad_m, probes, a, 0.25, n, sz.Ka, sz.Kp, nca_f, ncp_f, sz.NSf, sz.NBf, sz.PBf, False, 0.25)
        ms, cs = build_maps(sp, sr, spb, a, 0.25, n, sz.Ka, sz.Kp, nca_f, ncp_f, sz.NSs, sz.NBs, sz.PBs, True, 0.25)
        return mc, mf, ms, jnp.stack([cc, cf, cs])

    return jax.vmap(one)(acent, spos, srad, sprb)
