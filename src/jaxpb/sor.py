"""DelPhi red-black SOR with bounded lane pools (PB mode "delphi").

Even colour first, constant per-lane omega, zero interior start, max-change
checked every RC.CHECK sweeps, RES2 threshold and NLIT cap. Finished lanes retain
all outputs at their first passing block.

The GPU XLA stencil packs the two colours and decodes shared byte operators
with a 16-entry diagonal LUT. Odd grids use contiguous shifted colour arrays;
even grids use fixed neighbour indices. CPU retains the reference full-grid
stencil because packing changes its shape-dependent fusion/FMA rounding.
Only the checked sweep reduces max-change: a split final sweep on GPU, an
in-loop conditional on CPU (splitting the CPU loop also changes rounding).

A device while-loop compacts live lanes at block boundaries and switches among
fixed widths; there is no per-block host synchronisation. CPU work is tiled
into small lane groups to bound the coefficient gathers and stencil buffers.
Each initial width has one executable containing the smaller-width branches;
operator tables have the same padded shape in both PB phases.

Pallas retains its original whole-solve device loop and per-lane early exit.
Block compaction was exact but slower, so its production path is unchanged.
"""
from __future__ import annotations

from functools import lru_cache, partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from . import records as RC


def colour_masks(n, first=0):
    I = np.arange(n)
    par = (I[:, None, None] + I[None, :, None] + I[None, None, :]) % 2
    fm = face_mask_np(n)
    return (par == first) & ~fm, (par != first) & ~fm


def face_mask_np(n):
    m = np.zeros((n, n, n), bool)
    m[0] = m[-1] = True; m[:, 0] = m[:, -1] = True; m[:, :, 0] = m[:, :, -1] = True
    return m


@lru_cache(maxsize=16)
def colour_layout(n):
    """Equal-sized colour lists (one harmless padded point for an odd cube)."""
    f = np.arange(n ** 3, dtype=np.int32)
    parity = (f // (n*n) + (f // n) % n + f % n) % 2
    m = (n ** 3 + 1) // 2
    ids = np.stack([np.pad(f[parity == c], (0, m - np.sum(parity == c))) for c in (0, 1)])
    inner = ~face_mask_np(n).reshape(-1)[ids]
    if n % 2:
        inner[1, -1] = False
    inverse = np.empty(n ** 3, np.int32)
    for c in (0, 1):
        real = f[parity == c]
        inverse[real] = np.arange(len(real), dtype=np.int32)
    return ids, inner, parity, inverse


def pack(a, n):
    return a.reshape(a.shape[:-3] + (n ** 3,))[..., jnp.asarray(colour_layout(n)[0])]


def unpack(a, n):
    _, _, parity, inverse = colour_layout(n)
    return a[..., jnp.asarray(parity), jnp.asarray(inverse)].reshape(a.shape[:-2] + (n, n, n))


def _neighbour(a, colour, offset, n):
    # On odd grids, spatial parity equals flat-index parity: every neighbour is a
    # constant shift of the opposite packed colour. Only interior entries update.
    if n % 2:
        return jnp.roll(a, -((colour + offset) // 2), axis=-1)
    ids, _, _, inverse = colour_layout(n)
    return a[..., jnp.asarray(inverse[(ids[colour] + offset) % (n ** 3)])]


def sor_chunk_packed(phi, rhs, code, lut, om, it, mc, done, *, n,
              res2=RC.RES2, nlit=RC.NLIT, check=RC.CHECK):
    """One stop-check block on [2,ceil(n^3/2)] red/black points, even first.

    Decode shared byte operators inside the stencil instead of expanding four f32
    coefficient grids per lane. Keep the reference expression's left-to-right sums.
    Faces and the odd-grid pad are never updated; done lanes freeze all outputs.
    """
    masks = jnp.asarray(colour_layout(n)[1])

    def half(p, colour):
        c0 = code[colour].astype(jnp.int32)
        other = code[1-colour].astype(jnp.int32)
        cxm = _neighbour(other, colour, -n*n, n)
        cym = _neighbour(other, colour, -n, n)
        czm = _neighbour(other, colour, -1, n)
        e = lambda c, bit: jnp.where(((c >> bit) & 1) != 0, jnp.float32(EPS_IN), jnp.float32(EPS_OUT))
        v = p[1-colour]
        nb = (e(c0, 0) * _neighbour(v, colour, n*n, n) + e(cxm, 0) * _neighbour(v, colour, -n*n, n)
              + e(c0, 1) * _neighbour(v, colour, n, n) + e(cym, 1) * _neighbour(v, colour, -n, n)
              + e(c0, 2) * _neighbour(v, colour, 1, n) + e(czm, 2) * _neighbour(v, colour, -1, n))
        idx = ((c0 & 1) + (cxm & 1) + ((c0 >> 1) & 1) + ((cym >> 1) & 1)
               + ((c0 >> 2) & 1) + ((czm >> 2) & 1) + 8 * ((c0 >> 3) & 1))
        gs = (nb + rhs[colour]) * lut[idx]
        old = p[colour]
        new = jnp.where(masks[colour], old + om * (gs - old), old)
        return p.at[colour].set(new)

    def sweep(p):
        return half(half(p, 0), 1)

    p = lax.fori_loop(0, check - 1, lambda k, p: sweep(p), phi)
    new = sweep(p)
    mc2 = jnp.max(jnp.abs(new - p))
    it2 = it + check
    done2 = (mc2 < res2) | (it2 >= nlit)
    return (jnp.where(done, phi, new), jnp.where(done, it, it2), jnp.where(done, mc, mc2), done | done2)


@partial(jax.jit, donate_argnums=(0, 2, 3, 4), static_argnames=("n",))
def _packed_block(phi, rhs, it, mc, done, om, op, code, lut, *, n):
    """One device block for W packed lanes; shared codes [E,n^3], LUT [E,16]."""
    cd = code[op][..., jnp.asarray(colour_layout(n)[0])]
    return jax.vmap(partial(sor_chunk_packed, n=n))(phi, rhs, cd, lut[op], om, it, mc, done)


def sor_chunk(phi, rhs, ex, ey, ez, dinv, cA, cB, om, it, mc, done, res2=RC.RES2, nlit=RC.NLIT, check=RC.CHECK):
    """Reference CPU stencil, with a conditional reduction on the checked sweep."""
    def gs(p):
        nb = (ex * jnp.roll(p, -1, 0) + jnp.roll(ex, 1, 0) * jnp.roll(p, 1, 0)
              + ey * jnp.roll(p, -1, 1) + jnp.roll(ey, 1, 1) * jnp.roll(p, 1, 1)
              + ez * jnp.roll(p, -1, 2) + jnp.roll(ez, 1, 2) * jnp.roll(p, 1, 2))
        return (nb + rhs) * dinv

    def body(k, s):
        p, mc_ = s
        new = jnp.where(cA, p + om * (gs(p) - p), p)
        new = jnp.where(cB, new + om * (gs(new) - new), new)
        mc_ = lax.cond(k == check - 1, lambda _: jnp.max(jnp.abs(new - p)), lambda _: mc_, operand=None)
        return new, mc_

    new, mc2 = lax.fori_loop(0, check, body, (phi, mc))
    it2 = it + check
    done2 = (mc2 < res2) | (it2 >= nlit)
    return (jnp.where(done, phi, new), jnp.where(done, it, it2), jnp.where(done, mc, mc2), done | done2)


_lanes = jax.vmap(sor_chunk, in_axes=(0, 0, 0, 0, 0, 0, None, None, 0, 0, 0, 0))


@partial(jax.jit, donate_argnums=(0, 2, 3, 4))
def _cpu_block(phi, rhs, it, mc, done, om, op, exE, eyE, ezE, dvE, cA, cB):
    """phi/rhs [W,n,n,n]; it/mc/done/om/op [W]; coefficient table [E,n,n,n] x4. One 10-sweep block per lane."""
    return _lanes(phi, rhs, exE[op], eyE[op], ezE[op], dvE[op], cA, cB, om, it, mc, done)


def sor_block(phi, rhs, it, mc, done, om, op, *coefs, n):
    if phi.ndim == 4:
        cA, cB = (jnp.asarray(m) for m in colour_masks(n))
        return _cpu_block(phi, rhs, it, mc, done, om, op, *coefs, cA, cB)
    return _packed_block(phi, rhs, it, mc, done, om, op, *coefs, n=n)


@partial(jax.jit, static_argnames=("W",))
def lanes_init(phi0, rhs0, om0, item, W):
    """new pool of width W from the item arrays [NI,...]: lane l takes item[l] (item < 0 -> empty, done lane)."""
    ok = item >= 0
    ii = jnp.maximum(item, 0)
    phi = jnp.where(ok[:, None, None, None], phi0[ii], 0.0)
    rhs = jnp.where(ok[:, None, None, None], rhs0[ii], 0.0)
    if jax.default_backend() != "cpu":
        phi, rhs = pack(phi, phi0.shape[1]), pack(rhs, phi0.shape[1])
    return (phi, rhs, jnp.zeros(W, jnp.int32), jnp.full(W, 1e9, phi0.dtype), ~ok, jnp.where(ok, om0[ii], 1.0),
            jnp.zeros_like(phi0))


@partial(jax.jit, donate_argnums=(0,))
def lanes_store(store, phi, lane_item):
    """store[item] = phi[lane] for lanes with lane_item >= 0 (others dropped)."""
    tgt = jnp.where(lane_item >= 0, lane_item, store.shape[0])
    return store.at[tgt].set(unpack(phi, store.shape[1]) if phi.ndim == 3 else phi, mode="drop")


@jax.jit
def lanes_take(phi, rhs, it, mc, done, om, op, idx):
    """compact: new lane k = old lane idx[k] (idx < 0 -> empty done lane)."""
    ok = idx >= 0
    i = jnp.maximum(idx, 0)
    z = lambda a, fill: jnp.where(ok.reshape((-1,) + (1,) * (a.ndim - 1)), a[i], fill)
    return (z(phi, 0.0), z(rhs, 0.0), z(it, 0), z(mc, 1e9), jnp.where(ok, done[i], True), z(om, 1.0), z(op, 0))


def width_for(n, widths=None):
    widths = RC.LANE_WIDTHS if widths is None else widths
    for w in widths:
        if n <= w:
            return w
    raise ValueError(f"{n} lanes > max width {widths[-1]}")


def run_pool(phi0, rhs0, om0, op0, valid, coefs, stats=None):
    """XLA pool; shared CPU coefficients or GPU byte operators, fixed width ladder."""
    return _run_pool(phi0, rhs0, om0, op0, valid, coefs, stats=stats)


def _run_pool(phi0, rhs0, om0, op0, valid, coefs, stats=None):
    NI = phi0.shape[0]; n = phi0.shape[1]
    items = np.nonzero(valid)[0]
    # A bounded lane tile also limits CPU working-set size. No operator or grid
    # depends on a lane's neighbours, so chunks retain each lane's stop block.
    widths = RC.CPU_LANE_WIDTHS if jax.default_backend() == "cpu" else RC.LANE_WIDTHS
    chunk = widths[-1]
    op_h = np.asarray(op0)
    store = None
    res_it = np.zeros(NI, np.int32); res_mc = np.zeros(NI, np.float32)
    blocks = 0; lane_blocks = 0
    for start in range(0, max(len(items), 1), chunk):
        group = items[start:start + chunk]
        W = width_for(len(group), widths)
        item = np.full(W, -1, np.int32); item[:len(group)] = group
        phi, rhs, it, mc, done, om, empty = lanes_init(phi0, rhs0, om0, jax.device_put(item), W=W)
        if store is None:
            store = empty
        del empty
        op = jax.device_put(np.where(item >= 0, op_h[np.maximum(item, 0)], 0).astype(np.int32))
        phi, it, mc, done, nb, lb = device_solve(phi, rhs, it, mc, done, om, op, *coefs,
                                                n=n, widths=tuple(w for w in widths if w <= W))
        store = lanes_store(store, phi, jax.device_put(item))
        res_it[group] = np.asarray(it)[:len(group)]
        res_mc[group] = np.asarray(mc)[:len(group)]
        blocks += int(nb); lane_blocks += int(lb)
    if stats is not None:
        stats.append(dict(blocks=blocks, lane_blocks=lane_blocks, needed=int(res_it[valid].sum() // RC.CHECK)))
    return store, res_it, res_mc


@partial(jax.jit, donate_argnums=(0, 2, 3, 4), static_argnames=("n", "widths"))
def device_solve(phi, rhs, it, mc, done, om, op, *coefs, n, widths):
    """Compact live lanes and select a bounded width at stop blocks on the device.

    The carry is in original lane order. A sentinel drops padding scatters, so
    finished lanes retain their potential, sweep count and maxchange exactly.
    No host synchronisation occurs between blocks.
    """
    W = phi.shape[0]
    def body(carry):
        p, it, mc, done, blocks, lb = carry
        live = jnp.sum(~done)
        indices = jnp.nonzero(~done, size=W, fill_value=W)[0]
        ii = jnp.minimum(indices, W-1)
        def branch(w):
            def solve(_):
                ix = ii[:w]; active = indices[:w] < W
                args = (p[ix], rhs[ix], it[ix], mc[ix], ~active, om[ix], op[ix])
                if p.ndim == 4:
                    ca, cb = (jnp.asarray(m) for m in colour_masks(n))
                    q, qi, qm, qd = _cpu_block(*args, *coefs, ca, cb)
                else:
                    q, qi, qm, qd = _packed_block(*args, *coefs, n=n)
                idx = indices[:w]
                put = lambda a, v: a.at[idx].set(v, mode="drop")
                return put(p,q), put(it,qi), put(mc,qm), put(done,qd), blocks+1, lb+w
            return solve
        choice = sum(live > w for w in widths[:-1])
        return lax.switch(choice, tuple(branch(w) for w in widths), operand=None)
    init = phi, it, mc, done, jnp.int32(0), jnp.int32(0)
    return lax.while_loop(lambda c: ~jnp.all(c[3]), body, init)


# ================================================================================================ Pallas backend (M5)
EPS_IN, EPS_OUT = 15.0, 80.0          # = pb.EPSIN / pb.EPSOUT (pb imports this module)
BLK = 2048                            # grid points per program
NUM_WARPS = 4


def flat_layout(n, blk=BLK):
    """(OFF, NBLK, NF): interior offset (multiple of blk, >= n^2), blocks covering n^3, padded flat length."""
    n3 = n ** 3
    off = -(-(n * n) // blk) * blk
    nblk = -(-n3 // blk)
    return off, nblk, off + nblk * blk + off


def op_code(m):
    """maps dict (bool ex, ey, ez, ion [n,n,n]) -> uint8 [n^3] operator code."""
    u = lambda a: a.astype(jnp.uint8)
    return (u(m["ex"]) | (u(m["ey"]) << 1) | (u(m["ez"]) << 2) | (u(m["ion"]) << 3)).reshape(-1)


def op_lut(kap2h2, epsin=EPS_IN, epsout=EPS_OUT):
    """[16] f32: 1/diag for code index k + 8*ion, k = number of eps_in faces (0..6); 0 for k = 7. The partial sums of
    pb.operator_coeffs are exact integers in f32, so 1/diag here is bitwise the reference dinv."""
    t = jnp.arange(16)
    k = (t % 8).astype(jnp.float32); ion = (t // 8).astype(jnp.float32)
    S = k * jnp.float32(epsin) + (6.0 - k) * jnp.float32(epsout)
    diag = S + kap2h2 * epsout * ion
    return jnp.where(t % 8 <= 6, 1.0 / diag, 0.0).astype(jnp.float32)


def _half_kernel(col_ref, op_ref, om_ref, done_ref, lut_ref, rhs_ref, code_ref, phi_in, phi_out, mc_out, *, n, blk, off):
    """one colour half-sweep of lane l on grid points [b*blk, (b+1)*blk) (flat C order, i slowest)."""
    l = pl.program_id(0); b = pl.program_id(1)
    live = done_ref[l] == 0

    @pl.when(live)
    def _():
        o = op_ref[l]; om = om_ref[l]; colour = col_ref[0]
        base = off + b * blk
        f = b * blk + lax.broadcasted_iota(jnp.int32, (blk,), 0)
        i = f // (n * n); j = (f // n) % n; k = f % n
        inner = (f < n ** 3) & (i >= 1) & (i <= n - 2) & (j >= 1) & (j <= n - 2) & (k >= 1) & (k <= n - 2)
        upd = inner & (((i + j + k) % 2) == colour)
        ld = lambda s: phi_in[l, pl.ds(base + s, blk)]
        cd = lambda s: code_ref[o, pl.ds(base + s, blk)].astype(jnp.int32)
        p0 = ld(0)
        c0 = cd(0); cxm = cd(-n * n); cym = cd(-n); czm = cd(-1)
        e = lambda c, bit: jnp.where(((c >> bit) & 1) != 0, jnp.float32(EPS_IN), jnp.float32(EPS_OUT))
        nb = (e(c0, 0) * ld(n * n) + e(cxm, 0) * ld(-n * n) + e(c0, 1) * ld(n) + e(cym, 1) * ld(-n)
              + e(c0, 2) * ld(1) + e(czm, 2) * ld(-1))
        idx = ((c0 & 1) + (cxm & 1) + ((c0 >> 1) & 1) + ((cym >> 1) & 1) + ((c0 >> 2) & 1) + ((czm >> 2) & 1)
               + 8 * ((c0 >> 3) & 1))
        dinv = jnp.zeros((blk,), jnp.float32)
        for t in range(16):
            if t % 8 <= 6:
                dinv = jnp.where(idx == t, lut_ref[o, t], dinv)
        gs = (nb + rhs_ref[l, pl.ds(base, blk)]) * dinv
        new = p0 + om * (gs - p0)
        phi_out[l, pl.ds(base, blk)] = jnp.where(upd, new, p0)
        mc_out[l, b] = jnp.max(jnp.where(upd, jnp.abs(new - p0), jnp.float32(0.0)))

    @pl.when(jnp.logical_not(live))
    def _():
        mc_out[l, b] = jnp.float32(0.0)


def _triton_params(num_warps):
    from jax.experimental.pallas import triton as pltriton
    cls = getattr(pltriton, "CompilerParams", None) or getattr(pltriton, "TritonCompilerParams")
    return cls(num_warps=num_warps, num_stages=1)


def _half(phi, col, op, om, done, lut, rhs, code, n, blk, off, interpret, num_warps):
    W, NF = phi.shape
    nblk = (NF - 2 * off) // blk
    kw = {} if interpret else dict(compiler_params=_triton_params(num_warps))
    return pl.pallas_call(
        partial(_half_kernel, n=n, blk=blk, off=off), grid=(W, nblk),
        out_shape=(jax.ShapeDtypeStruct(phi.shape, phi.dtype), jax.ShapeDtypeStruct((W, nblk), jnp.float32)),
        input_output_aliases={7: 0}, interpret=interpret, name="sor_half", **kw)(col, op, om, done, lut, rhs, code, phi)


@partial(jax.jit, static_argnames=("interpret", "blk", "num_warps"))
def sor_solve(phi0, rhs0, om0, op0, valid, code, lut, interpret=False, blk=BLK, num_warps=NUM_WARPS):
    """Whole DelPhi-mode solve of W lanes on the device. phi0/rhs0 [W,n,n,n] f32, om0 [W] f32, op0 [W] i32, valid [W]
    bool, code [E,n^3] uint8, lut [E,16] f32. Returns phi [W,n,n,n] (0 for invalid lanes), sweeps [W] i32, last max
    change [W] f32, live lane-blocks executed (i32)."""
    W, n = phi0.shape[0], phi0.shape[1]
    n3 = n ** 3
    off, nblk, NF = flat_layout(n, blk)
    pw = ((0, 0), (off, NF - off - n3))
    phi = jnp.pad(phi0.reshape(W, n3), pw); rhs = jnp.pad(rhs0.reshape(W, n3), pw); cd = jnp.pad(code, pw)
    op = op0.astype(jnp.int32); om = om0.astype(jnp.float32)
    c0 = jnp.zeros((1,), jnp.int32); c1 = jnp.ones((1,), jnp.int32)

    def block(carry):
        phi, it, mc, done, lb = carry
        d32 = done.astype(jnp.int32)

        def sweep(s, c):
            p, _ = c
            p, m0 = _half(p, c0, op, om, d32, lut, rhs, cd, n, blk, off, interpret, num_warps)
            p, m1 = _half(p, c1, op, om, d32, lut, rhs, cd, n, blk, off, interpret, num_warps)
            return p, jnp.maximum(jnp.max(m0, axis=1), jnp.max(m1, axis=1))
        phi, m = lax.fori_loop(0, RC.CHECK, sweep, (phi, jnp.zeros((W,), jnp.float32)))
        it2 = it + RC.CHECK
        done2 = (m < RC.RES2) | (it2 >= RC.NLIT)
        return (phi, jnp.where(done, it, it2), jnp.where(done, mc, m), done | done2, lb + jnp.sum(~done).astype(jnp.int32))

    init = (phi, jnp.zeros((W,), jnp.int32), jnp.full((W,), 1e9, jnp.float32), jnp.logical_not(valid), jnp.int32(0))
    phi, it, mc, done, lb = lax.while_loop(lambda c: jnp.logical_not(jnp.all(c[3])), block, init)
    out = phi[:, off:off + n3].reshape(W, n, n, n)
    return jnp.where(valid[:, None, None, None], out, 0.0), it, mc, lb


def run_pool_pallas(phi0, rhs0, om0, op0, valid, codes, stats=None, interpret=False):
    """run_pool() contract on the Pallas backend: all NI items are lanes (W = NI). codes = (code [E,n^3] u8, lut [E,16])."""
    out, it, mc, lb = sor_solve(phi0, rhs0, om0, op0, jax.device_put(np.asarray(valid)), codes[0], codes[1], interpret=interpret)
    it_h = np.asarray(it).astype(np.int32); mc_h = np.asarray(mc).astype(np.float32)
    v = np.asarray(valid)
    it_h = np.where(v, it_h, 0).astype(np.int32); mc_h = np.where(v, mc_h, 0).astype(np.float32)
    if stats is not None:
        stats.append(dict(blocks=None, lane_blocks=int(lb), needed=int(it_h[v].sum() // RC.CHECK)))
    return out, it_h, mc_h


def pallas_available():
    """True when the default backend is a GPU on which the Triton Pallas lowering imports."""
    try:
        if jax.default_backend() != "gpu":
            return False
        _triton_params(1)
        return True
    except Exception:  # noqa: BLE001
        return False
