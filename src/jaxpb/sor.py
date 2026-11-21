"""jaxpb.sor -- DelPhi-emulating red-black SOR on an XLA lane pool (PB mode "delphi").

Per lane (one right-hand side on one operator): zero interior start, even colour ((i+j+k) even) first, constant omega,
stop after the first 10-sweep block whose last sweep changed phi by < RES2 (max abs), cap NLIT -- identical to v1
sor_delphi / v2 sor_chunk, so sweep counts and potentials are those of the prototypes.

Lane pool: lanes live in a pool of width W in LANE_WIDTHS; each lane carries its operator index into a coefficient table
[E, n^3] (ex, ey, ez, dinv) so lanes of different sites/grids share one program. One jitted `sor_block[W]` call = one
10-sweep stop-check block for all W lanes; after each block the host moves finished lanes to the output store and,
when the live lanes fit a smaller width, compacts the pool (no lane ever sweeps past its own stop test; frozen lanes
are bitwise unchanged). Executables: sor_block x3 widths, lanes_init x3, lanes_store x3, lanes_take <= 3 (the
coefficient table always has E = 2B entries: phase 2 pads it, so sor_block does not recompile per phase).

Pallas backend (M5, `sor_solve`): the same iteration as ONE jitted program per lane count. Lanes keep a flat, zero-padded
copy of phi/rhs ([W, NF], interior offset OFF >= n^2 so every neighbour load is in bounds) and read the operator as a
uint8 code per grid point (bit0 ex, bit1 ey, bit2 ez inside the solute (eps_in), bit3 ion-accessible) plus a 16-entry
1/diag table per operator (diag = number of eps_in faces * eps_in + rest * eps_out + kappa^2 h^2 eps_out ion, exact
integer partial sums as in pb.operator_coeffs). One Triton kernel launch = one colour half-sweep for all lanes on a
(lane, BLK-point block) grid; the launch boundary is the red/black synchronisation. A lane whose done flag is set
returns at once (per-lane early exit, no host repacking); the 10-sweep max-change stop and the NLIT cap are evaluated on
the device in a lax.while_loop, and the max over blocks is order-independent, so the solve is deterministic.
Why not one program per lane for the whole solve: a red/black half-sweep needs a grid-wide barrier between colours;
inside one Triton program that would mean a CTA barrier + global-memory fence that Pallas does not expose, and one
program per lane leaves the A40 with 256 CTAs of serial work. The per-colour launch is the DESIGN section 3 fallback.
"""
from __future__ import annotations

from functools import partial

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


def sor_chunk(phi, rhs, ex, ey, ez, dinv, cA, cB, om, it, mc, done, res2=RC.RES2, nlit=RC.NLIT, check=RC.CHECK):
    """one 10-sweep block of v2 maps2.sor_chunk (NBLK=1) for one lane."""
    def gs(p):
        nb = (ex * jnp.roll(p, -1, 0) + jnp.roll(ex, 1, 0) * jnp.roll(p, 1, 0)
              + ey * jnp.roll(p, -1, 1) + jnp.roll(ey, 1, 1) * jnp.roll(p, 1, 1)
              + ez * jnp.roll(p, -1, 2) + jnp.roll(ez, 1, 2) * jnp.roll(p, 1, 2))
        return (nb + rhs) * dinv

    def body(k, s):
        p, mc_ = s
        new = jnp.where(cA, p + om * (gs(p) - p), p)
        new = jnp.where(cB, new + om * (gs(new) - new), new)
        mc_ = jnp.where(k == check - 1, jnp.max(jnp.abs(new - p)), mc_)
        return new, mc_

    new, mc2 = jax.lax.fori_loop(0, check, body, (phi, mc))
    it2 = it + check
    done2 = (mc2 < res2) | (it2 >= nlit)
    return (jnp.where(done, phi, new), jnp.where(done, it, it2), jnp.where(done, mc, mc2), done | done2)


_lanes = jax.vmap(sor_chunk, in_axes=(0, 0, 0, 0, 0, 0, None, None, 0, 0, 0, 0))


@partial(jax.jit, donate_argnums=(0, 2, 3, 4))
def sor_block(phi, rhs, it, mc, done, om, op, exE, eyE, ezE, dvE, cA, cB):
    """phi/rhs [W,n,n,n]; it/mc/done/om/op [W]; coefficient table [E,n,n,n] x4. One 10-sweep block per lane."""
    return _lanes(phi, rhs, exE[op], eyE[op], ezE[op], dvE[op], cA, cB, om, it, mc, done)


@partial(jax.jit, static_argnames=("W",))
def lanes_init(phi0, rhs0, om0, item, W):
    """new pool of width W from the item arrays [NI,...]: lane l takes item[l] (item < 0 -> empty, done lane)."""
    ok = item >= 0
    ii = jnp.maximum(item, 0)
    phi = jnp.where(ok[:, None, None, None], phi0[ii], 0.0)
    rhs = jnp.where(ok[:, None, None, None], rhs0[ii], 0.0)
    return (phi, rhs, jnp.zeros(W, jnp.int32), jnp.full(W, 1e9, phi0.dtype), ~ok, jnp.where(ok, om0[ii], 1.0),
            jnp.zeros_like(phi0))


@partial(jax.jit, donate_argnums=(0,))
def lanes_store(store, phi, lane_item):
    """store[item] = phi[lane] for lanes with lane_item >= 0 (others dropped)."""
    tgt = jnp.where(lane_item >= 0, lane_item, store.shape[0])
    return store.at[tgt].set(phi, mode="drop")


@jax.jit
def lanes_take(phi, rhs, it, mc, done, om, op, idx):
    """compact: new lane k = old lane idx[k] (idx < 0 -> empty done lane)."""
    ok = idx >= 0
    i = jnp.maximum(idx, 0)
    z = lambda a, fill: jnp.where(ok.reshape((-1,) + (1,) * (a.ndim - 1)), a[i], fill)
    return (z(phi, 0.0), z(rhs, 0.0), z(it, 0), z(mc, 1e9), jnp.where(ok, done[i], True), z(om, 1.0), z(op, 0))


def width_for(n):
    for w in RC.LANE_WIDTHS:
        if n <= w:
            return w
    raise ValueError(f"{n} lanes > max width {RC.LANE_WIDTHS[-1]}")


def run_pool(phi0, rhs0, om0, op0, valid, coefs, stats=None):
    """Solve all valid items. phi0/rhs0 [NI,n,n,n] device, om0 [NI], op0 [NI] operator index, valid [NI] host bool,
    coefs = (exE, eyE, ezE, dvE) [E,n,n,n]. NI <= max lane width. Returns phi store [NI,n,n,n], sweeps [NI] (host),
    last max change [NI] (host)."""
    NI = phi0.shape[0]; n = phi0.shape[1]
    cA, cB = (jax.device_put(m) for m in colour_masks(n))
    items = np.nonzero(valid)[0]
    W = width_for(len(items))
    item = np.full(W, -1, np.int64); item[:len(items)] = items
    phi, rhs, it, mc, done, om, store = lanes_init(phi0, rhs0, om0, jax.device_put(item.astype(np.int32)), W=W)
    op = jax.device_put(np.where(item >= 0, np.asarray(op0)[np.maximum(item, 0)], 0).astype(np.int32))
    res_it = np.zeros(NI, np.int32); res_mc = np.zeros(NI, np.float32)
    lane_item = item.copy()
    blocks = 0; lane_blocks = 0
    while True:
        phi, it, mc, done = sor_block(phi, rhs, it, mc, done, om, op, *coefs, cA, cB)
        blocks += 1; lane_blocks += W
        d = np.asarray(done); it_h = np.asarray(it); mc_h = np.asarray(mc)
        fin = d & (lane_item >= 0)
        if fin.any():
            sel = np.where(fin, lane_item, -1)
            store = lanes_store(store, phi, jax.device_put(sel.astype(np.int32)))
            res_it[lane_item[fin]] = it_h[fin]; res_mc[lane_item[fin]] = mc_h[fin]
            lane_item = np.where(fin, -1, lane_item)
        live = np.nonzero(lane_item >= 0)[0]
        if len(live) == 0:
            break
        W2 = width_for(len(live))
        if W2 < W:
            idx = np.full(W2, -1, np.int64); idx[:len(live)] = live
            phi, rhs, it, mc, done, om, op = lanes_take(phi, rhs, it, mc, done, om, op, jax.device_put(idx.astype(np.int32)))
            lane_item = np.where(idx >= 0, lane_item[np.maximum(idx, 0)], -1)
            W = W2
    if stats is not None:
        stats.append(dict(blocks=blocks, lane_blocks=lane_blocks, needed=int(res_it[valid].sum() // RC.CHECK)))
    return store, res_it, res_mc


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
