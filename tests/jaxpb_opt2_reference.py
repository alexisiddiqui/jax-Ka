"""Unchanged 10a3ddc XLA SOR lane pool, for regression tests only."""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jaxpb import records as RC
from jaxpb.sor import colour_masks

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


