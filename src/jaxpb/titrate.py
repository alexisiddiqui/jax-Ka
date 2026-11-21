"""jaxpb titration (energy track, M4): PyPKA mc.pyx Metropolis on GPU in fixed Mp buckets, PyPKA pKhalf, cluster-MF fallback.

Input per state = PyPKA's MC energies (mc-energies.json semantics, kT): g [M] (possible_states_g), occ [M] 0/1
(possible_states_occ), W [M,M] f32 (interactions, same-site blocks zero), start/nst [n] (contiguous interactions_look
blocks), pairs [P,2] (mc.pyx select_pairs rule; assemble.select_pairs builds them).

MC: `chain` below is mf-surrogate-v2 gmc.chain VERBATIM (same move set, invexp, maintained local field rebuilt every
BLOCK sweeps, counter-hash RNG). Changes are outside the chain function only:
  * fixed buckets Mp in MP_BUCKETS, Np = Mp/4, Pp = Mp/16, S = 32 states per launch (padding states cost nothing:
    their block count is 0); a state that does not fit (n > Np or P > Pp) is promoted to the next bucket, and a state
    that fits no bucket is flagged and, if allowed, solved by the cluster-MF fallback (never silent);
  * the number of MC blocks is a per-state ARGUMENT (meta[:,2]), so one executable per Mp serves every sweep count;
  * u = ln10*pH*occ + g is formed inside the jit from a host-computed f32 pH table (bitwise equal to v2's numpy u:
    the product with occ in {0,1} is exact, so FMA contraction cannot change it).
Two backends with identical RNG streams: "ref" (pure XLA, vmapped, lockstep) and "pallas" (one Triton kernel, one
program per chain). Counts are bitwise reproducible across buckets (padding adds exact zeros only).

Pure functions: every jitted function is module level; shapes come from the arguments (bucket tables), no closures over
arrays. Host-side helpers are numpy.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from jaxpb import records as RC

NPH = RC.NPH
PH = [round(-2 + i * 0.25, 2) for i in range(NPH)]   # PyPKA pH grid (-2..16 step 0.25)
DPH = 0.25
LN10 = math.log(10)
LN10F = np.float32(2.302585092994)                  # mc.pyx: cdef float LN10
PH_TERM = (LN10F * np.asarray(PH, np.float32)).astype(np.float32)   # [73] f32, as v2 bucket_arrays
BLOCK = 1000                                        # E rebuilt every BLOCK sweeps; eq = 1 block
MP_BUCKETS = RC.MP_BUCKETS
S_BATCH = RC.S_FLUSH
COUPLE_MIN = 2.0
MAXNPKHALFS = 3                                      # pypka.constants
MF_TOL = 2e-5
CMF_CAP = 200_000


def bucket_dims(Mp):
    """(Mp, Np, Pp) of a titration bucket (records.py tables)."""
    return Mp, RC.np_bucket(Mp), RC.pp_bucket(Mp)


# One protonation problem = records.Energies (g [M] f32, occ [M] 0/1, W [M,M] f32 with zero same-site blocks, start/nst
# [n] i32, pairs [P,2] i32 in mc.pyx order); titrate() only reads those six attributes.
TitrationState = RC.Energies


# ------------------------------------------------------------------ RNG / acceptance (gmc.py, unchanged)
def hash32(x):
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x7feb352d)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(0x846ca68b)
    return x ^ (x >> 16)


def urand(key, ctr):
    h = hash32(key ^ hash32(ctr))
    return (h >> 8).astype(jnp.int32).astype(jnp.float32) * jnp.float32(1.0 / 16777216.0)


_U32 = np.uint64(0xFFFFFFFF)


def np_hash32(x):
    """gmc.np_hash32 with explicit uint64 constants (same values under numpy 1.x and 2.x promotion rules)."""
    x = np.asarray(x, dtype=np.uint64) & _U32
    x ^= x >> np.uint64(16); x = (x * np.uint64(0x7feb352d)) & _U32
    x ^= x >> np.uint64(15); x = (x * np.uint64(0x846ca68b)) & _U32
    return (x ^ (x >> np.uint64(16))).astype(np.uint32)


def chain_keys(seed, chain_ids):
    a = np_hash32(np.uint32(seed)).astype(np.uint64)
    b = np_hash32(np.asarray(chain_ids, np.uint64) * np.uint64(0x9E3779B9) & _U32).astype(np.uint64)
    return np_hash32(a ^ b).view(np.int32)


def invexp(x):
    """mc.pyx invexp: piecewise rational approximation of exp(-x), float32."""
    x2 = x * x; x3 = x2 * x; x4 = x2 * x2
    lin = jnp.float32(8.194236147130614e-10) - jnp.float32(1.3290994520804703e-11) * x
    mid = ((jnp.float32(-0.0013245823657199278) + jnp.float32(0.00027464252539452071) * x
            - jnp.float32(0.000019314947607346905) * x2 + jnp.float32(4.598224667374957e-7) * x3)
           / (jnp.float32(1.0) - jnp.float32(0.5165170691890946) * x + jnp.float32(0.09211442135429947) * x2
              - jnp.float32(0.006143102546214945) * x3))
    low = ((jnp.float32(0.9999965470613797) - jnp.float32(0.3960827416191208) * x + jnp.float32(0.06303500815508939) * x2
            - jnp.float32(0.00476617578304489) * x3 + jnp.float32(0.00014392025197088043) * x4)
           / (jnp.float32(1.0) + jnp.float32(0.6038220689877429) * x + jnp.float32(0.16732494517488303) * x2
              + jnp.float32(0.026354026827091058) * x3 + jnp.float32(0.00289071552898347) * x4))
    return jnp.where(x > 13, lin, jnp.where(x > 6, mid, low))


def accept(dU, r):
    return (dU <= 0) | ((dU <= 50) & (invexp(dU) > r))


# ------------------------------------------------------------------ one chain: gmc.chain VERBATIM
def chain(acc, key, Mp, Np, mcblocks):
    """acc: dict of accessors: row(r)->[Mp] f32, w(a,b)->f32, start(i), nst(i), pa(p), pb(p) -> i32 scalars,
    u -> [Mp] f32, start_vec -> [Np] i32, n, P -> i32 scalars. Returns counts [Mp] f32 (sweeps spent in each microstate)."""
    idx = lax.broadcasted_iota(jnp.int32, (Mp,), 0)
    sidx = lax.broadcasted_iota(jnp.int32, (Np,), 0)
    n, P = acc['n'], acc['P']
    one, zero = jnp.float32(1.0), jnp.float32(0.0)

    def onehot(m):
        return jnp.where(idx == m, one, zero)

    def pick(v, m):
        return jnp.sum(jnp.where(idx == m, v, zero))

    def cur_of(curv, i):
        return jnp.sum(jnp.where(sidx == i, curv, 0))

    def rebuild(curv):
        def body(i, c):
            E, X = c
            ci = cur_of(curv, i)
            return E + acc['row'](ci), X + onehot(ci)
        return lax.fori_loop(0, n, body, (acc['u'], jnp.zeros((Mp,), jnp.float32)))

    def site_move(i, c):
        E, X, curv, ctr = c
        st = acc['start'](i); k = acc['nst'](i)
        oc = cur_of(curv, i)
        r1 = urand(key, ctr); r2 = urand(key, ctr + jnp.uint32(1))
        nw = st + jnp.minimum((r1 * k.astype(jnp.float32)).astype(jnp.int32), k - 1)
        dU = jnp.sum(jnp.where(idx == nw, E, zero) - jnp.where(idx == oc, E, zero))
        ok = accept(dU, r2) & (nw != oc)

        def upd(c2):
            E, X, curv = c2
            return E + (acc['row'](nw) - acc['row'](oc)), X + (onehot(nw) - onehot(oc)), jnp.where(sidx == i, nw, curv)
        E, X, curv = lax.cond(ok, upd, lambda c2: c2, (E, X, curv))
        return E, X, curv, ctr + jnp.uint32(2)

    def pair_move(p, c):
        E, X, curv, ctr = c
        a = acc['pa'](p); b = acc['pb'](p)
        oa = cur_of(curv, a); ob = cur_of(curv, b)
        r1 = urand(key, ctr); r2 = urand(key, ctr + jnp.uint32(1)); r3 = urand(key, ctr + jnp.uint32(2))
        ka = acc['nst'](a); kb = acc['nst'](b)
        na = acc['start'](a) + jnp.minimum((r1 * ka.astype(jnp.float32)).astype(jnp.int32), ka - 1)
        nb = acc['start'](b) + jnp.minimum((r2 * kb.astype(jnp.float32)).astype(jnp.int32), kb - 1)
        dU = jnp.sum(jnp.where(idx == na, E, zero) - jnp.where(idx == oa, E, zero)
                     + jnp.where(idx == nb, E, zero) - jnp.where(idx == ob, E, zero))
        dU = dU + acc['w'](na, nb) - acc['w'](na, ob) - acc['w'](oa, nb) + acc['w'](oa, ob)
        ok = accept(dU, r3) & ((na != oa) | (nb != ob))

        def upd(c2):
            E, X, curv = c2
            E = E + ((acc['row'](na) - acc['row'](oa)) + (acc['row'](nb) - acc['row'](ob)))
            X = X + ((onehot(na) - onehot(oa)) + (onehot(nb) - onehot(ob)))
            curv = jnp.where(sidx == a, na, jnp.where(sidx == b, nb, curv))
            return E, X, curv
        E, X, curv = lax.cond(ok, upd, lambda c2: c2, (E, X, curv))
        return E, X, curv, ctr + jnp.uint32(3)

    def block(b, c):
        curv, cnt, ctr = c
        E, X = rebuild(curv)
        flag = jnp.where(b > 0, one, zero)

        def sweep(t, c3):
            E, X, curv, cnt, ctr = c3
            E, X, curv, ctr = lax.fori_loop(0, n, site_move, (E, X, curv, ctr))
            E, X, curv, ctr = lax.fori_loop(0, P, pair_move, (E, X, curv, ctr))
            return E, X, curv, cnt + flag * X, ctr
        E, X, curv, cnt, ctr = lax.fori_loop(0, BLOCK, sweep, (E, X, curv, cnt, ctr))
        return curv, cnt, ctr
    curv0 = acc['start_vec']
    _, cnt, _ = lax.fori_loop(0, 1 + mcblocks, block, (curv0, jnp.zeros((Mp,), jnp.float32), jnp.uint32(0)))
    return cnt


# ------------------------------------------------------------------ jitted MC entry points (module level, argument-pure)
def _u(ph_term, g, occ):
    """u[s, j, m] = ln10*pH_j*occ + g (f32; exact product since occ in {0,1})."""
    return ph_term[None, :, None] * occ[:, None, :] + g[:, None, :]


@jax.jit
def mc_ref(ph_term, g, occ, W, start, nst, pa, pb, meta, keys):
    """Pure-XLA lockstep MC. g/occ [S,Mp] f32, W [S,Mp,Mp], start/nst [S,Np], pa/pb [S,Pp], meta [S,3] = (n, P, mcblocks;
    mcblocks = -1 for padding), keys [S,73] i32. Returns counts [S,73,Mp] f32."""
    Mp, Np = W.shape[1], start.shape[1]
    u = _u(ph_term, g, occ)

    def one(W, u, start, nst, pa, pb, meta, key):
        acc = dict(row=lambda r: W[r], w=lambda a, b: W[a, b], start=lambda i: start[i], nst=lambda i: nst[i],
                   pa=lambda p: pa[p], pb=lambda p: pb[p], u=u, start_vec=start, n=meta[0], P=meta[1])
        return chain(acc, key.astype(jnp.uint32), Mp, Np, meta[2])
    f = jax.vmap(jax.vmap(one, in_axes=(None, 0, None, None, None, None, None, 0)), in_axes=(0, 0, 0, 0, 0, 0, 0, 0))
    return f(W, u, start, nst, pa, pb, meta, keys)


def _triton_params(num_warps):
    from jax.experimental.pallas import triton as pltriton
    cls = getattr(pltriton, 'CompilerParams', None) or getattr(pltriton, 'TritonCompilerParams')
    return cls(num_warps=num_warps, num_stages=1)


def _mc_kernel(key_ref, u_ref, W_ref, start_ref, nst_ref, pa_ref, pb_ref, meta_ref, out_ref):
    Mp = W_ref.shape[-1]; Np = start_ref.shape[-1]
    acc = dict(row=lambda r: W_ref[r, :], w=lambda a, b: W_ref[a, b], start=lambda i: start_ref[i],
               nst=lambda i: nst_ref[i], pa=lambda p: pa_ref[p], pb=lambda p: pb_ref[p], u=u_ref[:],
               start_vec=start_ref[:], n=meta_ref[0], P=meta_ref[1])
    out_ref[:] = chain(acc, key_ref[0].astype(jnp.uint32), Mp, Np, meta_ref[2])


def _mc_pallas_impl(ph_term, g, occ, W, start, nst, pa, pb, meta, keys, interpret=False, num_warps=1):
    S, Mp = W.shape[:2]
    Np, Pp = start.shape[1], pa.shape[1]
    C = S * NPH
    u = _u(ph_term, g, occ).reshape(C, Mp)
    st = lambda c: c // NPH  # noqa: E731
    kw = {} if interpret else dict(compiler_params=_triton_params(num_warps))
    f = pl.pallas_call(
        _mc_kernel, grid=(C,),
        in_specs=[pl.BlockSpec((1,), lambda c: (c,)),
                  pl.BlockSpec((None, Mp), lambda c: (c, 0)),
                  pl.BlockSpec((None, Mp, Mp), lambda c: (st(c), 0, 0)),
                  pl.BlockSpec((None, Np), lambda c: (st(c), 0)),
                  pl.BlockSpec((None, Np), lambda c: (st(c), 0)),
                  pl.BlockSpec((None, Pp), lambda c: (st(c), 0)),
                  pl.BlockSpec((None, Pp), lambda c: (st(c), 0)),
                  pl.BlockSpec((None, 3), lambda c: (st(c), 0))],
        out_specs=pl.BlockSpec((None, Mp), lambda c: (c, 0)),
        out_shape=jax.ShapeDtypeStruct((C, Mp), jnp.float32),
        interpret=interpret, name='pypka_mc', **kw)
    out = f(keys.reshape(C), u, W, start, nst, pa, pb, meta)
    return out.reshape(S, NPH, Mp)


mc_pallas = jax.jit(_mc_pallas_impl, static_argnames=('interpret', 'num_warps'))


def pallas_available():
    """True when the default backend is a GPU on which the Triton Pallas lowering imports."""
    try:
        if jax.default_backend() != 'gpu':
            return False
        _triton_params(1)
        return True
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ bucketing (host)
def fit_bucket(M, n, P, buckets=MP_BUCKETS):
    """Smallest bucket Mp with M <= Mp, n <= Np, P <= Pp; None if none fits. Second value: promoted (did not fit the
    bucket its M alone would choose)."""
    first = None
    for Mp in buckets:
        _, Np, Pp = bucket_dims(Mp)
        if M <= Mp and first is None:
            first = Mp
        if M <= Mp and n <= Np and P <= Pp:
            return Mp, Mp != first
    return None, True


def pack_bucket(states, ids, Mp, mcblocks, keys, S=S_BATCH, NpPp=None):
    """Pad up to S states into one Mp bucket. ids: indices into states (len <= S); keys [len(ids),73] i32.
    NpPp: (Np, Pp) override (tests only; production shapes come from bucket_dims)."""
    _, Np, Pp = bucket_dims(Mp) if NpPp is None else (Mp,) + tuple(NpPp)
    assert len(ids) <= S
    g = np.zeros((S, Mp), np.float32); occ = np.zeros((S, Mp), np.float32)
    W = np.zeros((S, Mp, Mp), np.float32)
    start = np.zeros((S, Np), np.int32); nst = np.ones((S, Np), np.int32)
    pa = np.zeros((S, Pp), np.int32); pb = np.zeros((S, Pp), np.int32)
    meta = np.zeros((S, 3), np.int32); meta[:, 2] = -1        # padding: 0 blocks -> no work
    kk = np.zeros((S, NPH), np.int32)
    for k, i in enumerate(ids):
        s = states[i]
        M, n, P = len(s.g), len(s.nst), len(s.pairs)
        assert M <= Mp and n <= Np and P <= Pp, ('state does not fit bucket', M, n, P, Mp)
        g[k, :M] = s.g; occ[k, :M] = s.occ; W[k, :M, :M] = s.W
        start[k, :n] = s.start; nst[k, :n] = s.nst
        if P:
            pa[k, :P] = s.pairs[:, 0]; pb[k, :P] = s.pairs[:, 1]
        meta[k] = (n, P, mcblocks)
        kk[k] = keys[k]
    return dict(g=g, occ=occ, W=W, start=start, nst=nst, pa=pa, pb=pb, meta=meta, keys=kk)


def counts_to_prot(state, counts):
    """counts [73, Mp] -> (prot [73,n] int64 = PyPKA avgs, tot [73,n] int64); exactly v2 gmc.to_avgs."""
    occ, start = np.asarray(state.occ), np.asarray(state.start)
    c = np.asarray(counts, np.float64)[:, :len(occ)]
    prot = np.add.reduceat(c * occ[None, :], start, axis=1)
    tot = np.add.reduceat(c, start, axis=1)
    return np.rint(prot).astype(np.int64), np.rint(tot).astype(np.int64)


# ------------------------------------------------------------------ pKhalf (pypka.mc.run_mc.calcpKhalfs + getpK)
def curves_to_pka(curves, ph=PH, dph=DPH):
    """curves [73, n] (avgs/mcsteps, f64). Returns [n] f64, nan where PyPKA's pK is None. First crossing of 0.5 (or,
    elif, 1.5) in pH order, pKhalf = pH - dpH*(mean - x)/(mean - p): pKs[site][0] as Site.getpK."""
    curves = np.asarray(curves, np.float64)
    out = np.full(curves.shape[1], np.nan)
    for k in range(1, len(ph)):
        p, mean = curves[k - 1], curves[k]
        c05 = ((p > 0.5) & (mean <= 0.5)) | ((p < 0.5) & (mean >= 0.5))
        c15 = ~c05 & (((p > 1.5) & (mean <= 1.5)) | ((p < 1.5) & (mean >= 1.5)))
        with np.errstate(divide='ignore', invalid='ignore'):
            v05 = ph[k] - dph * (mean - 0.5) / (mean - p)
            v15 = ph[k] - dph * (mean - 1.5) / (mean - p)
        hit = np.isnan(out) & (c05 | c15)
        out[hit] = np.where(c05, v05, v15)[hit]
    return out


@jax.jit
def curves_to_pka_jax(curves, ph):
    """Device version of curves_to_pka. curves [..., 73, n], ph [73] (same dtype; use f64 under x64 for PyPKA identity)."""
    dph = ph[1] - ph[0]
    p, mean = curves[..., :-1, :], curves[..., 1:, :]
    x = ph[1:, None]
    c05 = ((p > 0.5) & (mean <= 0.5)) | ((p < 0.5) & (mean >= 0.5))
    c15 = ~c05 & (((p > 1.5) & (mean <= 1.5)) | ((p < 1.5) & (mean >= 1.5)))
    v = jnp.where(c05, x - dph * (mean - 0.5) / (mean - p), x - dph * (mean - 1.5) / (mean - p))
    hit = c05 | c15
    first = jnp.argmax(hit, axis=-2)
    val = jnp.take_along_axis(v, first[..., None, :], axis=-2)[..., 0, :]
    return jnp.where(hit.any(axis=-2), val, jnp.nan)


# ------------------------------------------------------------------ exact-cluster mean field (v2 cmf.py, host numpy)
def select_pairs_np(W32, start, nst, couple_min=COUPLE_MIN):
    """mc.pyx select_pairs: pair (i<j) if any |W[s,t]| >= LN10*couple_min (float32)."""
    thr = np.float32(LN10F * np.float32(couple_min))
    A = np.abs(np.asarray(W32, np.float32)) >= thr
    out = []
    for i in range(len(start)):
        row = A[start[i]:start[i] + nst[i]]
        for j in range(i + 1, len(start)):
            if row[:, start[j]:start[j] + nst[j]].any():
                out.append((i, j))
    return np.array(out, dtype=np.int32).reshape(-1, 2)


def _components(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comp = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)
    return sorted(comp.values(), key=lambda c: c[0])


def _make_clusters(n, start, nst, W, pairs, cap):
    strength = {(int(a), int(b)): float(np.abs(W[start[a]:start[a] + nst[a], start[b]:start[b] + nst[b]]).max())
                for a, b in pairs}
    todo = [(c, [e for e in strength if e[0] in c and e[1] in c]) for c in
            (set(x) for x in _components(n, list(strength)))]
    out, nsplit, removed = [], 0, 0
    while todo:
        c, edges = todo.pop()
        if np.prod([float(nst[i]) for i in c]) <= cap:
            out.append(sorted(c)); continue
        nsplit += 1
        edges = sorted(edges, key=lambda e: strength[e])
        loc = sorted(c); idx = {s: k for k, s in enumerate(loc)}
        while True:
            edges = edges[1:]; removed += 1
            comps = _components(len(loc), [(idx[a], idx[b]) for a, b in edges])
            if len(comps) > 1:
                break
        for cc in comps:
            sub = {loc[k] for k in cc}
            todo.append((sub, [e for e in edges if e[0] in sub and e[1] in sub]))
    out.sort(key=lambda c: c[0])
    return out, dict(n_split_clusters=nsplit, links_removed=removed)


def cluster_mf(state, ph=PH, tol=1e-10, report_tol=MF_TOL, max_iter=20000, cap=CMF_CAP):
    """Arm C of mf-surrogate-v2 (cmf.cluster_mf), deterministic. Clusters = components of select_pairs pairs, each
    enumerated exactly in the mean field of the rest; split at the weakest link above `cap` joint states."""
    start = np.asarray(state.start, int); nst = np.asarray(state.nst, int); n = len(start)
    occ = np.asarray(state.occ, float); g = np.asarray(state.g, float); W = np.asarray(state.W, float)
    M = len(g); site_of = np.repeat(np.arange(n), nst)
    pairs = np.asarray(state.pairs).reshape(-1, 2)
    clusters, split = _make_clusters(n, start, nst, W, pairs, cap)
    multi = [c for c in clusters if len(c) > 1]
    clus_of = np.empty(n, int)
    for k, c in enumerate(clusters):
        clus_of[c] = k
    cm = clus_of[site_of]
    Wout = np.where(cm[:, None] == cm[None, :], 0.0, W)
    enum = []
    for c in multi:
        conf = np.array(list(itertools.product(*[range(start[i], start[i] + nst[i]) for i in c])), dtype=np.int64)
        eint = np.zeros(len(conf))
        for a, b in itertools.combinations(range(len(c)), 2):
            eint += W[conf[:, a], conf[:, b]]
        mask = np.zeros(M, bool)
        for i in c:
            mask[start[i]:start[i] + nst[i]] = True
        enum.append((conf, eint, mask))

    def softmax_seg(a):
        mx = np.maximum.reduceat(a, start)
        ex = np.exp(a - mx[site_of])
        return ex / np.add.reduceat(ex, start)[site_of]

    def target(p, x):
        v = LN10 * x * occ + g + Wout @ p
        t = softmax_seg(-v)
        for conf, eint, mask in enum:
            E = eint + v[conf].sum(1)
            w = np.exp(-(E - E.min()))
            w /= w.sum()
            tt = np.zeros(len(v))
            for a in range(conf.shape[1]):
                tt += np.bincount(conf[:, a], weights=w, minlength=len(v))
            t[mask] = tt[mask]
        return t

    def sweep(order):
        curves = np.zeros((len(ph), n)); res = np.zeros(len(ph)); its = np.zeros(len(ph), dtype=int)
        p = None
        for k in order:
            x = ph[k]
            if p is None:
                p = softmax_seg(-(LN10 * x * occ + g))
            alpha = 1.0; r_old = np.inf; it = 0
            while True:
                t = target(p, x); r = np.abs(t - p).max()
                if r < tol or it >= max_iter:
                    break
                if r > r_old:
                    alpha = max(alpha * 0.5, 0.01)
                else:
                    alpha = min(alpha * 1.05, 1.0)
                r_old = r
                p = p + alpha * (t - p); it += 1
            res[k] = r; its[k] = it
            curves[k] = np.add.reduceat(p * occ, start)
        return curves, res, its

    up, rup, iup = sweep(range(len(ph)))
    dn, rdn, idn = sweep(range(len(ph) - 1, -1, -1))
    sizes = [len(c) for c in clusters]
    nconf = [len(e[0]) for e in enum]
    return dict(curve=up, curve_down=dn, residual=rup, residual_down=rdn, iters=iup, iters_down=idn,
                converged=rup < report_tol, converged_down=rdn < report_tol, hysteresis=np.abs(up - dn).max(0),
                n_multi=len(multi), max_cluster_sites=max(sizes) if sizes else 0,
                max_conf=max(nconf) if nconf else 1, **split)


# ------------------------------------------------------------------ driver
def titrate(states, *, mc_sweeps=200_000, seed=1, chain_ids=None, backend='auto', buckets=MP_BUCKETS, S=S_BATCH,
            fallback='cmf', on_batch=None):
    """Titrate a list of TitrationState. chain_ids[k]: stable integer id of state k (default k); chain c of state k at
    pH index j uses key chain_keys(seed, chain_ids[k]*73 + j) (v2 convention). backend: 'pallas' | 'ref' | 'auto' |
    'cmf' (cluster MF only). States that fit no bucket go to `fallback` ('cmf' or None -> flagged, no result).
    Returns list of dicts: method, Mp, promoted, prot/tot [73,n] int64 (MC), curves [73,n] f64, pka [n] f64 (nan = None),
    plus per-bucket timing in titrate.last_timing."""
    assert mc_sweeps % BLOCK == 0 and 0 < mc_sweeps < (1 << 24), 'sweeps: multiple of 1000, below 2^24 (exact f32 counts)'
    if backend == 'auto':
        backend = 'pallas' if pallas_available() else 'ref'
    chain_ids = np.arange(len(states)) if chain_ids is None else np.asarray(chain_ids)
    out = [None] * len(states)
    timing = []
    groups = {}
    for k, s in enumerate(states):
        if backend == 'cmf':
            Mp, promoted = None, False
        else:
            Mp, promoted = fit_bucket(len(s.g), len(s.nst), len(s.pairs), buckets)
        if Mp is None:
            if backend != 'cmf' and fallback != 'cmf':
                out[k] = dict(method=None, flag='no_bucket', Mp=None, promoted=True)
                continue
            r = cluster_mf(s)
            out[k] = dict(method='cmf', flag=None if backend == 'cmf' else 'no_bucket', Mp=None, promoted=promoted,
                          curves=r['curve'], pka=curves_to_pka(r['curve']), cmf=r)
            continue
        groups.setdefault(Mp, []).append((k, promoted))
    fn = mc_pallas if backend == 'pallas' else mc_ref
    ph_term = jax.device_put(PH_TERM)   # not jnp.asarray: JAX 0.11 compiles a jit(stage) program for host->device asarray
    mcb = mc_sweeps // BLOCK
    for Mp in sorted(groups):
        mem = sorted(groups[Mp], key=lambda r: -(len(states[r[0]].nst) + len(states[r[0]].pairs)))
        for b0 in range(0, len(mem), S):
            chunk = mem[b0:b0 + S]
            ids = [k for k, _ in chunk]
            keys = chain_keys(seed, chain_ids[ids][:, None].astype(np.int64) * NPH + np.arange(NPH)).reshape(len(ids), NPH)
            B = pack_bucket(states, ids, Mp, mcb, keys, S)
            t0 = time.time()
            cnt = fn(ph_term, B['g'], B['occ'], B['W'], B['start'], B['nst'], B['pa'], B['pb'], B['meta'], B['keys'])
            cnt = np.asarray(jax.block_until_ready(cnt))
            timing.append(dict(Mp=Mp, states=len(ids), seconds=time.time() - t0))
            for j, (k, promoted) in enumerate(chunk):
                prot, tot = counts_to_prot(states[k], cnt[j])
                if not (tot == mc_sweeps).all():
                    out[k] = dict(method='mc', flag='count_mismatch', Mp=Mp, promoted=promoted)
                    continue
                curves = prot / mc_sweeps
                out[k] = dict(method='mc', flag=None, Mp=Mp, promoted=promoted, prot=prot, tot=tot,
                              curves=curves, pka=curves_to_pka(curves))
            if on_batch is not None:
                on_batch(timing[-1])
    titrate.last_timing = timing
    return out


titrate.last_timing = []


# ------------------------------------------------------------------ toy systems (tests)
def toy(nst, seed, scale):
    rng = np.random.default_rng(seed)
    n = len(nst); start = np.concatenate([[0], np.cumsum(nst)[:-1]]).astype(np.int32)
    site_of = np.repeat(np.arange(n), nst)
    occ = np.concatenate([[1] * (k - 1) + [0] for k in nst]).astype(np.int8)
    g = np.concatenate([rng.normal(-2.302585 * 5, 3, k - 1).tolist() + [0.0] for k in nst]).astype(np.float32)
    W = rng.normal(0, scale, (int(sum(nst)),) * 2); W = (W + W.T) / 2
    W[site_of[:, None] == site_of[None, :]] = 0
    W = W.astype(np.float32)
    st = np.array(nst, np.int32)
    return TitrationState(g=g, occ=occ, W=W, start=start, nst=st, pairs=select_pairs_np(W, start, st))


def brute(state):
    n = len(state.nst); occ = state.occ.astype(float); g = state.g.astype(float); W = state.W.astype(float)
    confs = np.array(list(itertools.product(*[range(state.start[i], state.start[i] + state.nst[i]) for i in range(n)])))
    eint = sum(W[confs[:, a], confs[:, b]] for a, b in itertools.combinations(range(n), 2))
    out = np.zeros((NPH, n))
    for k, x in enumerate(PH):
        E = eint + (LN10 * x * occ + g)[confs].sum(1)
        w = np.exp(-(E - E.min())); w /= w.sum()
        out[k] = (occ[confs] * w[:, None]).sum(0)
    return out
