"""jaxpb.pb -- DelPhi-mode PB for StateRecords: geometry -> maps -> SOR lane pool -> energies / site potentials -> PBOut.

Per tautomer, PyPKA's three DelPhi calls (jaxpb-pilot report section 1): whole-molecule coarse (h 1 A, Debye-screened
Coulomb BC, fixed spectral radius 0.75), whole-molecule focused (h 0.25 A, faces interpolated from the coarse map, spectral
radius from DelPhi's estimator -> M2; M1 uses the logged value), model compound (h 0.25 A, Coulomb BC). Energies:
esolvM = raw reaction-field energy of the focused run (no Gauss rescale), esolvS = Gauss-corrected model energy,
esolvM0 = raw coarse energy (diagnostic); sitpotM = focused potential at every atom, coarse value where the focused grid
gives 0 (v2 jaxrun2._post). Numerics = jaxpb-pilot v2.

Jitted programs (all module level, argument-pure, static shapes from records.py):
  geometry.geo_* [c], geometry.geometry_sites, maps.maps_batch[c, Sizes], setup1, setup2, sor.* (lane pool), post[c].
DEVIATIONS from DESIGN.md (reasons in the M1 report): x64 only inside geometry (process-wide x64 would promote the f32
map/PB code through weak-typed scalars); coarse+fine+model maps in ONE per-class program (fine/model read the class atom
and probe arrays); site potentials computed for all class-c atoms in post[c]; lane pool = per-phase pool with width
compaction (16/64/256), no cross-batch refill (a phase never has more than 2*B*T_MAX = 256 lanes).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Callable, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from . import records as RC
from . import geometry as GEO
from . import maps as MP
from . import sor as SOR

N = RC.GSIZE
EPKT = RC.EPKT
EPSIN, EPSOUT, CONC = 15.0, 80.0, 0.1
DEBL = RC.DEBYE_I1 / np.sqrt(CONC)
DT = jnp.float32


# ------------------------------------------------------------------------------------------------ grid helpers (v1)
def face_mask(n):
    return jnp.asarray(SOR.face_mask_np(n))


def spread_scatter(q_pos, q, acent, h, n):
    """v1/v2 trilinear charge spreading by float scatter-add. On GPU the scatter-add uses atomics, so a grid point with
    >= 3 contributions gets a run-to-run varying float32 sum (the M1 pass-to-pass differences). Kept only as the M5
    determinism reference (tests/test_determinism.py); spread() is the pipeline version."""
    g = (q_pos - acent) / h + (n - 1) / 2
    i0 = jnp.floor(g).astype(jnp.int32); f = g - i0
    rho = jnp.zeros((n, n, n), q.dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wgt = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                ii = jnp.clip(i0 + jnp.array([dx, dy, dz]), 0, n - 1)
                rho = rho.at[ii[:, 0], ii[:, 1], ii[:, 2]].add(wgt * q)
    return rho


def spread(q_pos, q, acent, h, n):
    """Deterministic trilinear charge spreading (M5 part A) with spread_scatter's SEQUENTIAL summation order. The K = 8*QM
    contributions are taken in the v1/v2 loop order (corner-major, atom-minor) and folded in that order on a [K] vector:
    acc_i += V_j for every i on the same grid point as j (j = 0..K-1, fixed trip count), so acc_i ends as the sequential
    sum of its grid point. The result is written with a scatter-SET in which every duplicate writes identical bits, so
    the GPU atomic order cannot change anything; bitwise equal to spread_scatter on the CPU."""
    g = (q_pos - acent) / h + (n - 1) / 2
    i0 = jnp.floor(g).astype(jnp.int32); f = g - i0
    F, V = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wgt = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                ii = jnp.clip(i0 + jnp.array([dx, dy, dz]), 0, n - 1)
                F.append((ii[:, 0] * n + ii[:, 1]) * n + ii[:, 2]); V.append(wgt * q)
    F = jnp.concatenate(F); V = jnp.concatenate(V)

    def step(j, acc):
        return jnp.where(F == F[j], acc + V[j], acc)
    acc = jax.lax.fori_loop(0, F.shape[0], step, jnp.zeros_like(V), unroll=16)
    return jnp.zeros(n ** 3, q.dtype).at[F].set(acc).reshape(n, n, n)


def interp(phi, pts, acent, h, n):
    g = (pts - acent) / h + (n - 1) / 2
    inside = jnp.all((g >= 0) & (g <= n - 1), axis=1)
    g = jnp.clip(g, 0, n - 1 - 1e-6)
    i0 = jnp.floor(g).astype(jnp.int32); f = g - i0
    val = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wgt = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                ii = jnp.minimum(i0 + jnp.array([dx, dy, dz]), n - 1)
                val = val + wgt * phi[ii[:, 0], ii[:, 1], ii[:, 2]]
    return jnp.where(inside, val, 0.0)


@lru_cache(maxsize=4)
def face_indices(n):
    """Fixed, unique C-order face list (38,402 entries at n=81)."""
    return np.flatnonzero(SOR.face_mask_np(n).reshape(-1)).astype(np.int32)


def face_points(acent, h, n):
    # Use the original grid coordinate arithmetic before gathering; do not reassociate it.
    x, y, z = MP.grid_coords(acent, h, n)
    f = jnp.asarray(face_indices(n))
    return jnp.stack([x[f // (n * n)], y[(f // n) % n], z[f % n]], -1)


def face_grid(values, n):
    return jnp.zeros(n ** 3, values.dtype).at[jnp.asarray(face_indices(n))].set(values).reshape(n, n, n)


def coulomb_bc(q_pos, q, acent, h, n, epsout, deblen):
    G = face_points(acent, h, n)
    # Tile only points: the charge axis and its original jnp.sum reduction are unchanged.
    chunk = 2048
    pad = (-G.shape[0]) % chunk
    points = jnp.concatenate([G, jnp.zeros((pad, 3), G.dtype)]).reshape(-1, chunk, 3)

    def one(g):
        r = jnp.linalg.norm(g[..., None, :] - q_pos, axis=-1)
        return jnp.sum(EPKT * q * jnp.exp(-r / deblen) / (epsout * r), axis=-1)
    values = jax.lax.map(one, points).reshape(-1)[:G.shape[0]]
    return face_grid(values, n)


def operator_coeffs(maps, epsin, epsout, kap2h2):
    ex = jnp.where(maps["ex"], epsin, epsout); ey = jnp.where(maps["ey"], epsin, epsout); ez = jnp.where(maps["ez"], epsin, epsout)
    diag = ex + jnp.roll(ex, 1, 0) + ey + jnp.roll(ey, 1, 1) + ez + jnp.roll(ez, 1, 2) + kap2h2 * epsout * maps["ion"]
    return ex, ey, ez, diag


def tsum(x, axis=-1):
    """Sum along `axis` by a fixed pairwise tree of ELEMENTWISE adds (zero-padded to a power of two). XLA may tile a
    reduce differently in every compile (the M5 fresh-compile esolvS differences); elementwise adds cannot be
    reassociated, so this order is the same in every executable."""
    x = jnp.moveaxis(x, axis, -1)
    n = x.shape[-1]; p = 1 << max(n - 1, 0).bit_length()
    if p != n:
        x = jnp.concatenate([x, jnp.zeros(x.shape[:-1] + (p - n,), x.dtype)], -1)
    while x.shape[-1] > 1:
        x = x.reshape(x.shape[:-1] + (x.shape[-1] // 2, 2))
        x = x[..., 0] + x[..., 1]
    return x[..., 0]


def rf_energy(phi, bidx, surf, rho, q_pos, q, h, epsin, epsout, sum_=tsum):
    """DelPhi reaction-field energy from boundary polarisation charges on the molecular surface (v1 rf_energy).
    Returns (E_raw, E_corrected-to-Gauss). All sums are fixed-order trees (tsum); sum_=jnp.sum gives the M1 version
    (kept for the determinism probe only)."""
    lap = (jnp.roll(phi, -1, 0) + jnp.roll(phi, 1, 0) + jnp.roll(phi, -1, 1) + jnp.roll(phi, 1, 1)
           + jnp.roll(phi, -1, 2) + jnp.roll(phi, 1, 2) - 6 * phi)
    sig0 = (-(h / (4 * jnp.pi * EPKT)) * lap).reshape(-1); rhof = rho.reshape(-1)
    b = bidx; bb = jnp.maximum(b, 0)
    s0 = jnp.where(b >= 0, sig0[bb], 0.0); qb = jnp.where(b >= 0, rhof[bb], 0.0)
    sv = s0 - qb / epsin
    tot = sum_(sv)
    target = sum_(q) * (1 / epsout - 1 / epsin)
    d = jnp.linalg.norm(q_pos[:, None, :] - surf[None], axis=-1)
    G = EPKT / jnp.maximum(d, 1e-3)

    def E(svv):
        return 0.5 * sum_(q * sum_(G * svv[None], axis=1))
    okc = (jnp.abs(target) > 1e-6) & (jnp.abs(tot) > 1e-8)
    scale = jnp.where(okc, target / jnp.where(jnp.abs(tot) > 1e-8, tot, 1.0), 1.0)
    return E(sv), E(sv * scale)


# ------------------------------------------------------------------------------------------------ setup / post
def _kap2h2(h):
    return jnp.where(CONC > 0, (h / (RC.DEBYE_I1 / jnp.sqrt(jnp.maximum(CONC, 1e-12)))) ** 2, 0.0)


def _coefs(maps, h):
    fm = face_mask(N)
    ex, ey, ez, diag = operator_coeffs(maps, EPSIN, EPSOUT, _kap2h2(h))
    return ex, ey, ez, jnp.where(fm, 0.0, 1.0 / diag)


def _codes(maps, h):
    """Pallas SOR operator: uint8 code [n^3] + 1/diag table [16] (sor.op_code / sor.op_lut)."""
    return SOR.op_code(maps), SOR.op_lut(_kap2h2(h), EPSIN, EPSOUT)


def _rhs_setup(qp, q, acent, h, spec, bc_phi):
    fm = face_mask(N)
    rho = spread(qp, q, acent, h, N)
    src = 4 * jnp.pi * EPKT * rho / h
    omega = jnp.asarray(-1.0, DT)
    om = jnp.where(omega > 0, omega, 2.0 / (1.0 + jnp.sqrt(1.0 - spec))).astype(DT)
    return jnp.where(fm, bc_phi, 0.0), jnp.where(fm, 0.0, src), om


@partial(jax.jit, static_argnames=("backend",))
def setup1(mc, ms, acent, qp, q, sqp, sq, spec, backend="xla"):
    """phase 1 items: coarse (B x T) then model (B x T), flattened to [2BT]; coefficient table [2B] (coarse, model) for
    the selected backend only: XLA coefficients or Pallas operator codes."""
    if backend == "xla":
        cc = jax.vmap(lambda m: _coefs(m, 1.0))(mc); cs = jax.vmap(lambda m: _coefs(m, 0.25))(ms)
        kc = ks = ()
    elif backend == "pallas":
        kc = jax.vmap(lambda m: _codes(m, 1.0))(mc); ks = jax.vmap(lambda m: _codes(m, 0.25))(ms)
        cc = cs = ()
    else:
        raise ValueError(backend)

    def coarse(a, qp_, q_, sp):
        return _rhs_setup(qp_, q_, a, 1.0, sp, coulomb_bc(qp_, q_, a, 1.0, N, EPSOUT, DEBL))

    def model(a, qp_, q_, sp):
        return _rhs_setup(qp_, q_, a, 0.25, sp, coulomb_bc(qp_, q_, a, 0.25, N, EPSOUT, DEBL))
    tv = lambda f: jax.vmap(jax.vmap(f, in_axes=(None, 0, 0, 0)))
    c = tv(coarse)(acent, qp, q, spec[..., 0]); s_ = tv(model)(acent, sqp, sq, spec[..., 2])
    items = tuple(jnp.concatenate([x, y]).reshape((-1,) + x.shape[2:]) for x, y in zip(c, s_))
    return items, tuple(jnp.concatenate([x, y]) for x, y in zip(cc, cs)), tuple(jnp.concatenate([x, y]) for x, y in zip(kc, ks))


@partial(jax.jit, static_argnames=("backend",))
def setup2(mf, acent, qp, q, spec, store1, backend="xla"):
    """phase 2 items: focused (B x T) padded to [2BT]; faces from the coarse potentials (store1[:BT]). Coefficient
    table and codes padded to 2B entries (same shapes as phase 1: one SOR executable per lane width for both phases)."""
    B_, T_ = qp.shape[:2]
    phic = store1[:B_ * T_].reshape((B_, T_) + store1.shape[1:])
    pad2 = lambda x: jnp.concatenate([x, jnp.zeros_like(x)])
    if backend == "xla":
        cf = tuple(pad2(x) for x in jax.vmap(lambda m: _coefs(m, 0.25))(mf)); kf = ()
    elif backend == "pallas":
        kf = tuple(pad2(x) for x in jax.vmap(lambda m: _codes(m, 0.25))(mf)); cf = ()
    else:
        raise ValueError(backend)

    def fine(a, qp_, q_, sp, pc):
        G = face_points(a, 0.25, N)
        bcf = face_grid(interp(pc, G, a, 1.0, N), N)
        return _rhs_setup(qp_, q_, a, 0.25, sp, bcf)
    it = jax.vmap(jax.vmap(fine, in_axes=(None, 0, 0, 0, 0)))(acent, qp, q, spec[..., 1], phic)
    items = tuple(jnp.concatenate([x.reshape((-1,) + x.shape[2:]), jnp.zeros_like(x.reshape((-1,) + x.shape[2:]))]) for x in it)
    return items, cf, kf


@jax.jit
def post(mc, mf, ms, acent, qp, q, sqp, sq, store1, store2, pos_all):
    """energies and site potentials per (site, tautomer). pos_all [NATOM_ALL_C[c],3] (FAR-padded)."""
    B_, T_ = qp.shape[:2]
    sh = store1.shape[1:]
    phic = store1[:B_ * T_].reshape((B_, T_) + sh); phis = store1[B_ * T_:2 * B_ * T_].reshape((B_, T_) + sh)
    phif = store2[:B_ * T_].reshape((B_, T_) + sh)

    def one(bc, sc, bf, sf_, bs, ss, a, qp_, q_, sqp_, sq_, pc, pf, ps):
        rhoc = spread(qp_, q_, a, 1.0, N); rhof = spread(qp_, q_, a, 0.25, N); rhos = spread(sqp_, sq_, a, 0.25, N)
        eM, _ = rf_energy(pf, bf, sf_, rhof, qp_, q_, 0.25, EPSIN, EPSOUT)
        eM0, _ = rf_energy(pc, bc, sc, rhoc, qp_, q_, 1.0, EPSIN, EPSOUT)
        sfi = interp(pf, pos_all, a, 0.25, N); sci = interp(pc, pos_all, a, 1.0, N)
        sit = jnp.where(sfi != 0, sfi, sci)
        _, eS = rf_energy(ps, bs, ss, rhos, sqp_, sq_, 0.25, EPSIN, EPSOUT)
        return eM, eM0, eS, sit.astype(jnp.float32)
    inner = jax.vmap(one, in_axes=(None, None, None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0))
    return jax.vmap(inner)(mc["bidx"], mc["surf"], mf["bidx"], mf["surf"], ms["bidx"], ms["surf"], acent, qp, q, sqp, sq,
                           phic, phif, phis)


# ------------------------------------------------------------------------------------------------ DelPhi spectral radius (M2)
# Port of DelPhi 5.1 relfac (delphi4py rundelphi .so, relfac_ disassembled 2026-11-19; reproduces the logged
# "gauss-seidel spectral radius" to float32 ulps on identical maps):
#   v(i,j,k) = s(i) s(j) s(k),  s(i) = sinf(i*pi/N) * sqrt(2) / sqrt(N),  N = igrid - 1 (s(0) = s(N) = 0), float32
#   w = J(J(v)) with J the point-Jacobi operator of the PB problem (phi_new = sum_6 eps_k phi_k / (sum eps + k^2h^2 eps_out ion),
#       zero on the faces), float32
#   rho = 2 * sum over the odd points ((i+j+k) odd) of v*w, accumulated SEQUENTIALLY in float32 in Fortran order
#   (x fastest); if rho > 1: rho = 0.995.
# The float32 sequential sum is part of the method (it lowers rho by ~1.2e-4 vs the exact dot product), so the products
# are formed on the device and summed on the host with numpy's sequential float32 cumsum.
_n = RC.GSIZE
_I = np.arange(_n)
_PAR = (_I[:, None, None] + _I[None, :, None] + _I[None, None, :]) % 2
_ODD_F = np.nonzero(_PAR.transpose(2, 1, 0).ravel() == 1)[0]                     # Fortran-order positions of odd points
_ODD_C = np.ravel_multi_index(np.unravel_index(_ODD_F, (_n, _n, _n), order="C")[::-1], (_n, _n, _n))   # -> C flat index


def _relfac_vec():
    f = np.float32
    N = f(_n - 1)
    s = (np.sin((_I.astype(f) * f(np.pi)) / N).astype(f) * f(np.sqrt(2.0)) / np.sqrt(N)).astype(f)
    s[0] = 0; s[-1] = 0
    return (s[:, None, None] * s[None, :, None] * s[None, None, :]).astype(f)


@jax.jit
def relfac_terms(mf, ms, v, odd):
    """products v * J(J(v)) at the odd points (Fortran order) for the focused and model maps of B sites -> [B,2,n_odd]."""
    def one(m, h):
        ex, ey, ez, dinv = _coefs(m, h)

        def J(p):
            nb = (ex * jnp.roll(p, -1, 0) + jnp.roll(ex, 1, 0) * jnp.roll(p, 1, 0)
                  + ey * jnp.roll(p, -1, 1) + jnp.roll(ey, 1, 1) * jnp.roll(p, 1, 1)
                  + ez * jnp.roll(p, -1, 2) + jnp.roll(ez, 1, 2) * jnp.roll(p, 1, 2))
            return nb * dinv
        return (v * J(J(v))).reshape(-1)[odd]
    f_ = jax.vmap(lambda m: one(m, 0.25))
    return jnp.stack([f_(mf), f_(ms)], 1)


def estimate_specrad(mf, ms):
    """DelPhi relfac spectral radius for the focused and model grids of a site batch -> host [B,2] float32."""
    P = np.asarray(relfac_terms(mf, ms, jax.device_put(_relfac_vec()), jax.device_put(_ODD_C.astype(np.int32))))
    rho = np.float32(2.0) * np.cumsum(P, axis=-1, dtype=np.float32)[..., -1]
    return np.where(rho > 1.0, np.float32(0.995), rho).astype(np.float32)


# ------------------------------------------------------------------------------------------------ host driver
def resolve_backend(backend: str) -> str:
    """SOR backend: "xla" (reference lane pool), "pallas" (Triton kernel), "auto" (pallas on a GPU with the Triton
    lowering, else xla)."""
    if backend == "auto":
        return "pallas" if SOR.pallas_available() else "xla"
    if backend not in ("xla", "pallas"):
        raise ValueError(backend)
    return backend


@dataclass
class PBConfig:
    pb_mode: str = "delphi"
    backend: str = "xla"
    B: int = RC.B
    specrad: str = "logged"           # "logged" (M1: record ref) | "estimate" (M2: DelPhi relfac port) | callable name
    sizes: MP.Sizes = field(default_factory=MP.Sizes)
    xl_sizes: MP.Sizes = field(default_factory=MP.Sizes.xl)


def _site_layout(rec: RC.StateRecord, sites: List[int], B: int, specfn: Callable):
    """Host arrays of one fixed site batch; padding has no atoms, probes or query rows."""
    T, QM, MA = RC.T_MAX, RC.QM, RC.MA
    nreal = len(sites); glp = sites
    ac = np.full((B, 3), RC.FAR); spos = np.full((B, MA, 3), RC.FAR); srad = np.zeros((B, MA))
    qp = np.zeros((B, T, QM, 3)); q = np.zeros((B, T, QM)); sqp = np.zeros((B, T, QM, 3)); sq = np.zeros((B, T, QM))
    spec = np.full((B, T, 4), 0.99); tl = np.full((B, T), -1, np.int64)
    pos64 = rec.pos.astype(np.float64)
    for b, s in enumerate(glp):
        ac[b] = rec.site_acent_M[s]
        mp, mr = rec.model_atoms(s)
        kk = mr > 0
        if kk.sum() > MA:
            raise OverflowError(f"model atoms {kk.sum()} > MA {MA}")
        spos[b, :kk.sum()] = mp[kk]; srad[b, :kk.sum()] = mr[kk]
        qp[b] = ac[b]; sqp[b] = ac[b]
        if b >= nreal:
            continue
        taus = rec.site_taus(s)
        if len(taus) > T:
            raise OverflowError(f"site with {len(taus)} tautomers > T_MAX {T}")
        idx = rec.site_atom_idx(s)
        for j, t in enumerate(taus):
            qt = rec.tau_charges(t); nz = np.nonzero(qt)[0]
            if len(nz) > QM:
                raise OverflowError(f"{len(nz)} charged atoms > QM {QM}")
            qp[b, j, :len(nz)] = pos64[idx[nz]]; q[b, j, :len(nz)] = qt[nz]
            mq = rec.tau_model_charges(t)
            if len(mq) > QM:
                raise OverflowError(f"{len(mq)} model atoms > QM {QM}")
            sqp[b, j, :len(mq)] = mp; sq[b, j, :len(mq)] = mq
            spec[b, j] = [RC.RELPAR_COARSE, specfn(t, 1), specfn(t, 2), -1.0]
            tl[b, j] = t
    return dict(ac=ac, spos=spos, srad=srad, qp=qp, q=q, sqp=sqp, sq=sq, spec=spec, active=np.arange(B) < nreal), tl


def site_probes(spos, srad, OUT):
    """model-compound probes for a batch (geometry_sites inside the x64 context). spos/srad host [B,MA,*] (FAR/0)."""
    sv = srad > 0
    sR = np.where(sv, srad + RC.RADPRB, 0.0)
    with GEO.x64():
        U = RC.dev(GEO.fib_sphere(RC.NVERT), np.float64)
        pts, cnt, mmax, ntr = GEO.geometry_sites(RC.dev(spos, np.float64), RC.dev(sR, np.float64), jax.device_put(sv), U,
                                                 OUT=OUT)
    return pts, np.asarray(cnt), np.asarray(mmax), np.asarray(ntr)


def solve_state(rec: RC.StateRecord, cfg: PBConfig = PBConfig(), specfn: Optional[Callable] = None, timing: Optional[dict] = None,
                c: Optional[int] = None, sites: Optional[List[int]] = None, geo_parent=None, geo_out: Optional[dict] = None) -> RC.PBOut:
    """Precision-explicit wrapper: the PB float32 path runs with x64 OFF whatever the process-wide flag (records.py)."""
    with GEO.x32():
        return _solve_state(rec, cfg, specfn, timing, c, sites, geo_parent, geo_out)


def _solve_state(rec: RC.StateRecord, cfg: PBConfig = PBConfig(), specfn: Optional[Callable] = None, timing: Optional[dict] = None,
                 c: Optional[int] = None, sites: Optional[List[int]] = None, geo_parent=None, geo_out: Optional[dict] = None) -> RC.PBOut:
    """All three DelPhi problems for every tautomer of `rec`. specfn(t, k) -> Gauss-Seidel spectral radius of tautomer t
    (MC order) for call k (1 focused, 2 model); default: the logged values (rec.ref['specrad']).
    SPEED lever 3: geo_parent = the AB state's geometry.GeoReuse (chain states of a triple; identical probes);
    geo_out: dict that receives 'reuse' (this state's GeoReuse, for AB states)."""
    tim = timing if timing is not None else {}
    mt = tim.pop("memtrack", None)                 # M6: optional GPU memory tracker (api.MemTrack), never numerics
    mark = mt.mark if mt is not None else (lambda stage: None)
    T0 = time.perf_counter()
    if specfn is None and cfg.specrad == "logged":
        sr = rec.ref["specrad"]
        specfn = lambda t, k: float(sr[k, t])
    elif specfn is None:
        specfn = lambda t, k: 0.99      # placeholder; replaced by the estimator after the maps (cfg.specrad == "estimate")
    for k, v in (("epsin", EPSIN), ("epsout", EPSOUT), ("conc", CONC)):
        assert abs(float(rec.params.get(k, v)) - v) < 1e-9, (k, rec.params.get(k))
    flags = {"xl_retries": 0, "class_bumps": 0}
    c = rec.size_class if c is None else c
    while True:
        if not 0 <= c < len(RC.NA_C):
            raise OverflowError(f"geometry exceeds last size class {len(RC.NA_C) - 1}")
        try:
            geo = GEO.geometry_state(rec.pos, rec.rad, c, parent=geo_parent, want_reuse=geo_out is not None)
            break
        except OverflowError as e:
            flags["class_bumps"] += 1; flags["class_bump_reason"] = str(e)
            c += 1
    flags.update({"geo_" + k: v for k, v in geo.flags.items() if isinstance(v, int)})
    flags["size_class"] = c
    if geo_out is not None:
        geo_out["reuse"] = geo.reuse
    tim["t_geometry"] = time.perf_counter() - T0
    mark("geometry")
    pos_m = jax.device_put(geo.pos); rad_m = jax.device_put(geo.rad)
    Nall = RC.NATOM_ALL_C[c]
    pa = np.full((Nall, 3), RC.FAR, np.float32); pa[:rec.n_atoms] = rec.pos
    pos_all = jax.device_put(pa)
    Tn = rec.n_taus
    out = dict(esolvM=np.full(Tn, np.nan, np.float32), esolvS=np.full(Tn, np.nan, np.float32), esolvM0=np.full(Tn, np.nan, np.float32),
               sitpotM=np.zeros((Tn, rec.n_atoms), np.float32), sweeps=np.zeros((Tn, 3), np.int32), maxchange=np.zeros((Tn, 3), np.float32),
               rho=np.zeros((Tn, 2), np.float32))
    B = cfg.B
    sl = list(range(rec.n_sites)) if sites is None else list(sites)
    nb = int(np.ceil(len(sl) / B))
    batches = [sl[i * B:(i + 1) * B] for i in range(nb)]
    tm = ts = tp = tsor = 0.0; sor_stats = []; maxctr = None
    backend = resolve_backend(cfg.backend)
    xla = backend == "xla"
    pool = SOR.run_pool if xla else SOR.run_pool_pallas
    for bl in batches:
        lay, tl = _site_layout(rec, bl, B, specfn)
        for attempt, sz in enumerate((cfg.sizes, cfg.xl_sizes)):
            t1 = time.perf_counter()
            sprb, scnt, smx, _ = site_probes(lay["spos"], lay["srad"], sz.PBs)
            ov_model = bool((scnt > sz.PBs).any() or (smx > GEO.MX).any())
            j = {k: RC.dev(v, bool if k == "active" else np.float32) for k, v in lay.items()}
            mc, mf, ms, ctr = MP.maps_batch(pos_m, rad_m, geo.probes, j["ac"], j["spos"], j["srad"], sprb, sz=sz, active=j["active"])
            ctr_h = np.asarray(ctr)
            ov = MP.overflow(ctr_h, sz)
            mark("maps")
            tm += time.perf_counter() - t1
            if not (ov_model or any(ov.values())):
                break
            if attempt == 1:
                raise OverflowError(f"XL sizes overflow: {ov} model {ov_model}")
            flags["xl_retries"] += 1
            flags.setdefault("xl_reasons", []).append({**ov, "model_probes": ov_model})
        del sprb
        del j["spos"], j["srad"]
        maxctr = ctr_h.max(0) if maxctr is None else np.maximum(maxctr, ctr_h.max(0))
        valid = tl >= 0
        spec_h = lay["spec"]
        if cfg.specrad == "estimate":
            rho_b = estimate_specrad(mf, ms)                       # [B,2] (focused, model)
            sp = lay["spec"].copy()
            sp[..., 1] = np.where(valid, rho_b[:, None, 0], sp[..., 1]); sp[..., 2] = np.where(valid, rho_b[:, None, 1], sp[..., 2])
            j["spec"] = RC.dev(sp, np.float32)
            spec_h = sp
        t2 = time.perf_counter()
        items1, cf1, kf1 = setup1(mc, ms, j["ac"], j["qp"], j["q"], j["sqp"], j["sq"], j["spec"], backend=backend)
        BT = B * RC.T_MAX
        if timing is not None:
            jax.block_until_ready(items1)
        t4 = time.perf_counter()
        op1 = np.concatenate([np.repeat(np.arange(B), RC.T_MAX), B + np.repeat(np.arange(B), RC.T_MAX)])
        st1, it1, mc1 = pool(items1[0], items1[1], items1[2], RC.dev(op1, np.int32),
                             np.concatenate([valid.reshape(-1), valid.reshape(-1)]), cf1 if xla else kf1, stats=sor_stats)
        tsor += time.perf_counter() - t4
        del items1, cf1, kf1
        items2, cf2, kf2 = setup2(mf, j["ac"], j["qp"], j["q"], j["spec"], st1, backend=backend)
        if timing is not None:
            jax.block_until_ready(items2)
        t4 = time.perf_counter()
        op2 = np.concatenate([np.repeat(np.arange(B), RC.T_MAX), np.zeros(BT, np.int64)])
        st2, it2, mc2 = pool(items2[0], items2[1], items2[2], RC.dev(op2, np.int32),
                             np.concatenate([valid.reshape(-1), np.zeros(BT, bool)]), cf2 if xla else kf2, stats=sor_stats)
        t3 = time.perf_counter(); ts += t3 - t2; tsor += t3 - t4
        del items2, cf2, kf2
        mark("sor")
        eM, eM0, eS, sit = (np.asarray(x) for x in post(mc, mf, ms, j["ac"], j["qp"], j["q"], j["sqp"], j["sq"], st1, st2, pos_all))
        tp += time.perf_counter() - t3
        mark("post")
        for b in range(B):
            for jj in range(RC.T_MAX):
                t = tl[b, jj]
                if t < 0:
                    continue
                k = b * RC.T_MAX + jj
                out["esolvM"][t] = eM[b, jj]; out["esolvM0"][t] = eM0[b, jj]; out["esolvS"][t] = eS[b, jj]
                out["sitpotM"][t] = sit[b, jj, :rec.n_atoms]
                out["sweeps"][t] = [it1[k], it2[k], it1[BT + k]]; out["maxchange"][t] = [mc1[k], mc2[k], mc1[BT + k]]
                out["rho"][t] = spec_h[b, jj, 1:3]
        del st1, st2, mc, mf, ms, j, sit, eM, eM0, eS
    lane = sum(s["lane_blocks"] for s in sor_stats); need = sum(s["needed"] for s in sor_stats)
    flags.update({"ctr_" + g + "_" + n_: int(maxctr[gi, ni]) for gi, g in enumerate("cfs") for ni, n_ in enumerate(MP.CTR_NAMES)})
    tim.update(t_maps=tm, t_solves=ts, t_sor=tsor, t_post=tp, t_total=time.perf_counter() - T0, nbatches=nb,
               sor_lane_blocks=lane, sor_needed_blocks=need, sor_waste=lane / max(need, 1))
    return RC.PBOut(esolvM=out["esolvM"], esolvS=out["esolvS"], esolvM0=out["esolvM0"], sitpotM=out["sitpotM"], sweeps=out["sweeps"],
                    maxchange=out["maxchange"], flags=flags, pb_mode=cfg.pb_mode, backend=backend, rho=out["rho"],
                    cfg_hash=RC.cfg_hash({"pb_mode": cfg.pb_mode, "backend": cfg.backend, "B": cfg.B, "specrad": cfg.specrad}))
