"""jaxpb assembly (energy track, M3): PyPKA 2.10's MC energies from per-tautomer PB outputs, in JAX.

Port of (pypka env on Coulson, site-packages/pypka):
  tautomer.py calcBackEnergy   e_back(t) = sum over atoms NOT in t's site of p_chrgv4[a] * sitpotM_t[a] (cutoff -1 = all
                               atoms; linear PB only: PyPKA sets cutoff 5 nm only for nonlinear runs)
  tautomer.py calcpKint        dX = (X(ref) - X(t))/ln10 for esolvM, esolvS, e_back; pKint = pKmod + s(dM - dS + dB),
                               s = +1 anionic / -1 cationic; dg = pKint ln10 kT chargediff (chargediff -1 a / +1 c)
  run_mc.py init_mc_vars       possible_states_g = dg/kT (ref state last, g 0); occ: a -> tautomers 1, ref 0;
                               c -> tautomers 0, ref 1
  tautomer.py calcInteractionWith + titsite.py calc_interaction_between + main.py:519-537
                               S[t1,t2] = sum over site(t1) atoms of (p_chrgv4 - q_t1)(phi_ref(site t2) - phi_t2);
                               gg = sign1 sign2 |S kT| / kT (sign = getRefProtState: +1 c, -1 a), computed in both directions
                               and averaged: W[t1,t2] = W[t2,t1] = sign1 sign2 (|S[t1,t2]| + |S[t2,t1]|)/2
  mc.pyx select_pairs          site pair (i<j) coupled if any |W| >= ln10 couple_min, float32 (couple_min 2.0)
Same-site W blocks are 0 here (PyPKA stores -999999 there; mc.pyx never reads them).

Layout: one state per call, padded to a records.py size class c: all atoms Na = NATOM_ALL_C[c], microstates Mc = Na/8,
sites Nc = Mc/2, pairs Pc = Mc/4, site atoms A = records.QM. A state that overflows any of them is promoted to the next
class (result['promoted']); a state that fits no class raises (never silent).
W is ONE f32 GEMM dQ[Mc,Na] x dPhi[Mc,Na]^T at precision HIGHEST (no TF32), e_back one masked GEMV.
Precision: e_back / pKint / g accumulate in float64 by default (acc='f64', needs jax_enable_x64, the pipeline's
process-wide setting): an f32 e_back sum over ~9k atoms is off by up to 1.7e-5 kT on the pilot states (M3 test), the f64
GEMV is exact to 1e-13 vs PyPKA. W stays f32 (sums over <= 14 site atoms; 1e-6 kT). g and W are returned as f32.
"""
from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from jaxpb import records as RC

LOG10 = RC.LOG10                    # pypka.constants.LOG10
LN10F = np.float32(2.302585092994)  # mc.pyx cdef float LN10
COUPLE_MIN = 2.0
A_SITE = RC.QM
N_CLASSES = len(RC.NA_C)
HI = lax.Precision.HIGHEST


def class_dims(c):
    """(Na, Mc, Nc, Pc, A) of size class c (records.py tables)."""
    Na = RC.NATOM_ALL_C[c]
    Mc = Na // 8
    return Na, Mc, Mc // 2, Mc // 4, A_SITE


def couple_threshold(couple_min=COUPLE_MIN):
    return np.float32(LN10F * np.float32(couple_min))


class AssemblyInput(NamedTuple):
    """One state; numpy. Microstates in PyPKA MC order: site i (all_sites order), state j = tautomer number, reference
    tautomer last; M = sum(nst)."""
    q_bg: np.ndarray       # [N] f32  delphimol.p_chrgv4 (background charges, sites in their reference charge set)
    phi: np.ndarray        # [M, N] f32  sitpotM of each microstate's tautomer (reference rows included)
    esolvM: np.ndarray     # [M]  corrected RF energy, whole molecule (focused)
    esolvS: np.ndarray     # [M]  corrected RF energy, model compound
    site_atoms: np.ndarray  # [n, A] int32 atom positions of each site (pad -1)
    q_taut: np.ndarray     # [M, A] f32 charges of the microstate's tautomer on its site atoms (aligned with site_atoms)
    anionic: np.ndarray    # [n] bool  site type 'a' (True) / 'c' (False)
    pkmod: np.ndarray      # [M] f32  pKmod of the tautomer (.st file first line); reference rows ignored
    nst: np.ndarray        # [n] int32 states per site (tautomers + reference)


# ------------------------------------------------------------------ jitted kernels (module level, argument-pure)
@partial(jax.jit, static_argnames=('Nc', 'Pc'))
def select_pairs(W, site_of, thr, *, Nc, Pc):
    """mc.pyx select_pairs on a padded W [Mc,Mc] f32. site_of [Mc] i32 (padding rows = Nc). Returns pairs [Pc,2] i32
    (row-major i<j = mc.pyx order; padding -1) and the true pair count (> Pc = overflow)."""
    A = jnp.abs(W.astype(jnp.float32))
    m1 = jax.ops.segment_max(A, site_of, num_segments=Nc + 1)            # [Nc+1, Mc]
    m2 = jax.ops.segment_max(m1.T, site_of, num_segments=Nc + 1)[:Nc, :Nc]  # [Nc, Nc] (symmetric)
    iu = jnp.arange(Nc)
    mask = (m2.T >= thr) & (iu[:, None] < iu[None, :])
    i, j = jnp.nonzero(mask, size=Pc, fill_value=-1)
    return jnp.stack([i, j], 1).astype(jnp.int32), jnp.sum(mask).astype(jnp.int32)


@partial(jax.jit, static_argnames=('Nc', 'Pc', 'acc'))
def assemble_core(q_bg, phi, esolvM, esolvS, sa, sa_mask, q_taut, site_of, ref_row, anionic_m, pkmod, valid, is_ref,
                  thr, *, Nc, Pc, acc='f64'):
    """Padded state of one size class. Shapes: q_bg [Na]; phi [Mc,Na] f32; esolvM/esolvS/pkmod [Mc]; sa/sa_mask/q_taut
    [Mc,A] (site atoms of the microstate's site, its tautomer's charges); site_of/ref_row [Mc] i32 (padding: Nc / own row);
    anionic_m/valid/is_ref [Mc] bool. acc: 'f32' or 'f64' (needs jax_enable_x64) for e_back / pKint / g.
    Returns dict g, occ, W, e_back, pkint, pairs, npairs."""
    fd = jnp.float64 if acc == 'f64' else jnp.float32
    Mc, Na = phi.shape
    rows = jnp.broadcast_to(jnp.arange(Mc)[:, None], sa.shape)
    idx = jnp.where(sa_mask, sa, 0)
    # background energy: all atoms except the site's own (calcBackEnergy)
    notsite = jnp.ones((Mc, Na), jnp.float32).at[rows, idx].multiply(jnp.where(sa_mask, 0.0, 1.0).astype(jnp.float32))
    e_back = jnp.einsum('ma,a->m', (phi * notsite).astype(fd), q_bg.astype(fd), precision=HI)
    # pKint / g (calcpKint, init_mc_vars)
    r = ref_row
    dM = (esolvM[r].astype(fd) - esolvM.astype(fd)) / LOG10
    dS = (esolvS[r].astype(fd) - esolvS.astype(fd)) / LOG10
    dB = (e_back[r] - e_back) / LOG10
    s = jnp.where(anionic_m, 1.0, -1.0).astype(fd)
    pkint = pkmod.astype(fd) + s * (dM - dS + dB)
    taut = valid & ~is_ref
    g = jnp.where(taut, pkint * LOG10 * (-s), 0.0).astype(jnp.float32)
    occ = jnp.where(valid, jnp.where(anionic_m, taut, is_ref), False).astype(jnp.int8)
    # interactions (calcInteractionWith, both directions, averaged)
    dq = jnp.where(sa_mask, q_bg[idx] - q_taut, 0.0).astype(jnp.float32)
    dQ = jnp.zeros((Mc, Na), jnp.float32).at[rows, idx].add(dq)
    dPhi = phi[r] - phi
    S = jnp.dot(dQ, dPhi.T, precision=HI)                                # S[t1,t2]: site(t1) atoms, potentials of t2
    sg = jnp.where(anionic_m, -1.0, 1.0).astype(jnp.float32)
    W = sg[:, None] * sg[None, :] * ((jnp.abs(S) + jnp.abs(S.T)) / 2)
    keep = valid[:, None] & valid[None, :] & (site_of[:, None] != site_of[None, :])
    W = jnp.where(keep, W, 0.0).astype(jnp.float32)
    pairs, npairs = select_pairs(W, site_of, thr, Nc=Nc, Pc=Pc)
    return dict(g=g, occ=occ, W=W, e_back=e_back, pkint=jnp.where(taut, pkint, 0.0), pairs=pairs, npairs=npairs)


# ------------------------------------------------------------------ host side
def state_dims(inp):
    n = len(inp.nst)
    A = int((np.asarray(inp.site_atoms) >= 0).sum(1).max()) if n else 0
    return len(inp.q_bg), int(np.sum(inp.nst)), n, A


def fits(c, N, M, n, A, P=0):
    Na, Mc, Nc, Pc, Ac = class_dims(c)
    return N <= Na and M <= Mc and n <= Nc and A <= Ac and P <= Pc


def first_class(N, M, n, A, P=0, min_class=0):
    for c in range(min_class, N_CLASSES):
        if fits(c, N, M, n, A, P):
            return c
    return None


def pad_state(inp, c):
    """AssemblyInput -> padded argument dict for assemble_core at class c."""
    Na, Mc, Nc, Pc, A = class_dims(c)
    N, M, n, Areal = state_dims(inp)
    assert fits(c, N, M, n, Areal), ('does not fit class', c, N, M, n, Areal)
    nst = np.asarray(inp.nst, np.int64); start = np.concatenate([[0], np.cumsum(nst)[:-1]])
    site_of_r = np.repeat(np.arange(n), nst)
    site_atoms = np.asarray(inp.site_atoms)
    sa = np.zeros((Mc, A), np.int32); msk = np.zeros((Mc, A), bool); qt = np.zeros((Mc, A), np.float32)
    w = min(site_atoms.shape[1], A)
    assert (site_atoms[:, w:] < 0).all()
    sa[:M, :w] = np.maximum(site_atoms[site_of_r], 0); msk[:M, :w] = site_atoms[site_of_r] >= 0
    qt[:M, :w] = np.where(msk[:M, :w], np.asarray(inp.q_taut, np.float32)[:, :w], 0)
    phi = np.zeros((Mc, Na), np.float32); phi[:M, :N] = inp.phi
    q = np.zeros(Na, np.float32); q[:N] = inp.q_bg
    site_of = np.full(Mc, Nc, np.int32); site_of[:M] = site_of_r
    ref_row = np.arange(Mc, dtype=np.int32); ref_row[:M] = (start + nst - 1)[site_of_r]
    is_ref = np.zeros(Mc, bool); is_ref[:M] = np.arange(M) == ref_row[:M]
    an = np.zeros(Mc, bool); an[:M] = np.asarray(inp.anionic, bool)[site_of_r]
    valid = np.zeros(Mc, bool); valid[:M] = True

    def vec(x, dt=np.float32):
        o = np.zeros(Mc, dt); o[:M] = x; return o
    return dict(q_bg=q, phi=phi, esolvM=vec(inp.esolvM, np.asarray(inp.esolvM).dtype),
                esolvS=vec(inp.esolvS, np.asarray(inp.esolvS).dtype), sa=sa, sa_mask=msk, q_taut=qt,
                site_of=site_of, ref_row=ref_row, anionic_m=an, pkmod=vec(inp.pkmod, np.asarray(inp.pkmod).dtype), valid=valid, is_ref=is_ref,
                thr=couple_threshold()), dict(Nc=Nc, Pc=Pc), (N, M, n, start, nst)


def assemble(inputs, acc='f64', min_classes=None):
    """List of AssemblyInput -> list of dicts: g [M] f32, occ [M] i8, W [M,M] f32, start/nst [n], pairs [P,2] i32,
    e_back/pkint [M], size_class, promoted. Overflow of any class limit (incl. pairs > Pc) re-runs at the next class.
    min_classes: optional per-state lower bound (the StateRecord's size_class). acc='f64' requires jax_enable_x64
    (raises otherwise, never a silent f32 fallback)."""
    if acc == 'f64' and not jax.config.jax_enable_x64:
        raise RuntimeError("assemble(acc='f64') needs jax.config.update('jax_enable_x64', True)")
    out = []
    for k, inp in enumerate(inputs):
        N, M, n, A = state_dims(inp)
        lo = 0 if min_classes is None else int(min_classes[k])
        c0 = first_class(N, M, n, A, min_class=lo)
        if c0 is None:
            raise ValueError(f'state fits no size class: N={N} M={M} n={n} A={A}')
        c = c0
        while True:
            args, static, (N, M, n, start, nst) = pad_state(inp, c)
            r = assemble_core(**args, **static, acc=acc)
            npairs = int(r['npairs'])
            if npairs <= static['Pc']:
                break
            c += 1
            if c >= N_CLASSES:
                raise ValueError(f'pair overflow in every class: {npairs}')
        r = {k: np.asarray(v) for k, v in r.items()}
        out.append(dict(g=r['g'][:M], occ=r['occ'][:M], W=r['W'][:M, :M], start=start.astype(np.int32),
                        nst=nst.astype(np.int32), pairs=r['pairs'][:npairs], e_back=r['e_back'][:M],
                        pkint=r['pkint'][:M], size_class=c, npairs=npairs,
                        promoted=c != next(j for j in range(lo, N_CLASSES) if N <= class_dims(j)[0])))
    return out


# ------------------------------------------------------------------ records.py interface
def inputs_from_record(rec, pb):
    """records.StateRecord + records.PBOut (or any object with esolvM/esolvS/sitpotM in MC order) -> AssemblyInput.
    Raises on configurations the port does not cover (nonlinear-PB cutoff, missing pKmod)."""
    cut = rec.params.get('cutoff_back', -1)
    if not (cut == -1 or (isinstance(cut, float) and math.isnan(cut))):
        raise ValueError(f'calcBackEnergy cutoff {cut} nm not supported (linear PB, cutoff -1 only)')
    nst = np.diff(rec.site_tau_off).astype(np.int32)
    n = len(nst); M = int(nst.sum())
    so, sa = rec.site_atoms_off, rec.site_atoms
    width = int(np.diff(so).max()) if n else 0
    site_atoms = np.full((n, max(width, 1)), -1, np.int32)
    for i in range(n):
        site_atoms[i, :so[i + 1] - so[i]] = sa[so[i]:so[i + 1]]
    q_taut = np.zeros((M, site_atoms.shape[1]), np.float32)
    for t in range(M):
        q = rec.tau_q[rec.tau_q_off[t]:rec.tau_q_off[t + 1]]
        q_taut[t, :len(q)] = q
    pkmod = np.where(rec.tau_isref, 0.0, rec.tau_pkmod).astype(np.float64)
    if np.isnan(pkmod).any():
        raise ValueError('tau_pkmod missing (nan) for a non-reference tautomer')
    return AssemblyInput(q_bg=np.asarray(rec.q_bg, np.float32), phi=np.asarray(pb.sitpotM, np.float32),
                         esolvM=np.asarray(pb.esolvM), esolvS=np.asarray(pb.esolvS), site_atoms=site_atoms,
                         q_taut=q_taut, anionic=np.asarray(rec.site_type) == 'a', pkmod=pkmod, nst=nst)


def energies(recs, pbs, acc='f64'):
    """StateRecords + PBOuts -> list of records.Energies (diag: pkint, e_back, size_class, promoted, npairs,
    occ_matches_record)."""
    inputs = [inputs_from_record(r, p) for r, p in zip(recs, pbs)]
    outs = assemble(inputs, acc=acc, min_classes=[r.size_class for r in recs])
    res = []
    for r, o in zip(recs, outs):
        res.append(RC.Energies(g=o['g'], occ=o['occ'], W=o['W'], start=o['start'], nst=o['nst'], pairs=o['pairs'],
                               diag=dict(pkint=o['pkint'], e_back=o['e_back'], size_class=o['size_class'],
                                         promoted=o['promoted'], npairs=o['npairs'],
                                         occ_matches_record=bool(np.array_equal(o['occ'], np.asarray(r.tau_occ))))))
    return res
