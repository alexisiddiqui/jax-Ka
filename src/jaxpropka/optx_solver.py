"""Optimistix root solve of the mean-field fixed point, with pH continuation.

Opt-in alternative to the damped fixed-iteration loop in ``model.py``. The
model equations, local terms and validity contract are unchanged:

    h = sigmoid(ln10 * (intrinsic - pH - Phi(h))),  Phi(h) = field0 + J h.

Unknowns are logits u (h = sigmoid(u)) on a static set of channels, and the
root residual is the logit-space fixed-point map

    G(u) = u - ln10 * (intrinsic - pH - Phi(sigmoid(u))),

whose Jacobian I + ln10 * J * diag(sigmoid'(u)) is identity plus coupling.

The default method is Levenberg-Marquardt (trust region on 1/2|G|^2). Plain
Newton has no globalization and was observed to cycle at a few pH points on
real complexes (64-step cap hit, residual ~3e-3); it remains available as
``method="newton"``. ``linear="dense"`` materializes the Jacobian on the active
set (cheap for native evaluation, ~100 channels); ``"iterative"`` is matrix-free
(GMRES for Newton, normal-equation CG for LM) for large soft-sequence sets. Each pH is warm-started from the
previous solution (``lax.scan`` over the grid) rather than solved cold under
``vmap``; only one pH state is live at a time.

Gradients use optimistix's implicit adjoint at the converged root: one linear
solve per pH, with no activation memory proportional to the iteration count.
Convergence is judged by the SAME full-state residual max|target(h) - h| and
tolerance as the default solver; optimistix's own termination flag is reported
separately and is not sufficient on its own.

Continuation follows one branch. With ``sweep="both"`` the grid is also solved
descending and the maximum |h_up - h_down| is reported: a nonzero gap is
direct evidence of multiple stable mean-field solutions (hysteresis), which
the damped solver cannot distinguish from slow convergence.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
from .model import SiteCurveResult, _local_terms, _field, _packed_field
from .parameters import GROUP_AA, Q_DEPROT


@dataclass(frozen=True)
class SolverConfig:
    method: str = "lm"             # "lm" (Levenberg-Marquardt) or "newton"
    max_steps: int = 128           # solver iterations per pH (static cap)
    rtol: float = 1e-6
    atol: float = 1e-6             # on logits; |dh| <= atol/4
    linear: str = "dense"          # "dense" (active-set Jacobian) or "iterative" (matrix-free)
    gmres_restart: int = 30
    gmres_max_steps: int = 300
    sweep: str = "up"              # "up" or "both" (adds descending sweep + hysteresis gap)
    coupling: str = "dense"        # "dense": active-set K extracted once; "operator": full field per call

    def __post_init__(self):
        if self.coupling not in ("dense", "operator"):
            raise ValueError("coupling must be 'dense' or 'operator'")
        if self.method not in ("lm", "newton"):
            raise ValueError("method must be 'lm' or 'newton'")
        if self.linear not in ("iterative", "dense"):
            raise ValueError("linear must be 'iterative' or 'dense'")
        if self.sweep not in ("up", "both"):
            raise ValueError("sweep must be 'up' or 'both'")
        for name in ("max_steps", "gmres_restart", "gmres_max_steps"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive Python int")


def _solver(scfg):
    tol = dict(rtol=scfg.rtol*1e-2, atol=scfg.atol*1e-2)
    if scfg.method == "lm":
        linear = (lx.QR() if scfg.linear == "dense"
                  else lx.Normal(lx.CG(**tol, max_steps=scfg.gmres_max_steps)))
        return optx.LevenbergMarquardt(rtol=scfg.rtol, atol=scfg.atol, linear_solver=linear)
    linear = (lx.AutoLinearSolver(well_posed=True) if scfg.linear == "dense"
              else lx.GMRES(**tol, restart=scfg.gmres_restart, max_steps=scfg.gmres_max_steps))
    return optx.Newton(rtol=scfg.rtol, atol=scfg.atol, linear_solver=linear)


def active_system(arrays, terms, active):
    """(intrinsic, field0, K) restricted to the active channels, with Phi_a = field0 + K h_a.

    K is gathered from terms.coupling [N,Kc,9,9] (row (i,g), column
    neighbors[i,k]*9+t); memory O(M*Kc*9). Exact for the active rows provided
    excluded channels carry zero weight (checked by the caller via the leak flag).
    """
    gm = arrays["group_mask"]; dtype = terms.intrinsic.dtype
    m = active.shape[0]
    rows_i, rows_g = active//9, active%9
    inverse = jnp.full(gm.size, -1, active.dtype).at[active].set(jnp.arange(m, dtype=active.dtype))
    cols = inverse[(arrays["neighbors"][rows_i][:, :, None]*9+jnp.arange(9)[None, None, :])]   # [M,Kc,9]
    vals = jnp.where(cols >= 0, terms.coupling[rows_i, :, rows_g, :], 0)                     # [M,Kc,9]
    rowid = jnp.broadcast_to(jnp.arange(m)[:, None, None], cols.shape)
    coupling = jnp.zeros((m, m), dtype).at[rowid, jnp.maximum(cols, 0)].add(vals)
    return terms.intrinsic.reshape(-1)[active], terms.field0.reshape(-1)[active], coupling


def free_energy(h, ph, intrinsic, field0, coupling, weights):
    """Mean-field free energy (units kT ln10) on the active set, up to a pH-dependent constant.

    F = sum_i w_i [ (h ln h + (1-h) ln(1-h))/ln10 + h (pH - pKint_i) + field0_i h ]
        + 1/2 sum_ij w_i K_ij h_i h_j.
    Its stationary points are the fixed points h = sigmoid(ln10 (pKint - pH - Phi)),
    because w_i K_ij = w_i w_j (C + Hd + Hr)_ij is symmetric. K is symmetrized
    here to remove float roundoff in the stored reciprocal kernels.
    """
    log10 = jnp.log(jnp.asarray(10, h.dtype))
    entropy = (jax.scipy.special.xlogy(h, h)+jax.scipy.special.xlogy(1-h, 1-h))/log10
    wk = weights[:, None]*coupling; wk = (wk+wk.T)/2
    return jnp.sum(weights*(entropy+h*(ph-intrinsic)+field0*h))+0.5*h@wk@h


def _sweep(arrays, terms, ph, field, active, scfg):
    """Warm-started root solves along ``ph`` (any order). Returns per-pH states."""
    gm = arrays["group_mask"]
    dtype = terms.intrinsic.dtype
    log10 = jnp.log(jnp.asarray(10, dtype=dtype))
    shape = gm.shape

    def full(u):
        return jnp.zeros(gm.size, dtype).at[active].set(jax.nn.sigmoid(u)).reshape(shape)

    def target(h, x):
        return jnp.where(gm, jax.nn.sigmoid(log10*(terms.intrinsic-x-field(terms, h))), 0)

    if scfg.coupling == "dense":
        # The field is linear in occupancy: Phi = field0 + K h, with K fixed for a
        # given sequence. Gather the active-set block of K directly from
        # terms.coupling [N,Kc,9,9] (row (i,g), column neighbors[i,k]*9+t);
        # memory O(M*Kc*9). Each solver step is then an [M,M] matvec/solve.
        # K, field0 and intrinsic enter via args, so the implicit adjoint
        # differentiates through them.
        base = active_system(arrays, terms, active)
        def residual_fn(u, args):
            (intrinsic, f0, k), x = args
            return u - log10*(intrinsic - x - f0 - k@jax.nn.sigmoid(u))
    else:
        base = terms
        def residual_fn(u, args):
            t, x = args
            drive = t.intrinsic - x - field(t, full(u))
            return u - log10*drive.reshape(-1)[active]

    solver = _solver(scfg)

    def step(u, x):
        sol = optx.root_find(residual_fn, solver, u, args=(base, x),
                             max_steps=scfg.max_steps, throw=False)
        u = sol.value
        # Channels outside the active set do not feed back (zero weight), so
        # their occupancy is the closed-form response to the active field.
        h = target(full(u), x)
        h = jnp.where(gm, h, 0)
        error = jnp.abs(target(h, x)-h)
        ok = sol.result == optx.RESULTS.successful
        return u, (h, jnp.max(error), jnp.max(error*terms.weights), ok, sol.stats["num_steps"])

    x0 = ph[0]
    u0 = (log10*(terms.intrinsic-x0-terms.field0)).reshape(-1)[active]
    _, out = jax.lax.scan(step, u0, ph)
    return out


def _result(arrays, probabilities, ph, terms, field, active, cfg, scfg):
    h, residual, wresidual, ok, steps = _sweep(arrays, terms, ph, field, active, scfg)
    extra = dict(optx_success=ok, newton_steps=steps)
    if scfg.sweep == "both":
        hd, rd, _, okd, stepsd = _sweep(arrays, terms, ph[::-1], field, active, scfg)
        hd, rd = hd[::-1], rd[::-1]
        extra.update(hysteresis=jnp.max(jnp.abs(h-hd), axis=0), down_residual=rd,
                     down_optx_success=okd[::-1], down_newton_steps=stepsd[::-1])
    # Active-set safety: an excluded channel with nonzero weight would feed back
    # into the field, so the closed-form fill would be wrong. Mark nonconverged.
    excluded = jnp.ones(arrays["group_mask"].size, bool).at[active].set(False)
    leak = jnp.max(jnp.where(excluded, terms.weights.reshape(-1), 0))
    extra["active_set_leak"] = leak
    converged = (residual < cfg.residual_tolerance) & (leak == 0)
    charge = terms.weights[None]*(jnp.asarray(Q_DEPROT, dtype=probabilities.dtype)[None, None]+h)
    residue_charge = charge.sum(-1)
    effective = jax.vmap(lambda x: terms.intrinsic-field(terms, x))(h)
    return SiteCurveResult(ph, h, charge, residue_charge, residue_charge.sum(-1),
                           terms.weights, terms.intrinsic, effective,
                           residual, wresidual, converged), extra


@partial(jax.jit, static_argnames=("config", "solver_config"))
def optx_curve_kernel(arrays, probabilities, ph, active, edges=None, *, config, solver_config):
    """Curves via warm-started optimistix Newton. ``active``: int32 flat channel indices."""
    terms = _local_terms(arrays, probabilities, config)
    if edges is None:
        field = lambda t, h: _field(arrays, t, h)
    else:
        field = lambda t, h: _packed_field(t, h, edges)
    return _result(arrays, probabilities, ph, terms, field, active, config, solver_config)


def active_channels(cache, probabilities=None):
    """Static flat channel indices for the solve.

    Default: every group channel (valid for any soft sequence). Passing a fixed
    sequence P restricts to channels with nonzero weight under that P — much
    smaller for native evaluation. Evaluating a different P on that set is
    detected (``active_set_leak`` > 0) and marked nonconverged.
    """
    gm = np.asarray(cache.group_mask)
    if probabilities is None:
        return np.flatnonzero(gm.reshape(-1)).astype(np.int32)
    p = np.array(probabilities, dtype=float)
    frozen = np.asarray(cache.frozen)
    p[frozen] = np.eye(20)[np.asarray(cache.native_index)[frozen]]  # as in _local_terms
    weights = np.concatenate((p[:, GROUP_AA], np.ones((len(p), 2))), axis=1)*gm
    return np.flatnonzero(weights.reshape(-1) > 0).astype(np.int32)


@partial(jax.jit, static_argnames=("config", "solver_config", "initialization",
                                   "seed_steps", "seed_dtype"))
def local_terms_curve_kernel(arrays, terms, ph, active, active_valid, *, config,
                             solver_config, initialization="production", gradient_mask=None,
                             seed_steps=None, seed_dtype=None):
    """Shared LocalTerms -> curves path, with optimistix's implicit adjoint.

    Production initialization is a detached damped solve at each pH. Dummy active
    channels have identity residuals. gradient_mask is supplied only in the
    backward evaluation after a separate forward validity audit: rejected pH
    points solve a parameter-independent identity system, so an invalid implicit
    Jacobian is never hidden by multiplying its loss by zero.

    The seeding solve is vmapped over the whole pH grid BEFORE the continuation
    scan, not run inside it. Run per-pH inside the scan it is config.steps * len(ph)
    sequential iterations on one small [N,9] state (~150k for a two-branch step at
    73 pH); vmapped it is config.steps iterations on [H,N,9], the same arithmetic
    with H times fewer loop trips, which is how the production readout does it
    (model._over_ph). The seed is stop_gradient'd either way, so it never enters
    the adjoint, and the converged root -- hence every number downstream -- is
    unchanged. ``seed_steps`` overrides config.steps for the seed only; None keeps
    the production count and leaves the branch selection bit-identical.
    """
    if solver_config.coupling != 'dense' or solver_config.linear != 'dense':
        raise ValueError('Shared training kernel currently requires dense active coupling')
    if initialization not in ('production', 'up', 'down'):
        raise ValueError('Unknown root initialization')
    ph = jnp.asarray(ph, dtype=terms.intrinsic.dtype)
    if gradient_mask is None:
        gradient_mask = jnp.ones(ph.shape, dtype=bool)
    gm = arrays['group_mask']; dtype = terms.intrinsic.dtype
    log10 = jnp.log(jnp.asarray(10., dtype)); valid = jnp.asarray(active_valid, bool)
    intrinsic, field0, coupling = active_system(arrays, terms, active)
    coupling = jnp.where(valid[:, None] & valid[None, :], coupling, 0)
    base = (jnp.where(valid, intrinsic, 0), jnp.where(valid, field0, 0), coupling)
    field = lambda h: _field(arrays, terms, h)

    def full(u):
        values = jnp.where(valid, jax.nn.sigmoid(u), 0)
        return jnp.zeros(gm.size, dtype).at[active].set(values).reshape(gm.shape)

    def target(h, x):
        return jnp.where(gm, jax.nn.sigmoid(log10 * (terms.intrinsic-x-field(h))), 0)

    def residual(u, args):
        (pk, f0, k), x = args
        return jnp.where(valid, u-log10*(pk-x-f0-k@jax.nn.sigmoid(u)), u)

    solver = _solver(solver_config)
    initial = jnp.where(valid, log10*(base[0]-ph[0]-base[1]), 0)

    reverse = initialization == 'down'
    xx = ph[::-1] if reverse else ph
    mm = gradient_mask[::-1] if reverse else gradient_mask
    if reverse:
        initial = jnp.where(valid, log10*(base[0]-xx[0]-base[1]), 0)

    if initialization == 'production':
        # Seed every pH at once: config.steps wide iterations rather than
        # config.steps * len(ph) narrow ones. Detached, so it never reaches the
        # adjoint and cannot change the converged root.
        steps = config.steps if seed_steps is None else seed_steps
        # The seed iterates the ACTIVE-SET system, not the full [N,Kc,9,9] field. The
        # two agree exactly on the active channels whenever excluded channels carry
        # zero weight -- the premise of active_system, enforced by the leak check
        # below -- and inactive channels are filled closed-form from the root anyway.
        # Per iteration that is an [M,M] matvec instead of an N*Kc*81 contraction:
        # ~730x less work at N=256/M=64, ~100x at N=1472/M=480. The seed can then
        # afford the full production step count on complexes where a short Picard
        # run does not reach the solver's basin of attraction.
        if seed_dtype is None:
            seed_base, seed_log10 = base, log10
        else:
            seed_base = jax.tree.map(lambda a: a.astype(seed_dtype), base)
            seed_log10 = log10.astype(seed_dtype)
        pk_s, f0_s, k_s = seed_base
        valid_s = valid

        # The logit transform needs a bound the WORKING dtype can represent either side
        # of: 1-1e-14 rounds to exactly 1.0 in float32, so a saturated occupancy would
        # give log1p(-1) = -inf and the LM linear solve would return NaN. float64 keeps
        # its original 1e-14 because that is already far above its eps.
        bound = max(1e-14, float(np.finfo(dtype).eps))

        def seed_one(x):
            xs = x.astype(pk_s.dtype)
            target = lambda h: jnp.where(
                valid_s, jax.nn.sigmoid(seed_log10*(pk_s-xs-f0_s-k_s@h)), 0)
            h = target(jnp.zeros_like(pk_s))
            h = jax.lax.fori_loop(0, steps,
                                  lambda _, old: old+config.damping*(target(old)-old), h)
            h = jnp.clip(h.astype(dtype), bound, 1-bound)
            return jnp.where(valid, jnp.log(h)-jnp.log1p(-h), 0)

        seeds = jax.lax.stop_gradient(jax.vmap(seed_one)(xx))
    else:
        seeds = jnp.zeros((xx.shape[0], active.shape[0]), dtype)

    def step(previous, inputs):
        x, accepted, seed = inputs
        u0 = seed if initialization == 'production' else jax.lax.stop_gradient(previous)
        # All rejected systems are well-conditioned, independent of model params.
        args = (jax.tree.map(lambda a: jnp.where(accepted, a, 0), base),
                jnp.where(accepted, x, 0))
        u0 = jnp.where(accepted, u0, 0)
        sol = optx.root_find(residual, solver, u0, args=args,
                            max_steps=solver_config.max_steps, throw=False)
        occupied = target(full(sol.value), x)
        error = jnp.abs(target(occupied, x)-occupied)
        finite = jnp.all(jnp.isfinite(occupied)) & jnp.all(jnp.isfinite(sol.value))
        return sol.value, (occupied, jnp.max(error), jnp.max(error*terms.weights),
                           sol.result == optx.RESULTS.successful, sol.stats['num_steps'], finite)

    _, outputs = jax.lax.scan(step, initial, (xx, mm, seeds))
    if reverse:
        outputs = jax.tree.map(lambda a: a[::-1], outputs)
    h, residuals, wresiduals, success, steps, finite = outputs
    included = jnp.zeros(gm.size, bool).at[active].set(valid)
    leak = jnp.max(jnp.where(included, 0, terms.weights.reshape(-1)))
    converged = finite & (residuals < config.residual_tolerance) & (leak == 0) & gradient_mask
    charge = terms.weights[None]*(jnp.asarray(Q_DEPROT, dtype)[None, None]+h)
    residue_charge = charge.sum(-1)
    effective = jax.vmap(lambda state: terms.intrinsic-field(state))(h)
    result = SiteCurveResult(ph, h, charge, residue_charge, residue_charge.sum(-1),
        terms.weights, terms.intrinsic, effective, residuals, wresiduals, converged)
    return result, dict(optx_success=success, newton_steps=steps, active_set_leak=leak,
                        finite=finite, gradient_mask=gradient_mask)
