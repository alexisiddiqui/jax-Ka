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

    def __post_init__(self):
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
                  else lx.NormalCG(**tol, max_steps=scfg.gmres_max_steps))
        return optx.LevenbergMarquardt(rtol=scfg.rtol, atol=scfg.atol, linear_solver=linear)
    linear = (lx.AutoLinearSolver(well_posed=True) if scfg.linear == "dense"
              else lx.GMRES(**tol, restart=scfg.gmres_restart, max_steps=scfg.gmres_max_steps))
    return optx.Newton(rtol=scfg.rtol, atol=scfg.atol, linear_solver=linear)


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

    def residual_fn(u, args):
        t, x = args
        drive = t.intrinsic - x - field(t, full(u))
        return u - log10*drive.reshape(-1)[active]

    solver = _solver(scfg)

    def step(u, x):
        sol = optx.root_find(residual_fn, solver, u, args=(terms, x),
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
