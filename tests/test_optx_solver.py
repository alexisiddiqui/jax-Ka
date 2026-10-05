from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

optx = pytest.importorskip("optimistix")

from jaxpropka import TitrationModel, ModelConfig, one_hot
from jaxpropka.model import curve_kernel, pack_interaction_edges
from jaxpropka.optx_solver import SolverConfig, active_channels, optx_curve_kernel
from jaxpropka.parameters import MODEL_PKA
from jaxpropka.synthetic import synthetic_cache

PH = np.linspace(-2., 16., 37)


def arrays_of(cache):
    return TitrationModel(cache)._d


def run(cache, p, config, scfg=SolverConfig(), active=None, packed=False, ph=PH):
    d = arrays_of(cache)
    act = jnp.asarray(active_channels(cache) if active is None else active)
    edges = (tuple(jnp.asarray(x) for x in pack_interaction_edges(cache.pair_mask, cache.neighbors))
             if packed else None)
    return optx_curve_kernel(d, p, jnp.asarray(ph, p.dtype), act, edges,
                             config=config, solver_config=scfg)


@pytest.fixture
def coupled():
    return synthetic_cache(n=10, neighbors=6, chains=2)


def native(cache):
    return jnp.asarray(one_hot(cache.native_index, np.float64))


def test_uncoupled_matches_henderson_hasselbalch(cache):
    cfg = ModelConfig(coulomb_scale=0, hbond_scale=0, desolv_scale=0)
    out, extra = run(cache, native(cache), cfg)
    expected = 1/(1+10**(PH[:, None, None]-MODEL_PKA[None, None, :]))*cache.group_mask
    np.testing.assert_allclose(out.protonated, expected, atol=1e-9)
    assert np.all(out.converged) and np.all(extra["optx_success"])


@pytest.mark.parametrize("method,linear", [("lm", "dense"), ("lm", "iterative"),
                                           ("newton", "dense"), ("newton", "iterative")])
def test_coupled_matches_converged_damped_solver(coupled, method, linear):
    cfg = ModelConfig(steps=4096)
    p = native(coupled)
    ref = curve_kernel(arrays_of(coupled), p, jnp.asarray(PH), config=cfg)
    assert np.all(ref.converged)
    out, extra = run(coupled, p, cfg, SolverConfig(method=method, linear=linear))
    assert np.all(out.converged)
    np.testing.assert_allclose(out.protonated, ref.protonated, atol=2e-6)
    # Warm-started continuation needs only a few solver steps per pH.
    assert int(np.max(extra["newton_steps"][1:])) <= 12


def test_native_active_set_matches_full_set_and_flags_leak(coupled):
    cfg = ModelConfig()
    p = native(coupled)
    full, _ = run(coupled, p, cfg)
    small = active_channels(coupled, p)
    assert len(small) < len(active_channels(coupled))
    reduced, extra = run(coupled, p, cfg, active=small)
    np.testing.assert_allclose(reduced.protonated, full.protonated, atol=1e-7)
    assert float(extra["active_set_leak"]) == 0 and np.all(reduced.converged)
    soft = jnp.full_like(p, 1/20)
    _, leak = run(coupled, soft, cfg, active=small)
    out, _ = run(coupled, soft, cfg, active=small)
    assert float(leak["active_set_leak"]) > 0 and not np.any(out.converged)


def test_packed_field_matches_dense(coupled):
    cfg = ModelConfig(); p = native(coupled)
    a, _ = run(coupled, p, cfg)
    b, _ = run(coupled, p, cfg, packed=True)
    np.testing.assert_allclose(a.protonated, b.protonated, atol=1e-9)


def test_implicit_gradient_matches_finite_difference_and_unrolled(coupled):
    cfg = ModelConfig(steps=4096)
    d = arrays_of(coupled); act = jnp.asarray(active_channels(coupled))
    ph = jnp.asarray([4., 7., 10.])
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(coupled.n_residues, 20)))

    def charge_optx(z):
        out, _ = optx_curve_kernel(d, jax.nn.softmax(z, -1), ph, act, None,
                                   config=cfg, solver_config=SolverConfig())
        return jnp.sum(out.total_charge*jnp.asarray([1., -2., .5]))

    def charge_damped(z):
        out = curve_kernel(d, jax.nn.softmax(z, -1), ph, config=cfg)
        return jnp.sum(out.total_charge*jnp.asarray([1., -2., .5]))

    g = jax.grad(charge_optx)(logits)
    np.testing.assert_allclose(g, jax.grad(charge_damped)(logits), atol=1e-6)
    direction = jnp.asarray(rng.normal(size=logits.shape))
    eps = 1e-5
    fd = (charge_optx(logits+eps*direction)-charge_optx(logits-eps*direction))/(2*eps)
    np.testing.assert_allclose(jnp.vdot(g, direction), fd, rtol=1e-5, atol=1e-8)


def test_both_sweeps_agree_without_bistability(coupled):
    out, extra = run(coupled, native(coupled), ModelConfig(), SolverConfig(sweep="both"))
    assert np.all(out.converged) and np.all(extra["down_residual"] < 2e-5)
    assert float(np.max(extra["hysteresis"])) < 1e-6


def test_strong_coupling_reports_hysteresis_or_nonconvergence(coupled):
    # Large H-bond-free Coulomb scale can create multiple mean-field minima;
    # the solver must either converge on both sweeps (and expose any gap) or
    # flag nonconvergence, never silently return an unconverged state as valid.
    cfg = ModelConfig(coulomb_scale=12.)
    out, extra = run(coupled, native(coupled), cfg, SolverConfig(sweep="both"))
    ok = np.asarray(out.converged)
    assert np.all(np.asarray(out.residual)[ok] < 2e-5)
    assert np.isfinite(np.asarray(extra["hysteresis"])).all()


@pytest.mark.integration
@pytest.mark.parametrize("name", ["peptide.pdb", "two_chains.pdb"])
def test_real_structure_matches_damped_solver(name):
    pytest.importorskip("biotite")
    from pathlib import Path
    from jaxpropka.topology import load_topology
    from jaxpropka.geometry import build_candidates
    from jaxpropka.precompute import build_cache, native_identities
    top = load_topology(Path(__file__).parent/"data"/name)
    cache = build_cache(top, build_candidates(top), dtype=np.float64,
                        identities=native_identities(top))
    cfg = ModelConfig(steps=4096); p = native(cache)
    ref = curve_kernel(arrays_of(cache), p, jnp.asarray(PH), config=cfg)
    out, extra = run(cache, p, cfg, active=active_channels(cache, p), packed=True)
    assert np.all(out.converged) and np.all(extra["optx_success"])
    np.testing.assert_allclose(out.protonated, ref.protonated, atol=2e-6)
