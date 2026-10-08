"""Execution-only memory options must not change any readout."""
import gc
import jax.numpy as jnp
import numpy as np
import pytest
from jaxpropka import TitrationModel, ModelConfig
from jaxpropka.synthetic import synthetic_cache

PH = np.linspace(-2., 16., 13)


@pytest.fixture
def coupled():
    return synthetic_cache(n=10, neighbors=6, chains=2)


@pytest.mark.parametrize("backend", ["dense", "packed", "packed_v2"])
@pytest.mark.parametrize("ph_batch", [1, 4, 13, 64])
def test_ph_batch_matches_all_at_once(coupled, backend, ph_batch):
    cfg = ModelConfig(steps=128)
    ref_model = TitrationModel(coupled, cfg, backend=backend)
    model = TitrationModel(coupled, cfg, backend=backend, ph_batch=ph_batch)
    p = ref_model.native_probabilities.astype(jnp.float64)
    a, b = ref_model.curves(PH)(p), model.curves(PH)(p)
    for name in ("protonated", "site_charge", "total_charge", "effective_pka", "residual"):
        np.testing.assert_allclose(getattr(b, name), getattr(a, name), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(a.converged, b.converged)
    ga, gb = ref_model.pka_from_grid(PH)(p), model.pka_from_grid(PH)(p)
    np.testing.assert_allclose(gb.value, ga.value, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(ga.valid, gb.valid)


def test_ph_batch_validation(coupled):
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError):
            TitrationModel(coupled, ph_batch=bad)


@pytest.mark.parametrize("backend", ["dense", "packed", "packed_v2"])
def test_release_host_arrays_keeps_readouts(backend):
    cache = synthetic_cache(n=10, neighbors=6, chains=2)
    fingerprint = cache.fingerprint()
    model = TitrationModel(cache, ModelConfig(steps=64), backend=backend, ph_batch=5)
    p = model.native_probabilities.astype(jnp.float64)
    before = model.curves(PH, residues=[cache.keys[3]])(p)
    model.release_host_arrays(); del cache; gc.collect()
    # Zero strides: a broadcast view with no per-element storage (.nbytes reports
    # the logical size even for views, so it cannot show the release).
    for name in ("volume", "mass", "hbond", "coulomb_geometry", "pair_mask"):
        assert all(x == 0 for x in getattr(model.cache, name).strides)
    assert model.cache.validate() is not None
    after = model.curves(PH, residues=[model.cache.keys[3]])(p)
    np.testing.assert_allclose(after.residue_charge, before.residue_charge, atol=0)
    np.testing.assert_allclose(after.chain_charge, before.chain_charge, atol=0)
    assert model.cache.fingerprint() != fingerprint  # documented: fingerprint first
