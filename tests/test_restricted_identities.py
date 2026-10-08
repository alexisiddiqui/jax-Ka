"""Identity-restricted structural caches must be exact at allowed sequences."""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxpropka import TitrationModel, ModelConfig, one_hot
from jaxpropka.topology import load_topology
from jaxpropka.geometry import build_candidates
from jaxpropka.precompute import build_cache, native_identities

pytestmark = pytest.mark.integration
DATA = Path(__file__).parent/'data'


@pytest.fixture(autouse=True)
def real_biotite():
    return pytest.importorskip('biotite')


@pytest.fixture(params=['peptide.pdb', 'two_chains.pdb'])
def topology(request):
    return load_topology(DATA/request.param)


def test_native_cache_matches_full_cache_at_native_sequence(topology):
    lib = build_candidates(topology)
    full = build_cache(topology, lib, dtype=np.float64)
    native = build_cache(topology, lib, dtype=np.float64, identities=native_identities(topology))
    assert 'allowed_identities' not in full.metadata and full.identity_columns is None
    assert native.volume.shape[-1] == 1                       # compact: one stored identity
    assert native.nbytes < full.nbytes
    cols = native_identities(topology)
    for name in ('volume', 'mass', 'hbond'):
        a, b = getattr(full, name), native.expanded_env(name)
        allowed_cols = cols[native.env_neighbors]           # [N,Ke,20]
        np.testing.assert_array_equal(np.where(allowed_cols[:, :, None, :], a, 0), b)
    for name in ('local_hbond', 'bb_volume', 'bb_mass', 'bb_hbond', 'reorganization',
                 'pair_mask', 'coulomb_geometry', 'hb_donor', 'hb_reverse', 'neighbors', 'env_neighbors'):
        np.testing.assert_array_equal(getattr(full, name), getattr(native, name))
    cfg = ModelConfig(steps=256)
    p = jnp.asarray(one_hot(topology.native_index, np.float64))
    ph = np.linspace(-2, 16, 19)
    a = TitrationModel(full, cfg).curves(ph)(p)
    b = TitrationModel(native, cfg).curves(ph)(p)
    np.testing.assert_allclose(a.protonated, b.protonated, atol=1e-12)
    np.testing.assert_allclose(a.total_charge, b.total_charge, atol=1e-12)


def test_restricted_cache_rejects_and_masks_disallowed_mass(topology):
    cache = build_cache(topology, build_candidates(topology), identities=native_identities(topology))
    model = TitrationModel(cache)
    model.validate_probabilities(np.asarray(model.native_probabilities))
    with pytest.raises(ValueError, match='excluded'):
        model.validate_probabilities(np.full((cache.n_residues, 20), 1/20))
    p = model.probabilities_from_logits(jnp.zeros((cache.n_residues, 20)))
    np.testing.assert_allclose(p, model.native_probabilities, atol=0)


def test_identity_mask_validation(topology):
    lib = build_candidates(topology)
    with pytest.raises(ValueError):
        build_cache(topology, lib, identities=np.zeros((topology.n_residues, 20), bool))
    with pytest.raises(ValueError):
        build_cache(topology, lib, identities=np.ones((topology.n_residues, 19), bool))


def test_jit_argument_readout_matches_and_does_not_embed_cache(topology):
    # Cache arrays must be jit arguments, not HLO constants (compile-memory OOM).
    cache = build_cache(topology, build_candidates(topology))
    model = TitrationModel(cache, ModelConfig(steps=32))
    p = model.native_probabilities
    ph = jnp.linspace(-2, 16, 7)
    from jaxpropka.model import curve_kernel
    direct = curve_kernel(model.arrays, p, ph, config=model.config)
    out = model.curves(np.asarray(ph))(p)
    np.testing.assert_allclose(out.protonated, direct.protonated, atol=0)
    hlo = model.curves(np.asarray(ph)).lower(p)
    assert hlo.compile().memory_analysis().argument_size_in_bytes >= sum(
        x.nbytes for x in jax.tree.leaves(model.arrays))*0.9


def test_two_identity_mask_matches_full_cache_at_mixed_sequence(topology):
    lib = build_candidates(topology)
    full = build_cache(topology, lib, dtype=np.float64)
    mask = native_identities(topology)
    ala = 0  # alphabet index of A
    free = ~np.asarray(topology.disulfide)
    mask[free, ala] = True
    restricted = build_cache(topology, lib, dtype=np.float64, identities=mask)
    assert restricted.volume.shape[-1] == 2
    rng = np.random.default_rng(1)
    w = rng.uniform(.2, .8, size=topology.n_residues)
    p = np.eye(20)[np.asarray(topology.native_index)]*w[:, None]
    p[:, ala] += 1-w
    p[~free] = np.eye(20)[np.asarray(topology.native_index)[~free]]
    p = jnp.asarray(p)
    cfg = ModelConfig(steps=256); ph = np.linspace(0, 14, 8)
    a = TitrationModel(full, cfg).curves(ph)(p)
    b = TitrationModel(restricted, cfg).curves(ph)(p)
    np.testing.assert_allclose(b.protonated, a.protonated, atol=1e-12)
    np.testing.assert_allclose(b.intrinsic_pka, a.intrinsic_pka, atol=1e-12)


def test_compact_cache_save_load_and_batching(topology, tmp_path):
    from jaxpropka.cache import StructureCache
    from jaxpropka.batching import pack_inputs
    from jaxpropka.model import curve_kernel
    cache = build_cache(topology, build_candidates(topology), dtype=np.float64,
                        identities=native_identities(topology))
    cache.save(tmp_path/'c.npz')
    loaded = StructureCache.load(tmp_path/'c.npz')
    np.testing.assert_array_equal(loaded.identity_columns, cache.identity_columns)
    assert loaded.fingerprint() == cache.fingerprint()
    p = one_hot(cache.native_index, np.float64)
    arrays, padded, n = pack_inputs(cache, p, bucket_multiple=(16, 8, 8))
    cfg = ModelConfig(steps=128); ph = jnp.linspace(0., 14., 5)
    batched = curve_kernel({k: jnp.asarray(v) for k, v in arrays.items()}, jnp.asarray(padded), ph, config=cfg)
    direct = TitrationModel(cache, cfg).curves(np.asarray(ph))(jnp.asarray(p))
    np.testing.assert_allclose(np.asarray(batched.protonated)[:, :n], direct.protonated, atol=1e-12)
