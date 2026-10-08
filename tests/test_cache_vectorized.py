"""The vectorized build_cache must reproduce the loop reference bitwise."""
from pathlib import Path
import numpy as np
import pytest
from jaxpropka.topology import load_topology
from jaxpropka.geometry import build_candidates
from jaxpropka.precompute import build_cache, build_cache_reference, native_identities

pytestmark = pytest.mark.integration
DATA = Path(__file__).parent/'data'
ARRAYS = ('env_neighbors', 'env_mask', 'volume', 'mass', 'hbond', 'local_hbond', 'bb_volume', 'bb_mass',
          'bb_hbond', 'reorganization', 'neighbors', 'pair_mask', 'coulomb_geometry', 'hb_donor', 'hb_reverse')


@pytest.fixture(autouse=True)
def real_biotite():
    return pytest.importorskip('biotite')


def structures():
    return sorted(p.name for p in DATA.glob('*.pdb'))


def assert_identical(a, b):
    for name in ARRAYS:
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape, name
        np.testing.assert_array_equal(x, y, err_msg=name)
    if a.identity_columns is None:
        assert b.identity_columns is None
    else:
        np.testing.assert_array_equal(a.identity_columns, b.identity_columns)
    assert a.fingerprint() == b.fingerprint()


@pytest.mark.parametrize('name', structures())
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('restricted', [False, True])
def test_vectorized_matches_reference(name, dtype, restricted):
    top = load_topology(DATA/name)
    lib = build_candidates(top, missing_sidechain='template')
    kwargs = dict(dtype=dtype, identities=native_identities(top) if restricted else None)
    assert_identical(build_cache_reference(top, lib, **kwargs), build_cache(top, lib, **kwargs))


@pytest.mark.parametrize('name', structures())
def test_chunk_size_does_not_change_results(name):
    top = load_topology(DATA/name)
    lib = build_candidates(top, missing_sidechain='template')
    assert_identical(build_cache(top, lib, chunk=3), build_cache(top, lib, chunk=100000))


def test_two_identity_mask_matches_reference():
    top = load_topology(DATA/'two_chains.pdb'); lib = build_candidates(top)
    mask = native_identities(top); mask[:, 0] = True; mask[:, 15] = True   # + Ala, Ser (neutral)
    assert_identical(build_cache_reference(top, lib, identities=mask), build_cache(top, lib, identities=mask))
