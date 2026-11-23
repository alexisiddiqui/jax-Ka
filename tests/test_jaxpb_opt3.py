"""Scalar CPU/ref contract: arbitrary buckets/cardinalities, cancellation, exact counts."""
import json
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import jax
import numpy as np
from jaxpb import titrate as T


def test_cpu_ref():
    states = [T.toy([3, 2, 4], 3, 6.), T.toy([2], 7, 0.), T.toy([1], 9, 0.),
              T.toy([11, 2], 77, 10.), T.toy([1, 2], 12, 10.), T.toy([2, 1], 13, 10.)]
    assert len(states[0].pairs) and len(states[3].pairs)
    # Force pair trials with respectively na==oa and nb==ob on every proposal.
    for state in states[-2:]:
        state.pairs = np.array([[0, 1]], np.int32)
    keys = T.chain_keys(71, np.arange(len(states) * T.NPH)).reshape(len(states), T.NPH)
    rows = []
    original = T.mc_ref
    # Forbid a fallback: every case must really execute scalar chains.
    for Mp, dims in [(23, (7, 11)), (64, None), (128, None), (2048, None)]:
        B = T.pack_bucket(states, list(range(len(states))), Mp, 2, keys, S=len(states)+1, NpPp=dims)
        args = (T.PH_TERM,) + tuple(B[k] for k in ('g','occ','W','start','nst','pa','pb','meta','keys'))
        ref = np.asarray(original(*args))
        with patch.object(T, 'mc_ref', side_effect=AssertionError('CPU fallback')):
            cpu = T.mc_cpu(*args)
        assert ref.tobytes() == cpu.tobytes(), (Mp, np.sum(ref != cpu))
        assert not cpu[-1].any()
        rows.append(dict(Mp=Mp, counts_bitwise=True))
    out = T.titrate(states, backend='cpu', mc_sweeps=30000, seed=9, S=4)
    ref = T.titrate(states, backend='ref', mc_sweeps=30000, seed=9, S=4)
    for s, a, b in zip(states, out, ref):
        for k in ('prot', 'tot', 'curves', 'pka'):
            assert a[k].tobytes() == b[k].tobytes(), k
        assert np.max(np.abs(a['curves'] - T.brute(s))) < .05
    # Sensitive cancellation: subtract each site's identical picks before adding.
    e = np.array([[1e20, 1e20, 3., 1.], [3., 1., 1e20, 1e20],
                  [1e20, 1e20, -1e20, -1e20], [7., 2., 3., 1.]], np.float32)
    expected = (e[:, 0] - e[:, 1]) + (e[:, 2] - e[:, 3])
    got = np.asarray(jax.jit(jax.vmap(T._pair_energy))(*e.T))
    assert got.tobytes() == expected.tobytes()
    # Partial pH grids and empty batches have no scalar-path domain restriction.
    short = (T.PH_TERM[:3],) + tuple(B[k] if k != 'keys' else B[k][:, :3]
                                  for k in ('g','occ','W','start','nst','pa','pb','meta','keys'))
    assert T.mc_cpu(*short).tobytes() == np.asarray(original(*short)).tobytes()
    empty = T.pack_bucket(states, [], 64, 200, np.empty((0,T.NPH),np.int32), S=4)
    assert not T.mc_cpu(T.PH_TERM, *(empty[k] for k in ('g','occ','W','start','nst','pa','pb','meta','keys'))).any()
    assert T.cpu_available() and T.resolve_backend() == 'cpu'
    with patch.object(T, 'pallas_available', return_value=True):
        assert T.resolve_backend() == 'pallas'
    with patch.object(T, 'pallas_available', return_value=False), patch.object(jax, 'default_backend', return_value='gpu'):
        assert not T.cpu_available() and T.resolve_backend() == 'ref'
    return dict(passed=True, buckets=rows, single_site=True, brute_max=.05,
                nst11=True, cancellation_cases=4, forced_partial_pair_proposals=True, partial_ph=True, no_fallback=True)


if __name__ == '__main__':
    print(json.dumps(test_cpu_ref(), indent=2), flush=True)
