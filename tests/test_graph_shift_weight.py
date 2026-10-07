import numpy as np
import jax
import jax.numpy as jnp

from pkatrain.graph_shift_weight import (
    ShiftWeightedEngine, inverse_frequency_weights, shift_bin_indices,
)
from pkatrain.graph_decay import ScheduledScalarEngine


def test_shift_bins_and_inverse_frequency_weights_equalize_bin_mass():
    labels = np.array([3.8, 4.3, 4.8, 6.5, 8.5], np.float32)
    groups = np.array([0, 0, 0, 2, 2])
    np.testing.assert_array_equal(shift_bin_indices(labels, groups), [0, 1, 2, 0, 3])
    counts = np.array([10, 5, 2, 1])
    weights = inverse_frequency_weights(counts)
    np.testing.assert_allclose(counts * weights, np.repeat(counts.sum() / 4, 4))
    assert np.isclose(np.average(weights, weights=counts), 1.0)


def test_uniform_weights_match_unweighted_engine_and_state_tree():
    def predict(params, graph):
        return graph["features"] @ params

    params = jnp.array([0.3, 0.7])
    graph = {
        "features": jnp.arange(24, dtype=jnp.float32).reshape(3, 4, 2) / 20,
        "query_group": jnp.zeros((3, 4), dtype=jnp.int32),
    }
    target = jnp.ones((3, 4)) * 3.8
    eligible = jnp.ones((3, 4), bool)
    valid = jnp.ones(3, bool)
    old = ScheduledScalarEngine(predict)
    new = ShiftWeightedEngine(predict, np.ones(4))
    old_state = old.optimizer.init(params)
    new_state = new.optimizer.init(params)
    assert str(jax.tree.structure(old_state)) == str(jax.tree.structure(new_state))
    po, so, lo = old.audited_batch_update(
        params, old_state, graph, target, eligible, valid, learning_rate=1e-3
    )
    pn, sn, ln = new.audited_batch_update(
        params, new_state, graph, target, eligible, valid, learning_rate=1e-3
    )
    np.testing.assert_allclose(lo, ln, rtol=1e-7)
    for left, right in zip(jax.tree.leaves((po, so)), jax.tree.leaves((pn, sn))):
        np.testing.assert_allclose(left, right, rtol=2e-6, atol=2e-7)
