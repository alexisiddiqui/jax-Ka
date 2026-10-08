import numpy as np
import jax
import jax.numpy as jnp

from pkanet.model import PKPDB_PK_MOD, initialize, predict_pkpdb, predict_shift
from pkatrain.graph_pkmod_compare import (
    ExplicitShiftEngine, inverse_frequency_weights, shift_bin_indices,
)


def tiny_graph():
    return {
        "nodes": jnp.zeros((2, 24)), "node_mask": jnp.ones(2, bool),
        "neighbors": jnp.array([[0, 1], [0, 1]]),
        "edge": jnp.zeros((2, 2, 20)), "edge_mask": jnp.ones((2, 2), bool),
        "switch": jnp.ones((2, 2)), "query_residue": jnp.array([0, 1]),
        "query_group": jnp.array([0, 1]),
    }


def test_pkmod_absolute_output_is_exact_baseline_plus_shift():
    graph = tiny_graph()
    params = initialize(jax.random.PRNGKey(3))
    np.testing.assert_allclose(
        predict_pkpdb(params, graph),
        np.asarray(PKPDB_PK_MOD)[np.asarray(graph["query_group"])] + predict_shift(params, graph),
        rtol=1e-7,
    )


def test_explicit_shift_loss_has_zero_error_at_matching_shift():
    graph = tiny_graph()
    params = initialize(jax.random.PRNGKey(4))
    target = predict_pkpdb(params, graph)
    batch = jax.tree.map(lambda value: value[None], graph)
    engine = ExplicitShiftEngine(np.ones(4))
    state = engine.optimizer.init(params)
    _, _, loss = engine.audited_batch_update(
        params, state, batch, target[None], jnp.ones((1, 2), bool), jnp.ones(1, bool)
    )
    assert loss < 1e-12


def test_pkmod_bins_and_weights():
    labels = np.array([3.79, 4.29, 4.8, 6.74, 8.75], np.float32)
    groups = np.array([0, 0, 0, 2, 2])
    np.testing.assert_array_equal(shift_bin_indices(labels, groups), [0, 1, 2, 0, 3])
    counts = np.array([10, 5, 2, 1])
    weights = inverse_frequency_weights(counts)
    np.testing.assert_allclose(counts * weights, np.repeat(counts.sum() / 4, 4))
